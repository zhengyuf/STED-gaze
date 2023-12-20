# --------------------------------------------------------
# Copyright (C) 2019 NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License (1-Way Commercial)
# Code written by Seonwook Park, Shalini De Mello
# --------------------------------------------------------

# Code modified by Yufeng Zheng
import os
import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset

import cv2
import h5py
from core import DefaultConfig
import random
import logging

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
config = DefaultConfig()


class HDFDataset(Dataset):
    """Dataset from HDF5 archives formed of 'groups' of specific persons."""

    def __init__(self, hdf_file_path,
                 prefixes=None,
                 is_bgr=False,
                 get_2nd_sample=False,
                 pick_at_least_per_person=None,
                 num_labeled_samples=None,
                 sample_target_label=False,
                 ):
        assert os.path.isfile(hdf_file_path)
        self.get_2nd_sample = get_2nd_sample
        self.hdf_path = hdf_file_path
        self.hdf = None
        self.is_bgr = is_bgr
        self.sample_target_label = sample_target_label

        with h5py.File(self.hdf_path, 'r', libver='latest', swmr=True) as h5f:
            hdf_keys = sorted(list(h5f.keys()))
            if prefixes is None:
                self.prefixes = hdf_keys
            else:
                self.prefixes = [k for k in prefixes if k in h5f]
            if pick_at_least_per_person is not None:
                self.prefixes = [k for k in self.prefixes if k in h5f and len(next(iter(h5f[k].values()))) >=
                            pick_at_least_per_person]
            self.index_to_query = sum([[(prefix, i) for i in range(len(next(iter(h5f[prefix].values()))))]
                                       for prefix in self.prefixes], [])
            if num_labeled_samples is not None:
                # randomly pick labeled samples for semi-supervised training
                ra = list(range(len(self.index_to_query)))
                random.seed(0)
                random.shuffle(ra)
                # Make sure that the ordering is the same
                # assert ra[:3] == [744240, 1006758, 1308368]
                ra = ra[:num_labeled_samples]
                list.sort(ra)
                self.index_to_query = [self.index_to_query[i] for i in ra]

            # calculate kernel density of gaze and head pose, for generating new redirected samples
            if sample_target_label:
                if num_labeled_samples is not None:
                    sample = []
                    old_key = -1
                    for key, idx in self.index_to_query:
                        if old_key != key:
                            group = h5f[key]
                        sample.append(group['labels'][idx, :4])
                    sample = np.asarray(sample, dtype=np.float32)
                else:
                    # can calculate faster if load by group
                    sample = None
                    for key in self.prefixes:
                        group = h5f[key]
                        if sample is None:
                            sample = group['labels'][:, :4]
                        else:
                            sample = np.concatenate([sample, group['labels'][:, :4]], axis=0)
                sample = sample.transpose()
                from scipy import stats
                self.kernel = stats.gaussian_kde(sample)
                logging.info("Finished calculating kernel density for gaze and head angles")
                # Sample new gaze and head pose angles
                new_samples = self.kernel.resample(len(self.index_to_query))
                self.gaze = new_samples[:2, :].transpose()
                self.head = new_samples[2:4, :].transpose()
                self.index_of_sample = 0

    def __len__(self):
        return len(self.index_to_query)

    def close_hdf(self):
        if self.hdf is not None:
            self.hdf.close()
            self.hdf = None

    def preprocess_image(self, image):
        if self.is_bgr:
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        else:
            ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        image = np.transpose(image, [2, 0, 1])  # Colour image
        image = 2.0 * image / 255.0 - 1
        return image

    def preprocess_entry(self, entry):
        for key, val in entry.items():
            if isinstance(val, np.ndarray):
                entry[key] = torch.from_numpy(val.astype(np.float32))
            elif isinstance(val, int):
                # NOTE: maybe ints should be signed and 32-bits sometimes
                entry[key] = torch.tensor(val, dtype=torch.long, requires_grad=False)
        return entry

    def __getitem__(self, idx):
        if self.hdf is None:  # Need to lazy-open this to avoid read error
            self.hdf = h5py.File(self.hdf_path, 'r', libver='latest', swmr=True)

        # Pick entry a and b from same person
        key_a, idx_a = self.index_to_query[idx]
        group_a = self.hdf[key_a]
        group_b = group_a

        def retrieve(group, index):
            eyes = self.preprocess_image(group['pixels'][index, :])
            g = group['labels'][index, :2]
            h = group['labels'][index, 2:4]
            return eyes, g, h
        # Grab 1st (input) entry
        eyes_a, g_a, h_a = retrieve(group_a, idx_a)
        entry = {
            'key': key_a,
            'image_a': eyes_a,
            'gaze_a': g_a,
            'head_a': h_a,
        }
        if self.sample_target_label:
            entry['gaze_b_r'] = self.gaze[self.index_of_sample]
            entry['head_b_r'] = self.head[self.index_of_sample]
            self.index_of_sample += 1
        if self.get_2nd_sample:
            all_indices = list(range(len(next(iter(group_a.values())))))
            if len(all_indices) == 1:
                # If there is only one sample for this person, just return the same sample.
                idx_b = idx_a
            else:
                all_indices_but_a = np.delete(all_indices, idx_a)
                idx_b = np.random.choice(all_indices_but_a)
            # Grab 2nd entry from same person
            eyes_b, g_b, h_b = retrieve(group_b, idx_b)
            entry['image_b'] = eyes_b
            entry['gaze_b'] = g_b
            entry['head_b'] = h_b
        return self.preprocess_entry(entry)

