import os
import torch
import numpy as np
from torch.utils.data import Dataset

import cv2 as cv
import h5py
from core import DefaultConfig
config = DefaultConfig()

class HDFDataset2(Dataset):
    """Dataset from HDF5 archives formed of 'groups' of specific persons."""

    def __init__(self, hdf_file_path, use_aug=True, use_real=True,
                 prefixes=None,
                 ):
        assert os.path.isfile(hdf_file_path)
        self.hdf_path = hdf_file_path
        self.hdf = None  # h5py.File(hdf_file, 'r')
        with h5py.File(self.hdf_path, 'r', libver='latest', swmr=True) as h5f:
            hdf_keys = sorted(list(h5f.keys()))
            self.prefixes = hdf_keys if prefixes is None else prefixes

            # Pick all entries of person
            self.prefixes = [  # to address erroneous inputs
                k for k in self.prefixes if k in h5f
                and len(next(iter(h5f[k].values()))) > 0
            ]
            if use_aug and use_real:
                self.index_to_query = sum([
                    [(prefix, i) for i in range(len(next(iter(h5f[prefix].values()))))]
                    for prefix in self.prefixes], [])
            elif use_real:
                self.index_to_query = sum([
                    [(prefix, i) for i in range(len(next(iter(h5f[prefix].values())))) if h5f[prefix]['real'][i]]
                    for prefix in self.prefixes], [])
            else:
                assert use_aug
                self.index_to_query = sum([
                    [(prefix, i) for i in range(len(next(iter(h5f[prefix].values())))) if (not h5f[prefix]['real'][i])]
                    for prefix in self.prefixes], [])


    def __len__(self):
        return len(self.index_to_query)

    def close_hdf(self):
        if self.hdf is not None:
            self.hdf.close()
            self.hdf = None

    def preprocess_image(self, image):
        return image

    def preprocess_entry(self, entry):
        for key, val in entry.items():
            if isinstance(val, np.ndarray):
                entry[key] = torch.from_numpy(val.astype(np.float32))
            elif isinstance(val, int):
                # NOTE: maybe ints should be signed and 32-bits sometimes
                entry[key] = torch.tensor(val, dtype=torch.int16, requires_grad=False)
        return entry

    def __getitem__(self, idx):
        if self.hdf is None:  # Need to lazy-open this to avoid read error
            self.hdf = h5py.File(self.hdf_path, 'r', libver='latest', swmr=True)

        # Pick entry a and b from same person
        key_a, idx_a = self.index_to_query[idx]
        group_a = self.hdf[key_a]

        def retrieve(group, index):
            eyes = self.preprocess_image(group['image'][index])
            g = group['gaze'][index]
            h = group['head'][index]
            return eyes, g, h
        # Grab 1st (input) entry
        eyes_a, g_a, h_a = retrieve(group_a, idx_a)
        entry = {
            'key': key_a,
            'image_a': eyes_a,
            'gaze_a': g_a,
            'head_a': h_a,
        }
        return self.preprocess_entry(entry)

