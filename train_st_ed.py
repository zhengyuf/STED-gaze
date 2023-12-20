# --------------------------------------------------------
# Copyright (C) 2019 NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License (1-Way Commercial)
# Code written by Seonwook Park, Shalini De Mello
# --------------------------------------------------------
# Code modified by Yufeng Zheng
# --------------------------------------------------------
import numpy as np
from collections import OrderedDict
import gc
import json
import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import logging
from dataset import HDFDataset
from utils import save_images, worker_init_fn, send_data_dict_to_gpu, recover_images, def_test_list, RunningStatistics,\
    adjust_learning_rate, script_init_common, get_example_images, save_model, load_model
from core import DefaultConfig
from models import STED
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

# Set Configurations
config = DefaultConfig()
script_init_common()
# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import warnings
warnings.filterwarnings('ignore')

if not config.skip_training:
    if config.semi_supervised:
        assert config.num_labeled_samples != 0
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)
    # save configurations
    config.write_file_contents(config.save_path)

# Create the train and test datasets.
all_data = OrderedDict()
# Read GazeCapture train/val/test split
with open('./gazecapture_split.json', 'r') as f:
    all_gc_prefixes = json.load(f)
if not config.skip_training:
    # Define single training dataset
    train_prefixes = all_gc_prefixes['train']
    train_dataset = HDFDataset(hdf_file_path=config.gazecapture_file,
                               prefixes=train_prefixes,
                               is_bgr=False,
                               get_2nd_sample=True,
                               num_labeled_samples=config.num_labeled_samples if config.semi_supervised else None)
    # Define multiple val/test datasets for evaluation during training
    for tag, hdf_file, is_bgr, prefixes in [
        ('gc/val', config.gazecapture_file, False, all_gc_prefixes['val']),
        ('gc/test', config.gazecapture_file, False, all_gc_prefixes['test']),
        ('mpi', config.mpiigaze_file, False, None),
        ('columbia', config.columbia_file, True, None),
        ('eyediap', config.eyediap_file, True, None),
    ]:
        dataset = HDFDataset(hdf_file_path=hdf_file,
                             prefixes=prefixes,
                             is_bgr=is_bgr,
                             get_2nd_sample=True,
                             pick_at_least_per_person=2)
        if tag == 'gc/test':
            # test pair visualization:
            test_list = def_test_list()
            test_visualize = get_example_images(dataset, test_list)
            test_visualize = send_data_dict_to_gpu(test_visualize, device)

        subsample = config.test_subsample
        # subsample test sets if requested
        if subsample < (1.0 - 1e-6):
            dataset = Subset(dataset, np.linspace(
                start=0, stop=len(dataset),
                num=int(subsample * len(dataset)),
                endpoint=False,
                dtype=np.uint32,
            ))
        all_data[tag] = {
            'dataset': dataset,
            'dataloader': DataLoader(dataset,
                                     batch_size=config.eval_batch_size,
                                     shuffle=False,
                                     num_workers=config.num_data_loaders,  # args.num_data_loaders,
                                     pin_memory=True,
                                     ),
        }

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=int(config.batch_size),
                                  shuffle=True,
                                  drop_last=True,
                                  num_workers=config.num_data_loaders,
                                  pin_memory=True,
                                  )
    all_data['gc/train'] = {'dataset': train_dataset, 'dataloader': train_dataloader}

    # Print some stats.
    logging.info('')
    for tag, val in all_data.items():
        tag = '[%s]' % tag
        dataset = val['dataset']
        original_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
        num_people = len(original_dataset.prefixes)
        num_original_entries = len(original_dataset)
        logging.info('%10s full set size:           %7d' % (tag, num_original_entries))
        logging.info('%10s current set size:        %7d' % (tag, len(dataset)))
        logging.info('')

    # Have dataloader re-open HDF to avoid multi-processing related errors.
    for tag, data_dict in all_data.items():
        dataset = data_dict['dataset']
        original_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
        original_dataset.close_hdf()

# Create redirection network
network = STED().to(device)
# Load weights if available
from checkpoints_manager import CheckpointsManager

saver = CheckpointsManager(network.GazeHeadNet_eval, config.eval_gazenet_savepath)
_ = saver.load_last_checkpoint()
del saver

saver = CheckpointsManager(network.GazeHeadNet_train, config.gazenet_savepath)
_ = saver.load_last_checkpoint()
del saver

if config.load_step != 0:
    load_model(network, os.path.join(config.save_path, "checkpoints", str(config.load_step) + '.pt'))
    logging.info("Loaded checkpoints from step " + str(config.load_step))

# Transfer on the GPU before constructing and optimizer
if torch.cuda.device_count() > 1:
    logging.info('Using %d GPUs!' % torch.cuda.device_count())
    network.encoder = nn.DataParallel(network.encoder)
    network.decoder = nn.DataParallel(network.decoder)
    network.discriminator = nn.DataParallel(network.discriminator)
    network.GazeHeadNet_eval = nn.DataParallel(network.GazeHeadNet_eval)
    network.GazeHeadNet_train = nn.DataParallel(network.GazeHeadNet_train)
    network.lpips = nn.DataParallel(network.lpips)


def execute_training_step(current_step):
    global train_data_iterator
    try:
        input = next(train_data_iterator)
    except StopIteration:
        np.random.seed()  # Ensure randomness
        # Some cleanup
        train_data_iterator = None
        torch.cuda.empty_cache()
        gc.collect()
        # Restart!
        global train_dataloader
        train_data_iterator = iter(train_dataloader)
        input = next(train_data_iterator)
    input = send_data_dict_to_gpu(input, device)
    network.train()
    # forward + backward + optimize
    loss_dict, generated = network.optimize(input, current_step)

    # save training samples in tensorboard
    if config.use_tensorboard and current_step % config.save_freq_images == 0 and current_step != 0:
        for image_index in range(5):
            tensorboard.add_image('train/input_image',
                                  torch.clamp((input['image_a'][image_index] + 1) * (255.0 / 2.0), 0, 255).type(
                                      torch.cuda.ByteTensor), current_step)
            tensorboard.add_image('train/target_image',
                                  torch.clamp((input['image_b'][image_index] + 1) * (255.0 / 2.0), 0, 255).type(
                                      torch.cuda.ByteTensor), current_step)
            tensorboard.add_image('train/generated_image',
                                  torch.clamp((generated[image_index] + 1) * (255.0 / 2.0), 0, 255).type(
                                      torch.cuda.ByteTensor), current_step)
    # If doing multi-GPU training, just take an average
    for key, value in loss_dict.items():
        if value.dim() > 0:
            value = torch.mean(value)
            loss_dict[key] = value
    # Store values for logging later
    for key, value in loss_dict.items():
        loss_dict[key] = value.detach().cpu()
    for key, value in loss_dict.items():
        running_losses.add(key, value.numpy())


def execute_test(tag, data_dict):
    test_losses = RunningStatistics()
    with torch.no_grad():
        network.eval()
        for input_dict in data_dict['dataloader']:
            input_dict = send_data_dict_to_gpu(input_dict, device)
            output_dict, loss_dict = network(input_dict)
            for key, value in loss_dict.items():
                test_losses.add(key, value.detach().cpu().numpy())
    test_loss_means = test_losses.means()
    logging.info('Test Losses at [%7d] for %10s: %s' %
                 (current_step, '[' + tag + ']',
                  ', '.join(['%s: %.6f' % v for v in test_loss_means.items()])))
    if config.use_tensorboard:
        for k, v in test_loss_means.items():
            tensorboard.add_scalar('test/%s/%s' % (tag, k), v, current_step)


def execute_visualize(data):
    output_dict, losses_dict = network(test_visualize)
    keys = data['key'].cpu().numpy()
    for i in range(len(keys)):
        path = os.path.join(config.save_path, 'samples', str(keys[i]))
        if not os.path.exists(path):
            os.makedirs(path)
        cv2.imwrite(os.path.join(path, 'redirect_' + str(current_step) + '.png'),
                    recover_images(output_dict['image_b_hat'][i]))
        cv2.imwrite(os.path.join(path, 'redirect_all_' + str(current_step) + '.png'),
                    recover_images(output_dict['image_b_hat_all'][i]))
    walks = network.latent_walk(test_visualize)
    save_images(os.path.join(config.save_path, 'samples'), walks, keys, cycle=True)


if config.use_tensorboard and ((not config.skip_training) or config.compute_full_result):
    from tensorboardX import SummaryWriter
    tensorboard = SummaryWriter(logdir=config.save_path)
current_step = config.load_step

if not config.skip_training:
    logging.info('Training')
    running_losses = RunningStatistics()
    train_data_iterator = iter(train_dataloader)
    # main training loop
    for current_step in range(config.load_step, config.num_training_steps):
        # Save model
        if current_step % config.save_interval == 0 and current_step != config.load_step:
            save_model(network, current_step)
        # lr decay
        if (current_step % config.decay_steps == 0) or current_step == config.load_step:
            lr = adjust_learning_rate(network.optimizers, config.decay, int(current_step /config.decay_steps), config.lr)
            if config.use_tensorboard:
                tensorboard.add_scalar('train/lr', lr, current_step)
        # Testing loop: every specified iterations compute the test statistics
        if current_step % config.print_freq_test == 0 and current_step != 0:
            network.eval()
            network.clean_up()
            torch.cuda.empty_cache()
            for tag, data_dict in list(all_data.items())[:-1]:
                execute_test(tag, data_dict)
                # This might help with memory leaks
                torch.cuda.empty_cache()
        # Visualization loop
        if (current_step != 0 and current_step % config.save_freq_images == 0) or current_step == config.num_training_steps - 1:
            network.eval()
            torch.cuda.empty_cache()
            with torch.no_grad():
                # save redirected, style modified samples
                execute_visualize(test_visualize)
            torch.cuda.empty_cache()
        # Training step
        execute_training_step(current_step)
        # Print training loss
        if current_step != 0 and (current_step % config.print_freq_train == 0):
            running_loss_means = running_losses.means()
            logging.info('Losses at [%7d]: %s' %
                         (current_step,
                          ', '.join(['%s: %.5f' % v
                                     for v in running_loss_means.items()])))
            if config.use_tensorboard:
                for k, v in running_loss_means.items():
                    tensorboard.add_scalar('train/' + k, v, current_step)
            running_losses.reset()
    logging.info('Finished Training')
    # Save model parameters
    save_model(network, config.num_training_steps)
    del all_data
# Compute evaluation results on complete test sets
if config.compute_full_result:
    logging.info('Computing complete test results for final model...')
    all_data = OrderedDict()
    for tag, hdf_file, is_bgr, prefixes in [
        ('gc/val', config.gazecapture_file, False, all_gc_prefixes['val']),
        ('gc/test', config.gazecapture_file, False, all_gc_prefixes['test']),
        ('mpi', config.mpiigaze_file, False, None),
        ('columbia', config.columbia_file, True, None),
        ('eyediap', config.eyediap_file, True, None),
    ]:
        # Define dataset structure based on selected prefixes
        dataset = HDFDataset(hdf_file_path=hdf_file,
                             prefixes=prefixes,
                             is_bgr=is_bgr,
                             get_2nd_sample=True,
                             pick_at_least_per_person=2)
        if tag == 'gc/test':
            # test pair visualization:
            test_list = def_test_list()
            test_visualize = get_example_images(dataset, test_list)
            test_visualize = send_data_dict_to_gpu(test_visualize, device)
            with torch.no_grad():
                # save redirected, style modified samples
                execute_visualize(test_visualize)
        all_data[tag] = {
            'dataset': dataset,
            'dataloader': DataLoader(dataset,
                                     batch_size=config.eval_batch_size,
                                     shuffle=False,
                                     num_workers=config.num_data_loaders,
                                     pin_memory=True,
                                     worker_init_fn=worker_init_fn),
        }
    logging.info('')

    for tag, val in all_data.items():
        tag = '[%s]' % tag
        dataset = val['dataset']
        original_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
        num_entries = len(original_dataset)
        num_people = len(original_dataset.prefixes)
        logging.info('%10s set size:                %7d' % (tag, num_entries))
        logging.info('%10s num people:              %7d' % (tag, num_people))
        logging.info('')

    for tag, data_dict in all_data.items():
        dataset = data_dict['dataset']
        # Have dataloader re-open HDF to avoid multi-processing related errors.
        original_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
        original_dataset.close_hdf()

    network.eval()
    torch.cuda.empty_cache()
    for tag, data_dict in list(all_data.items()):
        execute_test(tag, data_dict)
    if config.use_tensorboard:
        tensorboard.close()
        del tensorboard
    # network.clean_up()
    torch.cuda.empty_cache()

# Use Redirector to create new training data
if config.store_redirect_dataset:
    train_tag = 'gc/train'
    train_prefixes = all_gc_prefixes['train']
    train_dataset = HDFDataset(hdf_file_path=config.gazecapture_file,
                               prefixes=train_prefixes,
                               num_labeled_samples=config.num_labeled_samples,
                               sample_target_label=True
                               )
    train_dataset.close_hdf()
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.eval_batch_size,
                                  shuffle=False,
                                  num_workers=config.num_data_loaders,
                                  pin_memory=True,
                                  )
    current_person_id = None
    current_person_data = {}
    ofpath = os.path.join(config.save_path, 'Redirected_samples.h5')
    ofdir = os.path.dirname(ofpath)
    if not os.path.isdir(ofdir):
        os.makedirs(ofdir)
    import h5py

    h5f = h5py.File(ofpath, 'w')

    def store_person_predictions():
        global current_person_data
        if len(current_person_data) > 0:
            g = h5f.create_group(current_person_id)
            for key, data in current_person_data.items():
                g.create_dataset(key, data=data, chunks=tuple([1] + list(np.asarray(data).shape[1:])),
                                 compression='lzf', dtype=
                                 np.float32)
        current_person_data = {}

    with torch.no_grad():
        np.random.seed()
        num_batches = int(np.ceil(len(train_dataset) / config.eval_batch_size))
        for i, input_dict in enumerate(train_dataloader):
            batch_size = input_dict['image_a'].shape[0]
            input_dict = send_data_dict_to_gpu(input_dict, device)
            output_dict = network.redirect(input_dict)
            zipped_data = zip(
                input_dict['key'],
                input_dict['image_a'].cpu().numpy().astype(np.float32),
                input_dict['gaze_a'].cpu().numpy().astype(np.float32),
                input_dict['head_a'].cpu().numpy().astype(np.float32),
                output_dict['image_b_hat_r'].cpu().numpy().astype(np.float32),
                input_dict['gaze_b_r'].cpu().numpy().astype(np.float32),
                input_dict['head_b_r'].cpu().numpy().astype(np.float32)
            )

            for (person_id, image_a, gaze_a, head_a, image_b_r, gaze_b_r, head_b_r) in zipped_data:
                # Store predictions if moved on to next person
                if person_id != current_person_id:
                    store_person_predictions()
                    current_person_id = person_id
                # Now write it
                to_write = {
                    'real': True,
                    'gaze': gaze_a,
                    'head': head_a,
                    'image': image_a,
                }
                for k, v in to_write.items():
                    if k not in current_person_data:
                        current_person_data[k] = []
                    current_person_data[k].append(v)

                to_write = {
                    'real': False,
                    'gaze': gaze_b_r,
                    'head': head_b_r,
                    'image': image_b_r,
                }
                for k, v in to_write.items():
                    current_person_data[k].append(v)

            logging.info('processed batch [%04d/%04d] with %d entries.' %
                         (i + 1, num_batches, len(next(iter(input_dict.values())))))
        store_person_predictions()
    logging.info('Completed processing')
    logging.info('Done')
    del train_dataset, train_dataloader
