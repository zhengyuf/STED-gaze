from models import GazeHeadNet
import numpy as np
import os
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import logging
from core import DefaultConfig
from dataset import HDFDataset
from utils import script_init_common
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
config = DefaultConfig()
script_init_common()

batch_size = 64
num_data_loaders = 6
lr = 0.0001
train_loss_interval = 200
valid_loss_interval = 10000
save_interval = 10000
epochs = 200
test_subsample = 1.0
# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

################################################
network = faceNet()
# Transfer on the GPU before constructing and optimizer
if torch.cuda.device_count() > 1:
    logging.info('Using %d GPUs!' % torch.cuda.device_count())
    network = nn.DataParallel(network)
network = network.to(device)
optimizer = optim.Adam(network.parameters(), lr=lr, betas=(0.9, 0.95))
scheduler = StepLR(optimizer, step_size=50000, gamma=0.5)

from tensorboardX import SummaryWriter
tensorboard = SummaryWriter(logdir=config.save_path)
################################################
from checkpoints_manager import CheckpointsManager
saver = CheckpointsManager(network, config.save_path)
if not os.path.exists(config.save_path):
    os.makedirs(config.save_path)
step = saver.load_last_checkpoint()
print("initial step: {}".format(step))
################################################
import json
with open('./gazecapture_split.json', 'r') as f:
    all_gc_prefixes = json.load(f)
################################################
import losses
all_data = OrderedDict()

train_dataset = HDFDataset(hdf_file_path=config.gazecapture_file, prefixes=all_gc_prefixes['train'], num_labeled_samples=50000)
train_dataloader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True,
                              num_workers=num_data_loaders,
                              pin_memory=True,
                              )
all_data['gc/train'] = {'dataset': train_dataset, 'dataloader': train_dataloader}

for tag, hdf_file, is_bgr, prefixes in [
                                ('gc/test', config.gazecapture_file, False, all_gc_prefixes['test']),
                                ('mpi', config.mpiigaze_file, False, None),
                                ('columbia', config.columbia_file, True, None),
                                ('eyediap', config.eyediap_file, True, None),
                                ]:
    # Define dataset structure based on selected prefixes
    dataset = HDFDataset(hdf_file_path=hdf_file,
                         prefixes=prefixes)
    subsample = test_subsample
    if subsample < (1.0 - 1e-6):  # subsample if requested
        dataset = Subset(dataset, np.linspace(
            start=0, stop=len(dataset),
            num=int(subsample * len(dataset)),
            endpoint=False,
            dtype=np.uint32,
        ))
    all_data[tag] = {
        'dataset': dataset,
        'dataloader': DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_data_loaders,  # args.num_data_loaders,
                                 pin_memory=True,
                                 ),
    }

def send_data_dict_to_gpu(data):
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            data[k] = v.detach().to(device, non_blocking=True)
    return data

avg_train_loss = 0.0
avg_valid_loss = 0.0
avg_train_score_g = 0.0
avg_train_score_h = 0.0
avg_valid_score_g = 0.0
avg_valid_score_h = 0.0
num_training_batches = int(len(train_dataset)/batch_size)
for epoch in range(epochs):
    param_group = optimizer.param_groups[0]
    train_iter = iter(train_dataloader)
    for batch_i in range(num_training_batches):
        input_dict = next(train_iter)
        if step % save_interval == 0 and step != 0:
            saver.save_checkpoint(step)
        if step % train_loss_interval == 0 and step != 0:
            print("step: {}, train loss: {}, train score gaze: {}, train score head: {}, lr: {}".format(
                step, avg_train_loss / train_loss_interval, avg_train_score_g / train_loss_interval,
                avg_train_score_h / train_loss_interval, param_group['lr']))
            tensorboard.add_scalar('train/gaze', avg_train_score_g / train_loss_interval, step)
            tensorboard.add_scalar('train/head', avg_train_score_h / train_loss_interval, step)
            avg_train_loss = 0.0
            avg_train_score_g = 0.0
            avg_train_score_h = 0.0

        if step % valid_loss_interval == 0 and step != 0:
            network.eval()
            for param in network.parameters():
                param.requires_grad = False
            for tag in ['columbia', 'eyediap']:
                dataloader = all_data[tag]['dataloader']
                for i, input_dict_ in enumerate(dataloader):
                    input_dict_ = send_data_dict_to_gpu(input_dict_)
                    gaze_h, head_h = network(input_dict_['image_a'])
                    score_gaze = losses.gaze_angular_loss(input_dict_['gaze_a'], gaze_h)
                    score_head = losses.gaze_angular_loss(input_dict_['head_a'], head_h)
                    loss = score_gaze + score_head
                    avg_valid_loss += loss
                    avg_valid_score_g += score_gaze
                    avg_valid_score_h += score_head
                print("tag:{}, avg loss: {}, avg score gaze: {}, avg score head: {}"
                      .format(tag, avg_valid_loss / len(dataloader),avg_valid_score_g / len(dataloader),
                              avg_valid_score_h / len(dataloader)))
                tensorboard.add_scalar('test/'+tag+'/gaze', avg_valid_score_g / len(dataloader), step)
                tensorboard.add_scalar('test/'+tag+'/head', avg_valid_score_h / len(dataloader), step)

                avg_valid_loss = 0.0
                avg_valid_score_g = 0.0
                avg_valid_score_h = 0.0
            for param in network.parameters():
                param.requires_grad = True
        network.train()
        input_dict = send_data_dict_to_gpu(input_dict)
        gaze_h, head_h = network(input_dict['image_a'])
        optimizer.zero_grad()
        score_gaze = losses.gaze_angular_loss(input_dict['gaze_a'], gaze_h)
        score_head = losses.gaze_angular_loss(input_dict['head_a'], head_h)
        loss = score_head + score_gaze
        avg_train_loss += loss
        avg_train_score_g += score_gaze
        avg_train_score_h += score_head
        loss.backward()
        optimizer.step()
        scheduler.step(step)
        step += 1
saver.save_checkpoint(step)

