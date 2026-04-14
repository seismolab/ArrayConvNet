# %load predict_location3c4d.py

### train and test the localization model ####
import time
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pdb
import torch
import math
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader,IterableDataset,get_worker_info
from torch.profiler import profile
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from socket import gethostname
import geopy.distance
from geopy import distance

#========Preparing datasets for PyTorch DataLoader=====================================
class PrepareData(IterableDataset):
    def __init__(self, file_paths, batch_file_size=1):
        self.file_paths = file_paths
        self.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        self.num_workers = 1
        self.batch_file_size = batch_file_size

    def set_num_workers(self, num_workers):
        self.num_workers = num_workers    

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None: 
            iter_start = 0
            iter_end = len(self.file_paths)
        else: 
            per_worker = int(math.ceil(len(self.file_paths) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.file_paths))
            
        for file_path in self.file_paths[iter_start:iter_end]:
            X, y = torch.load(file_path)
            X = torch.nan_to_num(X, nan=0.0)
            for i in range(len(X)):
                yield X[i], y[i].float()

#========Network architecture=====================================
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # conv2d takes C_in, C_out, kernel size, stride, padding
        # input array (3,55,2500)
        self.conv1 = nn.Conv2d(3, 4, (1,9), stride=1, padding=(0,4))
        self.pool1 = nn.MaxPool2d((1,5), stride=(1,5))
        self.conv2 = nn.Conv2d(4, 4, (5,3), stride=1, padding=(2,1))
        self.pool2 = nn.MaxPool2d((1,2), stride=(1,2))
        self.conv3 = nn.Conv2d(4, 8, (5,3), stride=1, padding=(2,1))
        self.conv4 = nn.Conv2d(8, 16, (3,3), stride=1, padding=(1,1))
        self.pool3 = nn.MaxPool2d((2,2), stride=(2,2))
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(16*27*62, 128)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(torch.squeeze(x,1))))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool2(F.relu(self.conv3(x)))
        x = self.pool3(F.relu(self.conv4(x)))
        x = x.view(-1, 16*27*62)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def dist_list(true, predicted):
    dist_list = np.zeros((predicted.shape[0]))
    for i in range(predicted.shape[0]):
        origin = (true[i,0], true[i,1])
        dest = (predicted[i,0], predicted[i,1])
        dist_list[i] = distance.distance(origin, dest).km
    return dist_list

#========Testing the model =====================================
def test_model(ds_loader, net, local_rank, rank, world_size):
    net.to(local_rank)  # Move to GPU
    net.eval()
    criterion = nn.MSELoss()
    batch_size = 32
    all_outputs = []
    all_labels = []
    with torch.no_grad():
        for i, (_x, _y) in enumerate(ds_loader):
            _x, _y = _x.to(local_rank), _y.to(local_rank)  # Move data to GPU
            outputs = net(_x.unsqueeze(1))
            all_outputs.append(outputs)
            all_labels.append(_y)
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    gathered_outputs = [torch.zeros_like(all_outputs) for _ in range(world_size)]
    gathered_labels = [torch.zeros_like(all_labels) for _ in range(world_size)]
    dist.all_gather(gathered_outputs, all_outputs)
    dist.all_gather(gathered_labels, all_labels)

    if rank == 0:
        gathered_outputs = torch.cat(gathered_outputs)
        gathered_labels = torch.cat(gathered_labels)
        y_hat = gathered_outputs.cpu().numpy()
        y_ori = gathered_labels.cpu().numpy()
        event_coordref = (19.5, -155.5, 0, 0)
        event_norm = (1.0, 1.0, 50.0, 10.0)
        y_hat = np.multiply(y_hat, event_norm)
        y_ori = np.multiply(y_ori, event_norm)
        y_hat = np.add(y_hat, event_coordref)
        y_ori = np.add(y_ori, event_coordref)
        for k in range(len(y_hat)):
            if y_hat[k, 2] < 0:
                y_hat[k, 2] = 0  # no earthquake in the air
        dista = dist_list(y_ori, y_hat)
        for k in range(len(dista)):
            print(y_ori[k, 0], y_ori[k, 1], y_ori[k, 2], y_hat[k, 0], y_hat[k, 1], y_hat[k, 2], y_ori[k, 3], y_hat[k, 3])
        dep_diff = y_ori[:, 2] - y_hat[:, 2]
        lat_diff = y_ori[:, 0] - y_hat[:, 0]
        avg_lat = (y_ori[:, 0] + y_hat[:, 0]) / 2.0
        lon_diff= y_ori[:, 1] - y_hat[:, 1]
        ns_distance = lat_diff * (np.pi / 180) * 6371
        we_distance = lon_diff * (np.pi / 180) * 6371 * np.cos(avg_lat * (np.pi / 180))
        time_diff = y_ori[:, 3] - y_hat[:, 3]
        print(np.mean(ns_distance),np.std(ns_distance),np.mean(we_distance),np.std(we_distance),np.mean(dista), np.std(dista), np.mean(dep_diff), np.std(dep_diff),np.mean(time_diff), np.std(time_diff))
        df = pd.DataFrame({'ns mean': [np.mean(ns_distance)],'ns std': [np.std(ns_distance)],'we mean': [np.mean(we_distance)],'we std': [np.std(we_distance)],
                           'mean dist': [np.mean(dista)], 'std dist': [np.std(dista)], 'mean abs dep diff': [np.mean(dep_diff)], 'std dep diff': [np.std(dep_diff)],
                           'mean abs time diff': [np.mean(time_diff)], 'std time diff': [np.std(time_diff)]})
        df.to_csv('../models/test_pred_loc_5e-5dp0.3_b32.csv', index=False)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(y_ori[:, 0], y_ori[:, 1], -y_ori[:, 2], marker='o', s=1, label="Catalog")
        ax.scatter(y_hat[:, 0], y_hat[:, 1], -y_hat[:, 2], marker='^', s=1, label="predicted")
        ax.set_xlim(18, 20.5)
        ax.set_ylim(-154, -157)
        ax.set_xlabel('Latitude')
        ax.set_ylabel('Longitude')
        ax.set_zlabel('Depth')
        plt.legend(loc='upper left')
        plt.savefig("{}/{}".format('../models', 'test_pred.png'), dpi=500, bbox_inches='tight')

def get_file_paths_with_sizes(folder_paths):
    files_with_sizes = []
    for folder in folder_paths:
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            if file.endswith(".pt") and os.path.isfile(file_path):
                file_size = os.path.getsize(file_path)  
                files_with_sizes.append((file_path, file_size))
    return files_with_sizes

def split_files_across_ranks(files_with_sizes, rank, world_size):
    files_with_sizes = sorted(files_with_sizes, key=lambda x: x[1], reverse=True)
    rank_files = [[] for _ in range(world_size)]
    rank_sizes = [0] * world_size
    for file_path, file_size in files_with_sizes:
        target_rank = rank_sizes.index(min(rank_sizes))
        rank_files[target_rank].append(file_path)
        rank_sizes[target_rank] += file_size
    return rank_files[rank] 

def get_balanced_file_paths(folder_paths, rank, world_size):
    all_files_with_sizes = get_file_paths_with_sizes(folder_paths)
    rank_files = split_files_across_ranks(all_files_with_sizes, rank, world_size)
    return rank_files

def dict_fix(state_dict):
    """Remove the 'module.' prefix from the keys in state_dict"""
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k 
        new_state_dict[name] = v
    return new_state_dict

if __name__ == "__main__":
    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ["SLURM_LOCALID"])
    gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    assert gpus_per_node == torch.cuda.device_count()
    print(f"Hello from rank {rank} of {world_size} on {gethostname()} where there are" \
      f" {gpus_per_node} allocated GPUs per node.", flush=True)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    if rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    print(f"Running on Device {torch.cuda.get_device_name(local_rank)} (local_rank: {local_rank}, global_rank: {rank})")
    print(f"host: {os.uname().nodename}, rank: {rank}, local_rank: {local_rank}")
    test_paths = ["../pts/loc_test_new_big","../pts/loc_test_new_small"]
    test_files_for_rank = get_balanced_file_paths(test_paths, rank, world_size)
    print(f"Rank {rank} will process the following {len(test_files_for_rank)} testing files: {test_files_for_rank}")
    ds_test = PrepareData(test_files_for_rank)
    ds_test.set_num_workers(32)
    ds_test_loader = DataLoader(ds_test, batch_size=32, shuffle=False, num_workers=32, pin_memory=True,prefetch_factor=1,persistent_workers=True)
    net = Net()
    state_dict = torch.load('../models/model_final/loc_final.pth')
    state_dict = dict_fix(state_dict)
    net.load_state_dict(state_dict)
    test_model(ds_test_loader, net, local_rank, rank, world_size)
    