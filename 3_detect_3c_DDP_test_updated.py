import time
from datetime import timedelta
import pandas as pd
import pdb
import os
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
                yield X[i], y[i].long()

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
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(torch.squeeze(x,1))))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool2(F.relu(self.conv3(x)))
        x = self.pool3(F.relu(self.conv4(x)))
        x = x.view(-1, 16*27*62)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def test_model(ds, ds_loader, net, local_rank, rank, world_size):
    net.to(local_rank)  # Move to GPU
    net.eval()
    criterion = nn.CrossEntropyLoss()
    batch_size = 32
    df = pd.DataFrame(columns=['Threshold', 'Accuracy', 'Precision', 'Recall', 'TPR', 'FPR', 'FScore'])
    thre = torch.arange(0, 1.01, 0.05).to(local_rank)
    thre_no = len(thre)
    true_p = torch.zeros(thre_no, device=local_rank)
    false_p = torch.zeros(thre_no, device=local_rank)
    false_n = torch.zeros(thre_no, device=local_rank)
    true_n = torch.zeros(thre_no, device=local_rank)
    all_outputs = []
    all_labels = []
    with torch.no_grad():
        for i, (_x, _y) in enumerate(ds_loader):
            _x, _y = _x.to(local_rank,non_blocking=True), _y.to(local_rank,non_blocking=True)  # Move data to GPU
            outputs = net(_x.unsqueeze(1))
            prob = F.softmax(outputs, 1)
            for j in range(thre_no):
                pred_threshold = (prob > thre[j]).float()
                predicted = pred_threshold[:, 1]
                for m in range(len(_y)):
                    if _y[m] == 1. and pred_threshold[m, 1] == 1.:
                        true_p[j] += 1.
                    if _y[m] == 0. and pred_threshold[m, 1] == 1.:
                        false_p[j] += 1.
                    if _y[m] == 1. and pred_threshold[m, 1] == 0.:
                        false_n[j] += 1.
                    if _y[m] == 0. and pred_threshold[m, 1] == 0.:
                        true_n[j] += 1.
            all_outputs.append(outputs)
            all_labels.append(_y)
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    test_no = len(all_labels)
    dist.barrier()
    dist.all_reduce(true_p, op=dist.ReduceOp.SUM)
    dist.all_reduce(false_p, op=dist.ReduceOp.SUM)
    dist.all_reduce(false_n, op=dist.ReduceOp.SUM)
    dist.all_reduce(true_n, op=dist.ReduceOp.SUM)
    dist.barrier()

    if rank == 0:
        print("Threshold, Accuracy, Precision, Recall, TPR, FPR, FScore")
        for j in range(thre_no):
            acc = 100 * (true_p[j] + true_n[j]) / (true_p[j] + true_n[j] + false_p[j] + false_n[j])
            if (true_p[j] + false_p[j]) > 0.:
                pre = 100 * true_p[j] / (true_p[j] + false_p[j])
            else:
                pre = 100 * torch.ones(1, device=local_rank)
            if (true_p[j] + false_n[j]) > 0.:
                rec = 100 * true_p[j] / (true_p[j] + false_n[j])
            else:
                rec = 100 * torch.ones(1, device=local_rank)
            tpr = 100 * true_p[j] / (true_p[j] + false_n[j])
            fpr = 100 * false_p[j] / (false_p[j] + true_n[j])
            fscore = 2 * pre * rec / (pre + rec)
            print(" %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f" % (thre[j].item(), acc.item(), pre.item(), rec.item(), tpr.item(), fpr.item(), fscore))
            temp_df = pd.DataFrame({'Threshold': ['%.2f' % thre[j].item()],
                                    'Accuracy': ['%.2f' % acc.item()],
                                    'Precision': ['%.2f' % pre.item()],
                                    'Recall': ['%.2f' % rec.item()],
                                    'TPR': ['%.2f' % tpr.item()],
                                    'FPR': ['%.2f' % fpr.item()],
                                    'FScore': ['%.2f' % fscore]})
            df = pd.concat([df, temp_df], ignore_index=True)
        df.to_csv('../models/test_det_2e-5dp0.3_b32.csv', index=False)

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
    dist.init_process_group("nccl", rank=rank, world_size=world_size,timeout=timedelta(seconds=7200))
    if rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    print(f"Running on Device {torch.cuda.get_device_name(local_rank)} (local_rank: {local_rank}, global_rank: {rank})")
    print(f"host: {os.uname().nodename}, rank: {rank}, local_rank: {local_rank}")
    test_paths = ["../pts/det_test_new_big","../pts/det_test_new_small"]
    test_files_for_rank = get_balanced_file_paths(test_paths, rank, world_size)
    print(f"Rank {rank} will process the following {len(test_files_for_rank)} testing files: {test_files_for_rank}")
    ds_test = PrepareData(test_files_for_rank)
    ds_test.set_num_workers(32)
    ds_test_loader = DataLoader(ds_test, batch_size=32, shuffle=False, num_workers=32, pin_memory=True,prefetch_factor=1,persistent_workers=True)
    net = Net()
    state_dict = torch.load('../models/model_final/det_final.pth')  # Load the trained model
    state_dict = dict_fix(state_dict)
    net.load_state_dict(state_dict)
    test_model(ds_test, ds_test_loader, net, local_rank, rank, world_size)