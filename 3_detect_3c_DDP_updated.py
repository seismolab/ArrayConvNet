# %load detect_3c.py
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
            # print(f"Rank {self.rank} loaded {file_path}, X shape: {X.shape}, y shape: {y.shape}")
            for i in range(len(X)):
                yield X[i], y[i].long()
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
    
#========Training the model =====================================
def train_model(ds_train, ds_test,local_rank, rank):
    net = Net().to(local_rank) ##to gpu
    net = DDP(net, device_ids=[local_rank])
    # Cross Entropy Loss is used for classification
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr = 2e-5) #default 2e-5
    num_epoch = 80
    df=pd.DataFrame(columns=['Epoch', 'Test Loss', 'Training Loss'])
    losses = []
    accs = []
    # print(f"Rank {rank}: training start")
    for epoch in range(num_epoch):
        dist.barrier()
        epoch_start_time = time.time()
        total_train_size = 0
        total_test_size = 0
        running_loss = 0.0
        epoch_loss = 0.0
        test_loss = 0.0
        correct = 0
        total = 0
        for i, (_x, _y) in enumerate(ds_train):
            _x, _y = _x.to(local_rank,non_blocking=True), _y.to(local_rank,non_blocking=True) ### move data to gpu
            optimizer.zero_grad() 
            outputs = net(_x.unsqueeze(1))
            loss = criterion(outputs, _y)
            loss.backward() 
            optimizer.step()
            running_loss += loss.item()
            epoch_loss += loss.item() * _x.size(0)
            total_train_size += _x.size(0)
            if i % 10 == 9 and rank == 0:    # print every 10 mini-batches
                print('[%d, %5d] loss: %.3f' %
                (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0
        epoch_loss_tensor = torch.tensor(epoch_loss, device=local_rank)
        dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.SUM)
        train_size_tensor = torch.tensor(total_train_size, device=local_rank)
        dist.all_reduce(train_size_tensor, op=dist.ReduceOp.SUM)
        dist.barrier()
        with torch.no_grad():   
            net.eval()            
            for i, (_x, _y) in enumerate(ds_test):
                _x, _y = _x.to(local_rank), _y.to(local_rank) ### move data to gpu
                outputs = net(_x.unsqueeze(1))
                loss = criterion(outputs,_y)
                test_loss += loss.item() * _x.size(0)
                total_test_size += _x.size(0)
                _, predicted = torch.max(outputs.data,1)
                total += _y.size(0)
                correct += (predicted == _y).sum().item()
            test_loss_tensor = torch.tensor(test_loss, device=local_rank)
            dist.all_reduce(test_loss_tensor, op=dist.ReduceOp.SUM)
            test_size_tensor = torch.tensor(total_test_size, device=local_rank)
            dist.all_reduce(test_size_tensor, op=dist.ReduceOp.SUM)
            dist.barrier()
            net.train()
        dist.barrier()
        
        if rank == 0:
            epoch_loss = epoch_loss_tensor.item() / train_size_tensor.item()
            print(epoch_loss_tensor.item())
            print(train_size_tensor.item())
            test_loss = test_loss_tensor.item() / test_size_tensor.item()
            print(test_loss_tensor.item())
            print(test_size_tensor.item())
            print('[epoch %d] test loss: %.3f training loss: %.3f' %
                    (epoch + 1, test_loss, epoch_loss)) 
            temp_df = pd.DataFrame({'Epoch': ['%d'%(epoch+1)],'Test Loss': ['%.4f'%(test_loss)], 'Training Loss': ['%.4f'%(epoch_loss)]})
            df = pd.concat([df, temp_df], ignore_index=True)
            if epoch == num_epoch - 1:
                df.to_csv('../models/det_model_loss 2e-5dp0.3_b32.csv',index=False)
                print('Finished Training')
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        if rank == 0:
            print(f'Epoch {epoch + 1} completed in {epoch_duration:.2f} seconds.') 
    return net

def get_file_paths_with_sizes(folder_paths):
    """
    Args:
        folder_paths: 
    Returns:
        files_with_sizes: [(file_path, file_size), ...]
    """
    files_with_sizes = []
    for folder in folder_paths:
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            if file.endswith(".pt") and os.path.isfile(file_path):
                file_size = os.path.getsize(file_path)
                files_with_sizes.append((file_path, file_size))
    return files_with_sizes

def split_files_across_ranks(files_with_sizes, rank, world_size):
    """
    Args:
        files_with_sizes: [(file_path, file_size), ...]
        rank: 
        world_size: 
    Returns:
        rank_files: 
    """
    files_with_sizes = sorted(files_with_sizes, key=lambda x: x[1], reverse=True)
    rank_files = [[] for _ in range(world_size)]
    rank_sizes = [0] * world_size
    for file_path, file_size in files_with_sizes:
        target_rank = rank_sizes.index(min(rank_sizes))
        rank_files[target_rank].append(file_path)
        rank_sizes[target_rank] += file_size
    return rank_files[rank] 

def get_balanced_file_paths(folder_paths, rank, world_size):
    """
    Args:
        folder_paths: 
        rank: 
        world_size: 
    Returns:
        rank_files:
    """
    all_files_with_sizes = get_file_paths_with_sizes(folder_paths)
    rank_files = split_files_across_ranks(all_files_with_sizes, rank, world_size)
    return rank_files

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
    train_paths = ["../pts/det_train_big", "../pts/det_train_small"]
    test_paths = ["../pts/det_val_new_big", "../pts/det_val_new_small"]
    train_files_for_rank = get_balanced_file_paths(train_paths, rank, world_size)
    test_files_for_rank = get_balanced_file_paths(test_paths, rank, world_size)
    print(f"Rank {rank} will process the following {len(train_files_for_rank)} training files: {train_files_for_rank}")
    print(f"Rank {rank} will process the following {len(test_files_for_rank)} testing files: {test_files_for_rank}")
    ds_train = PrepareData(train_files_for_rank)
    ds_train.set_num_workers(32) 
    ds_train_loader = DataLoader(ds_train, batch_size=32, shuffle=False, num_workers=32, pin_memory=True,prefetch_factor=1,persistent_workers=True)
    ds_test = PrepareData(test_files_for_rank)
    ds_test.set_num_workers(32)
    ds_test_loader = DataLoader(ds_test, batch_size=32, shuffle=False, num_workers=32, pin_memory=True,prefetch_factor=1,persistent_workers=True)
    net = train_model(ds_train_loader, ds_test_loader,local_rank, rank)

    if rank == 0:
        pts_dir = "../models/model_final"
        os.makedirs(pts_dir, exist_ok=True)
        detect_net_path = os.path.join(pts_dir, 'det_final.pth')
        torch.save(net.state_dict(), detect_net_path)