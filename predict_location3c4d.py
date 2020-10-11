import pandas as pd
import numpy as np
from obspy import read
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
import pdb
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import geopy.distance
from sklearn.model_selection import train_test_split
#from mpl_toolkits.basemap import Basemap
from geopy import distance

# Custom data pre-processor to transform X and y from numpy arrays to torch tensors
class PrepareData(Dataset):
	def __init__(self, path):
		self.X, self.y = torch.load(path)

	def __len__(self):
		return self.X.shape[0]

	def __getitem__(self, idx):
		return self.X[idx], self.y[idx]


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # conv2d takes C_in, C_out, kernel size, stride, padding
        # input array size is (27,8000)
        #self.conv1 = nn.Conv2d(1, 8, 3, stride = 1, padding=1)
        # input array (3,55,2000) - 06-22-2020
        # input array (3,55,2500) - 09-02-2020
        self.conv1 = nn.Conv2d(3, 4, (1,9), stride = 1, padding=(0,4))
        #self.pool = nn.MaxPool2d((1,2), stride=(1,2))
        self.pool1 = nn.MaxPool2d((1,5), stride=(1,5))
        self.conv2 = nn.Conv2d(4, 4, (5,3), stride = 1, padding=(2,1))
        self.pool2 = nn.MaxPool2d((1,2), stride=(1,2))
        self.conv3 = nn.Conv2d(4, 8, (5,3), stride = 1, padding=(2,1))

        # for sample size of 2000
        # self.fc1 = nn.Linear(8*55*100, 128) 
        # for sample size of 2500 / 5 / 2 / 2 = 125;
        self.fc1 = nn.Linear(8*55*125, 128)
        #self.fc1 = nn.Linear(4*55*100, 128)
        #self.fc2 = nn.Linear(128, 3)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        #print(x.shape)
        x = self.pool1(F.relu(self.conv1(torch.squeeze(x,1))))
        #x = self.pool1(F.relu(self.conv1(x)))
        # print(x.shape)
        x = self.pool2(F.relu(self.conv2(x)))
        # print(x.shape)
        x = self.pool2(F.relu(self.conv3(x)))
        # print(x.shape)
        #x = self.pool2(F.relu(self.conv4(x)))
        # print(x.shape)
        # x = x.view(-1, 8*55*100)
        x = x.view(-1, 8*55*125)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Load trace data given a SAC file
def load_data(filename, see_stats=False, bandpass=True):
    st = read(filename)
    tr = st[0]
    if bandpass:
        tr.filter(type='bandpass', freqmin=5.0, freqmax=40.0)
    # tr.normalize()
    tr.taper(0.2)
    if see_stats:
        print(tr.stats)
        tr.plot()
    return tr

# Find all event directories     
def find_all_events(path):
    dirs = []
    for r, d, f in os.walk(path):
        for name in d:
            dirs.append(os.path.join(r, name))
    return dirs

def find_all_SAC(path):
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if '.SAC' in file:
                files.append(os.path.join(r, file))
    return files

def get_station_loc(path):
	SAC_filenames = find_all_SAC(path)
    # sample_size = 8000
	station_location = {}
	for filename in SAC_filenames:
		tr = load_data(filename)
		# pdb.set_trace()
		if tr.stats.station not in station_location:
			station_location[tr.stats.station]= (tr.stats.sac['stla'], tr.stats.sac['stlo'])
	south_to_north_stations = sorted(station_location, key=station_location.__getitem__)
	# print(station_location)
	return south_to_north_stations, station_location

def get_event(path, station_no, showPlots=False):
	sample_size = 8000
	event_array = torch.zeros(station_no,sample_size)
	sorted_event_array = torch.zeros(station_no,sample_size)
	max_amp_idx = []
	max_amp = []
	i=0
	for r, d, f in os.walk(path):
		if f == []:
			return []
		for filename in f:
			tr = load_data(os.path.join(r,filename), False)
			if len(tr.data) < sample_size:
				print('ERROR '+filename+' '+str(len(tr.data)))
			# only get vertical component
			elif tr.stats['channel'] == 'HHZ':
				# pdb.set_trace()
				# idx = sorted_stations.index(tr.stats.station)
				# print(idx)
				event_array[i,:] = torch.from_numpy(tr.data[:sample_size])
				max_amp_idx.append(np.argmax(abs(event_array[i,:])))
				max_amp.append(max(abs(event_array[i,:])))
				i+=1
				if i > station_no:
					break
	# ensure all values between 0 and 1
	event_array = event_array / max(max_amp)
	# sort traces in order of when their maximum amplitude arrives
	idx = np.argsort(max_amp_idx)
	# any stations not included (out of 28) will be the last row of zeros
	sorted_event_array[:i,:] = event_array[idx,:]
	# print(sorted_event_array.shape)

	if (showPlots):
		fig, axs = plt.subplots(station_no, sharey="col")
		fig.suptitle(path)	
		for i in range(station_no):
			axs[i].plot(sorted_event_array[i,:])
		plt.show()
		# plt.savefig("test")
	return event_array	

def separate_stations(path):
    SAC_filenames = find_all_SAC(path)
    sample_size = 8000
    trace_data = {}
    for filename in SAC_filenames:
        tr = load_data(filename, False)
        if len(tr.data) < sample_size:
            print('ERROR '+filename+' '+str(len(tr.data)))
        else:
            if tr.stats.station not in trace_data:
                trace_data[tr.stats.station]= np.expand_dims(tr.data[0:sample_size],axis=0)
            else:
                trace_data[tr.stats.station]=np.append(trace_data[tr.stats.station],\
                                                       np.expand_dims(tr.data[0:sample_size],axis=0),\
                                                       axis=0)
    return trace_data

def train_model(ds_train, ds_test):
	net = Net()
	# Cross Entropy Loss is used for classification
	# criterion = nn.CrossEntropyLoss()
	criterion = nn.MSELoss()
	#criterion = nn.SmoothL1Loss()
	#optimizer = optim.AdamW(net.parameters(), lr = 1e-4)
	optimizer = optim.AdamW(net.parameters(), lr = 5e-5)
	# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
	num_epoch = 80

	losses = []
	accs = []
	for epoch in range(num_epoch):  # loop over the dataset multiple times
		running_loss = 0.0
		epoch_loss = 0.0
		for i, (_x, _y) in enumerate(ds_train):

			optimizer.zero_grad() # zero the gradients on each pass before the update

			#========forward pass=====================================
			outputs = net(_x.unsqueeze(1))
			loss = criterion(outputs, _y)
			# acc = tr.eq(outputs.round(), _y).float().mean() # accuracy
			# print(loss.item())

			#=======backward pass=====================================
			loss.backward() # backpropagate the loss through the model
			optimizer.step() # update the gradients w.r.t the loss

			running_loss += loss.item()
			epoch_loss += loss.item()
			if i % 10 == 9:    # print running_loss for every 10 mini-batches
				print('[%d, %5d] loss: %.4f' %
				(epoch + 1, i + 1, running_loss / 10))
				running_loss = 0.0
		
		test_loss = 0.0
		with torch.no_grad():
			for i, (_x, _y) in enumerate(ds_test):
				outputs = net(_x.unsqueeze(1))
				loss = criterion(outputs,_y)
				test_loss += loss.item()
		print('[epoch %d] test loss: %.4f training loss: %.4f' %
				(epoch + 1, test_loss / len(ds_test), epoch_loss / len(ds_train))) 
	
	print('Finished Training')
	return net

def dist_list(true, predicted):
	dist_list = np.zeros((predicted.shape[0]))
	for i in range(predicted.shape[0]):
		origin = (true[i,0], true[i,1])
		dest = (predicted[i,0], predicted[i,1])
		dist_list[i] = distance.distance(origin, dest).km
	return dist_list

def test_model(ds,ds_loader, net):
	criterion = nn.MSELoss()
	#criterion = nn.SmoothL1Loss()
	# loss = 0
	test_no = len(ds)
	#batch_size=5
	#batch_size=8
	batch_size=32
	print(test_no)
	#y_hat = np.zeros((test_no,3))
	#y_ori = np.zeros((test_no,3))
	y_hat = np.zeros((test_no,4))
	y_ori = np.zeros((test_no,4))
	accurate = 0
	with torch.no_grad():
		for i, (_x, _y) in enumerate(ds_loader):
			#print(_x.shape, _y.shape)
			outputs = net(_x.unsqueeze(1))
			loss = criterion(outputs,_y)
			#print(_y.numpy(),outputs)
			y_hat[batch_size*i:batch_size*(i+1),:] = outputs
			y_ori[batch_size*i:batch_size*(i+1),:] = _y

	fig = plt.figure()
	#ax = fig.gca(projection='3d')
	ax = fig.add_subplot(111, projection='3d')

	# ds.y does not match with y_hat as ds.y is shuffled 
	#dist = dist_list(ds.y.numpy(), y_hat)
	#dist = dist_list(y_ori, y_hat)
	#print(dist)
	#print(ds.y.numpy(), y_ori)
	#print(np.mean(dist), np.std(dist))
	#ax.scatter(ds.y[:,0], ds.y[:,1], ds.y[:,2], marker='o',label="true")
	#ax.scatter(y_hat[:,0], y_hat[:,1], y_hat[:,2], marker='^', label="predicted")
	# account for the shift and scaling (see generate_event_array3c.py)
	event_coordref = (19.5,-155.5,0,0)
	# event_norm = (1.0,1.0,50.0,15)
	event_norm = (1.0,1.0,50.0,10.)
	# first rescale and then shift the earthquake source values, reverse the steps in generate*.py
	ds.y = np.multiply(ds.y,event_norm)
	y_hat = np.multiply(y_hat,event_norm)
	y_ori = np.multiply(y_ori,event_norm)
	ds.y = np.add(ds.y,event_coordref)
	y_hat = np.add(y_hat,event_coordref)
	y_ori = np.add(y_ori,event_coordref)

	for k in range(len(y_hat)):
		if y_hat[k,2] < 0:
			y_hat[k,2] = 0 # no earthquake in the air (note: HVO catalog ignore topo)

	dist = dist_list(y_ori, y_hat)
	for k in range(len(dist)):
		#print(dist[k],y_ori[k,2])
		print(y_ori[k,0],y_ori[k,1],y_ori[k,2],y_hat[k,0],y_hat[k,1],y_hat[k,2],y_ori[k,3],y_hat[k,3])
	dep_diff = y_ori[:,2] - y_hat[:,2]
	print(np.mean(dist), np.std(dist), np.mean(abs(dep_diff)), np.std(dep_diff))

	ax.scatter(ds.y[:,0], ds.y[:,1], -ds.y[:,2], marker='o',label="HVO")
	ax.scatter(y_hat[:,0], y_hat[:,1], -y_hat[:,2], marker='^', label="predicted")
	#for i in range(test_no):
		#lat = np.array((ds.y[i,0],y_hat[i,0]))
		#lon = np.array((ds.y[i,1],y_hat[i,1]))
		#dep = np.array((ds.y[i,2],y_hat[i,2]))
		#ax.plot3d(lat,lon,dep,'r-')
	ax.set_xlim(18, 20.5)
	ax.set_ylim(-154, -157)
	ax.set_xlabel('Latitude')
	ax.set_ylabel('Longitude')
	ax.set_zlabel('Depth')
	plt.legend(loc='upper left')
	plt.show()
	# print('MSE of the network on the test set: %d %%' % (loss))

def get_coords(dirname):
	earthquake_df = pd.read_csv('./EQinfo/eqs_2017.csv')
	uniqueID = dirname[-14:]
	earthquake_df = earthquake_df.set_index(earthquake_df['time'].str[:19])
	match = uniqueID[0:4]+"-"+uniqueID[4:6]+"-"+uniqueID[6:8]+"T"+uniqueID[8:10]+":"+uniqueID[10:12]+":"+uniqueID[12:]
	return torch.tensor([earthquake_df['latitude'][match], earthquake_df['longitude'][match], earthquake_df['depth'][match]])

if __name__ == "__main__":

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	# # # pdb.set_trace()
	#ds_train = PrepareData(path = 'train_data.pt')
	# absolute
	#ds_train = PrepareData(path = '/Volumes/jd/data.hawaii/pts/train_data3c4d_Abs.pt')
	# Not absolute
	ds_train = PrepareData(path = '/Volumes/jd/data.hawaii/pts/train_data3c4d_NotAbs2017Mcut50sLin.pt')
	#ds_train_loader = DataLoader(ds_train, batch_size=5, shuffle=True)
	ds_train_loader = DataLoader(ds_train, batch_size=32, shuffle=True)

	#ds_test = PrepareData(path = 'test_data.pt')
	# absolute
	#ds_test = PrepareData(path = '/Volumes/jd/data.hawaii/pts/test_data3c4d_Abs.pt')
	# NOT absolute
	ds_test = PrepareData(path = '/Volumes/jd/data.hawaii/pts/test_data3c4d_NotAbs2017Mcut50sLin.pt')

	#ds_test_loader = DataLoader(ds_test, batch_size=5, shuffle=True)
	ds_test_loader = DataLoader(ds_test, batch_size=32, shuffle=True)
	net = train_model(ds_train_loader, ds_test_loader)

	PATH = './SeisConvNetLoc_NotAbs2017Mcut50sLin.pth'
	torch.save(net.state_dict(), PATH)

	# net = torch.load('./SeisConvNet.pth')
	accuracy = test_model(ds_test, ds_test_loader, net)



