import numpy as np
from obspy import read
import os
import matplotlib.pyplot as plt
import pdb
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from geopy import distance
import math

#========Network architecture for detection=====================================
class DetectNet(nn.Module):
    def __init__(self):
        super(DetectNet, self).__init__()
        # conv2d takes C_in, C_out, kernel size, stride, padding
        self.conv1 = nn.Conv2d(3, 4, (1,9), stride = 1, padding=(0,4))
        self.pool1 = nn.MaxPool2d((1,5), stride=(1,5))
        self.conv2 = nn.Conv2d(4, 4, (5,3), stride = 1, padding=(2,1))
        self.pool2 = nn.MaxPool2d((1,2), stride=(1,2))
        self.conv3 = nn.Conv2d(4, 8, (5,3), stride = 1, padding=(2,1))

        self.fc1 = nn.Linear(8*55*125, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool2(F.relu(self.conv3(x)))
        x = x.view(-1, 8*55*125)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#========Network architecture for prediction=====================================
class PredictNet(nn.Module):
    def __init__(self):
        super(PredictNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 4, (1,9), stride = 1, padding=(0,4))
        self.pool1 = nn.MaxPool2d((1,5), stride=(1,5))
        self.conv2 = nn.Conv2d(4, 4, (5,3), stride = 1, padding=(2,1))
        self.pool2 = nn.MaxPool2d((1,2), stride=(1,2))
        self.conv3 = nn.Conv2d(4, 8, (5,3), stride = 1, padding=(2,1))

        self.fc1 = nn.Linear(8*55*125, 128) 
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool2(F.relu(self.conv3(x)))
        x = x.view(-1, 8*55*125)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#========Predict=====================================
# Run prediction model over a window
def predict(window, predictNet):
	with torch.no_grad():
		outputs = predictNet(window.unsqueeze(0))
	return outputs

#========Detect=====================================
# Run detection model over a window
def detect(window, detectNet):
	criterion = nn.CrossEntropyLoss()
	with torch.no_grad():
		#print(window.unsqueeze(0).shape)
		outputs = detectNet(window.unsqueeze(0))
		# outputs is a vector of two elements from CrossEntropyLoss
		#print(outputs)
		_, detection = torch.max(outputs.data,1)
	#return detection, 
	return detection, outputs

#========Load raw trace data=====================================
# Load trace data given a SAC file
def load_data(filename, see_stats=False, bandpass=False):
    st = read(filename)
    tr = st[0]
    if bandpass:
        tr.filter(type='bandpass', freqmin=5.0, freqmax=40.0)
    tr.taper(0.2)
    if see_stats:
        print(tr.stats)
        tr.plot()
    return tr

#========Read a SAC file=====================================
# Load trace data given a SAC file
def read_sac(path):
	length = 4320000
	channel_size = 3
	station_no = 55
	event_array = torch.zeros(channel_size, station_no,length)
	ii = 0
	for r, d, f in os.walk(path):
		print(len(f))
		if f == []:
			print ("no files")
			return []

		# Load sac files in alphabetically by station
		for filename in sorted(f):
			#print(filename)
			i = ii // 3 # station number
			j = ii % 3 # channel
			tr = load_data(os.path.join(r,filename), False)
			if tr.stats['network'] == 'FG':
				print('FG')
			else:
				event_array[j,i,:] = torch.from_numpy(tr.data)
			ii +=1
	return event_array
 
#========Return all days with seismic traces=====================================
# Each directory corresponds to a day     
def find_all_days(path):
    dirs = []
    for r, d, f in os.walk(path):
        for name in d:
            dirs.append(os.path.join(r, name))
    return dirs

#========Parse the year and day=====================================
# Given a path, return the year and day
def parse(path):
	dir_name = path.split('/')[-1]
	year = dir_name.split('.')[0]
	day = dir_name.split('.')[1]
	return year, day

#========Run detection model over an entire day=====================================
# Detect earthquakes within a day
def daily_detection(day_path, detectNet, predictNet):
	sampling_rate = 50 # we collect 50 samples per second
	interval = 3
	window_n = 2500 # number of samples in the window
	n = 4320000	
	station_no = 55
	channels = 3
	idx = range(2*window_n, n, sampling_rate*interval) # skip the begining due to tapering effect

	saved_vars = []
	num_detected = 0

	continuous_sac = read_sac(day_path)
	skip = False
	skipTracker = 0
	prob_max = 0
	window_max = torch.zeros(channels,station_no,window_n)
	peak_med = 40
	peak_medmax = 40
	peak_ave = 40
	ttdiff = 40
	det_flag = 0
	prob_cutoff = 0.95
	prob_det = 0

	for i in idx:
		# ensure we get full length windows 
		if i+window_n > n:
			break
		if skip:
			skipTracker += 1
			# at 5 s interval
			if skipTracker == 2:
				skip = False
			continue
		window = continuous_sac[:,:,i:i+window_n] # get all channels, all stations within the specific time frame

		window_normalized = torch.zeros(channels, station_no,window_n)
		trace_max = np.amax(abs(np.array(window[:,:,:])),axis=2)
		max_amp_idx = np.ones(station_no) * window_n
		idx = np.ones(3) * window_n

		for k in reversed(range(channels)):
			for j in range(station_no):
				if trace_max[k,j] != 0:
					window_normalized[k,j,:] = window[k,j,:]/trace_max[k,j]
		for j in range(station_no):
			if trace_max[0,j] != 0:
				idx[0] = np.argmax(abs(window_normalized[0, j,:]))
				idx[1] = np.argmax(abs(window_normalized[1, j,:]))
				idx[2] = np.argmax(abs(window_normalized[2, j,:]))
				max_amp_idx[j] = np.median(idx)
			else:
				max_amp_idx[j] = np.argmax(abs(window_normalized[2, j,:]))

		peak_med = np.median(max_amp_idx)/50

		if peak_med > 7 and peak_med < 43:
			sort_idx = np.argsort(max_amp_idx) # sort by index of maximum amplitude
			sorted_window = abs(window_normalized[:,sort_idx,:])

			detection, outputs = detect(sorted_window, detectNet)

			prob_outputs = F.softmax(outputs,1)
			prob = prob_outputs[:,1].item()

			# If the probability of an earthquake occuring is greater than the designated threshold,
			# run the prediction model
			if prob > prob_cutoff:
				prediction = predict(window_normalized, predictNet)
				event_coordref = (19.5,-155.5,0,0)
				event_norm = (1.0,1.0,50.0,10.0)
				y_hat = torch.squeeze(prediction)
				y_hat = np.multiply(y_hat,event_norm)
				y_hat = np.add(y_hat,event_coordref)

				if prob >= prob_max and y_hat[3].item() > 4 and y_hat[3].item() < 9 and y_hat[2].item() > -1.5:
					prob_max = prob
					i_max = i
					window_max = window_normalized
					peak_medmax = peak_med
			if prob < prob_cutoff and prob_max > prob_cutoff:
				det_flag = 1
				prob_det = prob_max
				prob_max = 0

		if det_flag == 1:
			num_detected+=1
			prediction = predict(window_max, predictNet)
			event_coordref = (19.5,-155.5,0,0)
			event_norm = (1.0,1.0,50.0,10.0)
			y_hat = torch.squeeze(prediction)
			y_hat = np.multiply(y_hat,event_norm)
			y_hat = np.add(y_hat,event_coordref)

			year, day = parse(day_path)

			# 50 sps, 3600 s per hour
			ori_time = y_hat[3].item()+i_max/50
			hour = int(ori_time) // 3600
			minute = int(ori_time) % 3600 // 60
			second = int(ori_time) % 3600 % 60 
			timestamp = str(hour) + '.' + str(minute) + '.' + str(second)

			print(year +" " + day + " " +' %s %s %s %.4f %.4f %.2f %.2f %.2f %.5f %.2f' % (hour,minute,second,y_hat[0].item(),y_hat[1].item(),y_hat[2].item(),y_hat[3].item()+i_max/50,i_max/50,prob_det,peak_medmax)) # 50 sps

			saved_vars.append([year, day, hour, minute, second, y_hat[0].item(), y_hat[1].item(),y_hat[2].item(),y_hat[3].item()+i_max/50,i_max/50])

			skip = True
			skipTracker = 0
			det_flag = 0

	return num_detected, saved_vars

if __name__ == "__main__":
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	# Load trained detection model
	DETECT_PATH = './SeisConvNetDetect_sortedAbs50s.pth'
	detectNet = DetectNet()
	detectNet.load_state_dict(torch.load(DETECT_PATH))

	# Load trained prediction model
	PREDICT_PATH = './SeisConvNetLoc_NotAbs2017Mcut50s.pth'
	predictNet = PredictNet()
	predictNet.load_state_dict(torch.load(PREDICT_PATH))

	val_path = "/Volumes/jd/data.hawaii/sac_allstns_cont/screened"
	f = open("val_consecution_output_test.dat", "w")

	val_dirs = find_all_days(val_path)
	for day in val_dirs:
		daily_num_detected, daily_outputs = daily_detection(day, detectNet, predictNet)
		for output in daily_outputs:
			f.write('%s %s %s %s %s %.4f %.4f %.2f %.2f %.2f \n' % (output[0],output[1],output[2],output[3],output[4],output[5],output[6],output[7],output[8],output[9])) # 50 sps
		print('Number of detected earthquakes on this day: %.0f' %(daily_num_detected))

	f.close()