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

class DetectNet(nn.Module):
    def __init__(self):
        super(DetectNet, self).__init__()
        # conv2d takes C_in, C_out, kernel size, stride, padding
        #self.conv1 = nn.Conv2d(1, 8, 3, stride = 1, padding=1)
        # input array (3,55,2000) - 06-22-2020
        #self.conv1 = nn.Conv2d(3, 4, (1,9), stride = 1, padding=(0,4))
        self.conv1 = nn.Conv2d(3, 4, (1,9), stride = 1, padding=(0,4))
        #self.pool1 = nn.MaxPool2d((1,2), stride=(1,2))
        self.pool1 = nn.MaxPool2d((1,5), stride=(1,5))

        self.conv2 = nn.Conv2d(4, 4, (5,3), stride = 1, padding=(2,1))
        #self.conv2 = nn.Conv2d(4, 4, (7,7), stride = 1, padding=(3,3))
        self.pool2 = nn.MaxPool2d((1,2), stride=(1,2))

        self.conv3 = nn.Conv2d(4, 8, (5,3), stride = 1, padding=(2,1))
        #self.conv4 = nn.Conv2d(64, 128, (5,3), stride = 1, padding=(2,1))

        # 3000 / 5 / 2 / 2 = 150 ; 3000/3/2/2 = 250; 2000/5/2/2
        #self.fc1 = nn.Linear(64*27*1000, 128)  
        #self.fc1 = nn.Linear(8*27*150, 256) 
        #self.fc1 = nn.Linear(8*55*100, 128) 
        # remember to make that x.view in forward is consistent
        self.fc1 = nn.Linear(8*55*125, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        #print(x.shape)
        x = self.pool1(F.relu(self.conv1(x)))
        #x = self.pool1(F.relu(self.conv1(x)))
        # print(x.shape)
        x = self.pool2(F.relu(self.conv2(x)))
        # print(x.shape)
        x = self.pool2(F.relu(self.conv3(x)))
        # print(x.shape)
        #x = self.pool2(F.relu(self.conv4(x)))
        # print(x.shape)
        x = x.view(-1, 8*55*125)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class PredictNet(nn.Module):
    def __init__(self):
        super(PredictNet, self).__init__()
        # conv2d takes C_in, C_out, kernel size, stride, padding
        # input array size is (27,8000)
        #self.conv1 = nn.Conv2d(1, 8, 3, stride = 1, padding=1)
        # input array (3,55,2000) - 06-22-2020
        self.conv1 = nn.Conv2d(3, 4, (1,9), stride = 1, padding=(0,4))
        #self.pool = nn.MaxPool2d((1,2), stride=(1,2))
        self.pool1 = nn.MaxPool2d((1,5), stride=(1,5))
        self.conv2 = nn.Conv2d(4, 4, (5,3), stride = 1, padding=(2,1))
        self.pool2 = nn.MaxPool2d((1,2), stride=(1,2))
        self.conv3 = nn.Conv2d(4, 8, (5,3), stride = 1, padding=(2,1))
        #self.conv4 = nn.Conv2d(64, 128, (5,3), stride = 1, padding=(2,1))

        # 3000 / 5 / 2 / 2 = 150 ; 3000/3/2/2 = 250; 2000/5/2/2
        #self.fc1 = nn.Linear(64*27*1000, 128)  
        #self.fc1 = nn.Linear(8*27*150, 256) 
        self.fc1 = nn.Linear(8*55*125, 128) 
        #self.fc2 = nn.Linear(128, 3)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        #print(x.shape)
        x = self.pool1(F.relu(self.conv1(x)))
        #x = self.pool1(F.relu(self.conv1(x)))
        # print(x.shape)
        x = self.pool2(F.relu(self.conv2(x)))
        # print(x.shape)
        x = self.pool2(F.relu(self.conv3(x)))
        # print(x.shape)
        #x = self.pool2(F.relu(self.conv4(x)))
        # print(x.shape)
        x = x.view(-1, 8*55*125)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Run prediction model over a window
def predict(window, predictNet):
	with torch.no_grad():
		outputs = predictNet(window.unsqueeze(0))
	return outputs

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

# Load trace data given a SAC file
def load_data(filename, see_stats=False, bandpass=False):
    st = read(filename)
    tr = st[0]
    if bandpass:
        tr.filter(type='bandpass', freqmin=5.0, freqmax=40.0)
    tr.normalize()
    #tr.taper(0.2)
    #tr.taper(0.02)
    if see_stats:
        print(tr.stats)
        tr.plot()
    return tr


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

# Find all event directories     
def find_all_days(path):
    dirs = []
    for r, d, f in os.walk(path):
        for name in d:
            dirs.append(os.path.join(r, name))
    return dirs

# Get year amd day from path
def parse(path):
	dir_name = path.split('/')[-1]
	year = dir_name.split('.')[0]
	day = dir_name.split('.')[1]
	return year, day


# Detect earthquakes within a day
def daily_detection(day_path, detectNet, predictNet):
	sampling_rate = 50 # we collect 50 samples per second
	# interval = 10 # we want to window every 10 seconds
	# interval = 5 # try every 5 seconds
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
	# prob_cutoff = 0.68
	# prob_cutoff = 0.997
	prob_det = 0

	for i in idx:
		# ensure we get full length windows 
		if i+window_n > n:
			break
		if skip:
			skipTracker += 1
			# if skipTracker == 2:
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
					# if k == 2:
					# 	max_amp_idx[j] = np.argmax(abs(window_normalized[2, j,:]))
					# use the horizontal (E) channel (and shear wave) for max amp, if it exists
					# if k == 0:
					# 	max_amp_idx[j] = np.argmax(abs(window_normalized[0, j,:]))
		for j in range(station_no):
			if trace_max[0,j] != 0:
				idx[0] = np.argmax(abs(window_normalized[0, j,:]))
				idx[1] = np.argmax(abs(window_normalized[1, j,:]))
				idx[2] = np.argmax(abs(window_normalized[2, j,:]))
				max_amp_idx[j] = np.median(idx)
			else:
				max_amp_idx[j] = np.argmax(abs(window_normalized[2, j,:]))

		peak_med = np.median(max_amp_idx)/50
		# peak_ave = np.mean(max_amp_idx)/50
		# peak_kilauea = (max_amp_idx[49].item()+max_amp_idx[53].item()+max_amp_idx[42].item())/3.0/50.
		# peak_kilauea = (max_amp_idx[49].item())/50.
		#window_normalized = abs(window_normalized)
		
		# pdb.set_trace()
		# fig, axs = plt.subplots(55, sharey="col")
		# for i in range(55):
		# 		axs[i].plot(continuous_sac[2,i,:])
		# 		axs[i].axis('off')
		# plt.show()
		# Sort the window by earthquake "arrival" time for detection in the HHZ or EHZ direction
		# max_amp_idx = np.argmax(abs(window_normalized[2,:,:]), axis=1) # get the idx of the maximum amplitude in time series direction
		
		if peak_med > 7 and peak_med < 43:
			sort_idx = np.argsort(max_amp_idx) # sort by index of maximum amplitude
			#print(sort_idx)
			# case 3 
			sorted_window = abs(window_normalized[:,sort_idx,:])
			# case 4
			#sorted_window = window_normalized[:,sort_idx,:]

			# case 1: normalized, unsorted and NOT absolute traces
			#detection = detect(window_normalized, detectNet)
			#detection, outputs = detect(window_normalized, detectNet)

			# case 2: normalized, unsorted and absolute traces
			#detection = detect(abs(window_normalized), detectNet)
			
			# case 3/4: sorted 
			#detection = detect(sorted_window, detectNet)

			detection, outputs = detect(sorted_window, detectNet)

			prob_outputs = F.softmax(outputs,1)
			prob = prob_outputs[:,1].item()
			# detection is [0] = noise or [1] = earthquake
			#print(detection)
			# Run prediction only if an earthquake is detected
			# because most stations are near Kilauea, events near/far from Kilauea have
			# different median and average values

			# if prob > 0.999:
			# 	if prob > prob_max:
			# 		prob_max = prob
			# 		i_max = i
			# 		window_max = window_normalized
			# if prob < 0.999 and prob_max > 0.999:
			# 	det_flag = 1
			# 	prob_max =0

			if prob > prob_cutoff:
				prediction = predict(window_normalized, predictNet)
				event_coordref = (19.5,-155.5,0,0)
				event_norm = (1.0,1.0,50.0,10.0)
				y_hat = torch.squeeze(prediction)
				y_hat = np.multiply(y_hat,event_norm)
				y_hat = np.add(y_hat,event_coordref)
				# refstn = (19.41, -155.28)
				# epi = (y_hat[0].item(),y_hat[1].item())
				# dist = distance.distance(refstn, epi).km 
				# dist = math.sqrt(dist**2+y_hat[2].item()**2)
				# if dist < 50:
				# 	tt = dist/3.4
				# else:
				# 	tt = dist/4.0
				
				# # tt = tt + y_hat[3].item()
				# # print(dist)
				# # print(y_hat[3].item())
				# # print(tt)
				# ttdiff = peak_med-tt
				# ttdiff = peak_kilauea - tt

			# if prob > prob_cutoff and ttdiff < 10 and ttdiff > 2:
			# if prob > prob_cutoff and ttdiff < 15:
			# if prob > prob_cutoff:
				if prob >= prob_max and y_hat[3].item() > 4 and y_hat[3].item() < 9 and y_hat[2].item() > -1.5:
					prob_max = prob
					i_max = i
					window_max = window_normalized
					peak_medmax = peak_med
			if prob < prob_cutoff and prob_max > prob_cutoff:
				det_flag = 1
				prob_det = prob_max
				prob_max = 0

			# if peak_ave >= peak_med:
			# 	# if prob > 0.999 and peak_med < 15 and peak_med > 9:
			# 	# 	det_flag = 1
			# 	# 	prev_i = i
			# 	# else:
			# 	# 	det_flag = 0
			# 	if prob > prob_cutoff and peak_med < 28 and peak_med > 18:
			# 		if prob >= prob_max:
			# 			prob_max = prob
			# 			i_max = i
			# 			window_max = window_normalized
			# 			peak_medmax = peak_med
			# 	if prob < prob_cutoff and prob_max > prob_cutoff:
			# 		det_flag = 1
			# 		prob_det = prob_max
			# 		prob_max =0
			# if peak_ave < peak_med:
			# 	# if prob > 0.999 and peak_med < 25 and peak_med > 19:
			# 	# 	det_flag = 1
			# 	# 	prev_i = i
			# 	# else:
			# 	# 	det_flag = 0
			# 	if prob > prob_cutoff and peak_med < 40 and peak_med > 27:
			# 		if prob >= prob_max:
			# 			prob_max = prob
			# 			i_max = i
			# 			window_max = window_normalized
			# 			peak_medmax = peak_med
			# 	if prob < prob_cutoff and prob_max > prob_cutoff:
			# 		det_flag = 1
			# 		prob_det = prob_max
			# 		prob_max =0

		if det_flag == 1:
			# print("detected")
			num_detected+=1
			# localization uses normalized, but unsorted and not absolute traces
			# prediction = predict(window_normalized, predictNet)
			prediction = predict(window_max, predictNet)
			# prediction = predict(prev_norm, predictNet)
			# prediction = scaled and shifted lat, long, depth, and origin time, see predict_location3c4d.py
			event_coordref = (19.5,-155.5,0,0)
			event_norm = (1.0,1.0,50.0,10.0)
			#print(prediction)
			y_hat = torch.squeeze(prediction)
			y_hat = np.multiply(y_hat,event_norm)
			y_hat = np.add(y_hat,event_coordref)

			# refstn = (19.4, -155.3)
			# epi = (y_hat[0].item(),y_hat[1].item())
			# dist = distance.distance(refstn, epi).km 
			# print(dist)

			year, day = parse(day_path)

			# 50 sps, 3600 s per hour
			# ori_time = y_hat[3].item()+prev_i/50
			ori_time = y_hat[3].item()+i_max/50
			hour = int(ori_time) // 3600
			minute = int(ori_time) % 3600 // 60
			second = int(ori_time) % 3600 % 60 
			timestamp = str(hour) + '.' + str(minute) + '.' + str(second)

			# print(year +" " + day + " " +' %s %s %s %.4f %.4f %.2f %.2f %.2f' % (hour,minute,second,y_hat[0].item(),y_hat[1].item(),y_hat[2].item(),y_hat[3].item()+prev_i/50,prev_i/50)) # 50 sps

			# saved_vars.append([year, day, hour, minute, second, y_hat[0].item(), y_hat[1].item(),y_hat[2].item(),y_hat[3].item()+prev_i/50,prev_i/50])
			print(year +" " + day + " " +' %s %s %s %.4f %.4f %.2f %.2f %.2f %.5f %.2f' % (hour,minute,second,y_hat[0].item(),y_hat[1].item(),y_hat[2].item(),y_hat[3].item()+i_max/50,i_max/50,prob_det,peak_medmax)) # 50 sps

			saved_vars.append([year, day, hour, minute, second, y_hat[0].item(), y_hat[1].item(),y_hat[2].item(),y_hat[3].item()+i_max/50,i_max/50])

			# fig, axs = plt.subplots(55,2, sharey="col")
			# fig.suptitle(timestamp)

			# for i in range(55):
			# 		axs[i,0].plot(sorted_window[2,i,:])
			# 		axs[i,0].axis('off')
			# 		axs[i,1].plot(window_normalized[2,i,:])
			# 		axs[i,1].axis('off')
			# plt.draw()
			# plt.pause(0.1)
			# plt.savefig(timestamp+'.png', dpi=300)
			# plt.close()

			#detected.append((window_normalized, prediction))
			# if detected move forward ignore overlapping window
			skip = True
			skipTracker = 0
			det_flag = 0

	return num_detected, saved_vars

if __name__ == "__main__":
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	#os.environ['KMP_DUPLICATE_LIB_OK']='True'
	# Load trained detection model
	#DETECT_PATH = './DetectNet.pth'

	# case 1
	#DETECT_PATH = './SeisConvNetDetect_unsortedNotAbs.pth'
	
	# case 3
	DETECT_PATH = './SeisConvNetDetect_sortedAbs50s.pth'

	# case 4
	#DETECT_PATH = './SeisConvNetDetect_sortedNotAbs.pth'

	detectNet = DetectNet()
	detectNet.load_state_dict(torch.load(DETECT_PATH))

	# Load trained prediction model
	#PREDICT_PATH = './PredictNet.pth'
	PREDICT_PATH = './SeisConvNetLoc_NotAbs2017Mcut50s.pth'
	predictNet = PredictNet()
	predictNet.load_state_dict(torch.load(PREDICT_PATH))

	val_path = "/Volumes/jd/data.hawaii/sac_allstns_cont/screened"
	f = open("val_consecution_output_test.dat", "w")

	val_dirs = find_all_days(val_path)
	for day in val_dirs:
		daily_num_detected, daily_outputs = daily_detection(day, detectNet, predictNet)
		for output in daily_outputs:
			# print('%s %s %.4f %.4f %.2f %.2f %.2f \n' % (output[0], output[1], output[2],output[3],output[4],output[5],output[6])) 
			f.write('%s %s %s %s %s %.4f %.4f %.2f %.2f %.2f \n' % (output[0],output[1],output[2],output[3],output[4],output[5],output[6],output[7],output[8],output[9])) # 50 sps
		print('Number of detected earthquakes on this day: %.0f' %(daily_num_detected))

	f.close()

	#print(detected)
	#pdb.set_trace()
	#print('Number of detected earthquakes: %.2f%%' %(len(detected)))
