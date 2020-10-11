import pandas as pd
import numpy as np
from obspy import read
import os
import matplotlib.pyplot as plt
import pdb
import torch

from sklearn.model_selection import train_test_split

#========Load raw trace data=====================================
# Load trace data given a SAC file
def load_data(filename, see_stats=False, bandpass=False):
    st = read(filename)
    tr = st[0]
    if bandpass:
        tr.filter(type='bandpass', freqmin=5.0, freqmax=40.0)
    tr.taper(0.02)
    if see_stats:
        print(tr.stats)
        tr.plot()
    return tr

#========Find all events=====================================
# Inputs a path and returns all events (directories) is list     
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

#========Get all station traces for a  given event=====================================
# Given a path, find all station traces. If a station did not record an event, zero-fill      
def get_event(path, station_no, showPlots=False):
	sample_size = 2500
	channel_size = 3
	event_array = torch.zeros(channel_size, station_no,sample_size)
	sorted_event_array = torch.zeros(channel_size, station_no,sample_size)
	max_amp_idx = np.ones(station_no) * sample_size
	snr = []
	ii=0
	for r, d, f in os.walk(path):
		if f == []:
			return []
		for filename in sorted(f):
			i = ii // 3
			j = ii % 3
			tr = load_data(os.path.join(r,filename), False)
			if len(tr.data) < sample_size:
				print('ERROR '+filename+' '+str(len(tr.data)))
			else:
				event_array[j,i,:] = torch.from_numpy(tr.data[:sample_size])
				peak_amp = max(abs(event_array[j,i,:]))
				if tr.stats['network'] != 'FG':
					event_array[j,i,:] = event_array[j,i,:] / peak_amp
					if tr.stats['channel'] == 'HHZ' or tr.stats['channel'] == 'EHZ':
						max_amp_idx[i] = np.argmax(abs(event_array[j,i,:])).numpy()
				else:
					event_array[j,i,:] = event_array[j,i,:] * 0
			ii+=1
			if i == station_no:
				break

	# sort traces in order of when their maximum amplitude arrives
	idx = np.argsort(max_amp_idx)
	sorted_event_array = event_array[:,idx,:]

	# sorted and absolute
	event_array = abs(sorted_event_array)

	# Include option to visualize traces for each event
	if (showPlots):
		fig, axs = plt.subplots(station_no, sharey="col")
		fig.suptitle(path)
		for i in range(station_no):
			axs[i].plot(sorted_event_array[2,i,:])
			axs[i].axis('off')
		plt.show()
	return event_array	


if __name__ == "__main__":
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	pos_path = "/Volumes/jd/data.hawaii/data_prepared_FCreviewedRand50s"
	neg_path = "/Volumes/jd/data.hawaii/data_prepared_noise50s"
	sample_size = 2500
	pos_dirs = find_all_events(pos_path)
	neg_dirs = find_all_events(neg_path)
	print(len(pos_dirs)) # number of earthquake events
	print(len(neg_dirs)) # number of noise events
	station_no = 55
	channel_size=3
	X_all = torch.zeros(len(pos_dirs)+len(neg_dirs), channel_size, station_no, sample_size)
	y_all = torch.zeros(len(pos_dirs)+len(neg_dirs))
	
	for i,dirname in enumerate(pos_dirs):
	 	print(dirname)
	 	event_array = get_event(dirname, station_no)
	 	X_all[i,:,:] = event_array
	 	y_all[i] = torch.tensor(1)

	for i,dirname in enumerate(neg_dirs):
		print(dirname)
		event_array = get_event(dirname, station_no)
		X_all[i+len(pos_dirs),:,:] = event_array
		y_all[i+len(pos_dirs)] = torch.tensor(0)

	# Split all data randomly into a 75-25 training/test set 
	X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size = 0.25, random_state=42)

	# Save all processed data into training and test files
	torch.save((X_train, y_train), '/Volumes/jd/data.hawaii/pts/detect_train_data_sortedAbs50s.pt')
	torch.save((X_test, y_test), '/Volumes/jd/data.hawaii/pts/detect_test_data_sortedAbs50s.pt')