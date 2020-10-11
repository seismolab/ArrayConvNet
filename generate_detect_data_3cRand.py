import pandas as pd
import numpy as np
from obspy import read
import os
import matplotlib.pyplot as plt
import pdb
import torch

from sklearn.model_selection import train_test_split

# Load trace data given a SAC file
def load_data(filename, see_stats=False, bandpass=False):
    st = read(filename)
    tr = st[0]
    if bandpass:
        tr.filter(type='bandpass', freqmin=5.0, freqmax=40.0)
    tr.normalize()
    #tr.taper(0.2)
    tr.taper(0.02)
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

def get_event(path, station_no, showPlots=False):
#def get_event(path, station_no, showPlots=True):
	sample_size = 2500
	channel_size = 3
	event_array = torch.zeros(channel_size, station_no,sample_size)
	sorted_event_array = torch.zeros(channel_size, station_no,sample_size)
	#max_amp_idx = []
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
			#print(tr.stats)
			#print(filename,ii,i,j)
			if len(tr.data) < sample_size:
				print('ERROR '+filename+' '+str(len(tr.data)))
			# elif tr.stats['channel'] == 'HHZ':
			else:
				# pdb.set_trace()
				# idx = sorted_stations.index(tr.stats.station)
				# print(idx)
				#j = i % 3
				event_array[j,i,:] = torch.from_numpy(tr.data[:sample_size])
				#traces start 10 s or 500 sample points before the origin time
				# peak_amp = max(abs(event_array[j,i,500:]))
				peak_amp = max(abs(event_array[j,i,:]))
				# Organize by max amplitude in the horizontal direction
				#noise_amp = max(abs(event_array[j,i,50:450]))
				#snr_tmp = peak_amp / noise_amp
				if tr.stats['network'] != 'FG':
					#if snr_tmp < 2: 
					#	event_array[j,i,:] = torch.zeros(1,1, sample_size)
					#else:
					#	event_array[j,i,:] = event_array[j,i,:] / peak_amp
					#	if tr.stats['channel'] == 'HHZ' or tr.stats['channel'] == 'EHZ':
					#		max_amp_idx[i] = np.argmax(abs(event_array[j,i,:])).numpy()
					event_array[j,i,:] = event_array[j,i,:] / peak_amp
					if tr.stats['channel'] == 'HHZ' or tr.stats['channel'] == 'EHZ':
						max_amp_idx[i] = np.argmax(abs(event_array[j,i,:])).numpy()
				else:
					event_array[j,i,:] = event_array[j,i,:] * 0
					#if tr.stats['channel'] == 'HHZ' or tr.stats['channel'] == 'EHZ':
					#		max_amp_idx.append(np.array(sample_size))
				#snr.append(snr_tmp)
				# max_amp_idx.append(np.argmax(abs(event_array[i,:])))
				# max_amp.append(max(abs(event_array[i,:])))
				#if tr.stats['network'] == 'FG':
				#	event_array[j,i,:] = event_array[j,i,:] * 0
				#if tr.stats['channel'] == 'HHZ' or tr.stats['channel'] == 'EHZ':
					#max_amp_idx.append(np.argmax(abs(event_array[j,i,500:])))
			ii+=1
			if i == station_no:
				break

	# sort traces in order of when their maximum amplitude arrives
	#print(max_amp_idx.astype(int))
	idx = np.argsort(max_amp_idx)
	#print(idx)
	#sorted_max_amp = max_amp_idx[idx]
	#print(sorted_max_amp)
	#for k in range(station_no):
	#	print(max_amp_idx[k],idx[k],sorted_max_amp[k])
	#print(len(max_amp_idx))
	# any stations not included (out of 28) will be the last row of zeros
	# pdb.set_trace()
	# print(idx.shape)
	# print(event_array.shape)
	# print(sorted_event_array.shape)
	sorted_event_array = event_array[:,idx,:]
	# print(sorted_event_array.shape)

	# sorted and absolute
	event_array = abs(sorted_event_array)
	# sorted and NOT absolute
	#event_array = sorted_event_array

	if (showPlots):
		fig, axs = plt.subplots(station_no, sharey="col")
		fig.suptitle(path)
		for i in range(station_no):
			#j = ii % 3
			#i = ii // 3
			axs[i].plot(sorted_event_array[2,i,:])
			axs[i].axis('off')
		plt.show()
		# plt.savefig("test")
	return event_array	

def separate_stations(path):
    SAC_filenames = find_all_SAC(path)
    sample_size = 2000
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


if __name__ == "__main__":
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	pos_path = "/Volumes/jd/data.hawaii/data_prepared_FCreviewedRand50s"
	neg_path = "/Volumes/jd/data.hawaii/data_prepared_noise50s"
	sample_size = 2500
	pos_dirs = find_all_events(pos_path)
	neg_dirs = find_all_events(neg_path)
	print(len(pos_dirs))
	print(len(neg_dirs))
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

	print(X_all.shape, y_all.shape)	
	print(X_all)
	print(y_all)	
	X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size = 0.25, random_state=42)

	torch.save((X_train, y_train), '/Volumes/jd/data.hawaii/pts/detect_train_data_sortedAbs50s.pt')
	torch.save((X_test, y_test), '/Volumes/jd/data.hawaii/pts/detect_test_data_sortedAbs50s.pt')