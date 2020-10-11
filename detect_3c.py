import numpy as np
import os
import matplotlib.pyplot as plt
import pdb
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Custom data pre-processor to transform X and y from numpy arrays to torch tensors
class PrepareData(Dataset):
	def __init__(self, path):
		self.X, self.y = torch.load(path)
		self.y = self.y.type(torch.LongTensor) # CE loss only accepts ints as classes

	def __len__(self):
		return self.X.shape[0]

	def __getitem__(self, idx):
		return self.X[idx], self.y[idx]


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # conv2d takes C_in, C_out, kernel size, stride, padding
        #self.conv1 = nn.Conv2d(1, 8, 3, stride = 1, padding=1)
        # input array (3,55,2000) - 06-22-2020
        #self.conv1 = nn.Conv2d(3, 4, (1,9), stride = 1, padding=(0,4))
        self.conv1 = nn.Conv2d(3, 4, (1,9), stride = 1, padding=(0,4))
        #self.conv1 = nn.Conv2d(3, 8, (1,9), stride = 1, padding=(0,4))

        #self.pool1 = nn.MaxPool2d((1,2), stride=(1,2))
        self.pool1 = nn.MaxPool2d((1,5), stride=(1,5))

        self.conv2 = nn.Conv2d(4, 4, (5,3), stride = 1, padding=(2,1))
        #self.conv2 = nn.Conv2d(8, 16, (5,3), stride = 1, padding=(2,1))

        self.pool2 = nn.MaxPool2d((1,2), stride=(1,2))

        self.conv3 = nn.Conv2d(4, 8, (5,3), stride = 1, padding=(2,1))
        #self.conv3 = nn.Conv2d(16, 32, (5,3), stride = 1, padding=(2,1))

        #self.conv4 = nn.Conv2d(64, 128, (5,3), stride = 1, padding=(2,1))

        # 3000 / 5 / 2 / 2 = 150 ; 3000/3/2/2 = 250; 2000/5/2/2
        #self.fc1 = nn.Linear(64*27*1000, 128)  
        #self.fc1 = nn.Linear(8*27*150, 256) 
        #self.fc1 = nn.Linear(8*55*100, 128) 
        # remember to make that x.view in forward is consistent
        self.fc1 = nn.Linear(8*55*125, 128)
        #self.fc1 = nn.Linear(32*55*100, 128)
        self.fc2 = nn.Linear(128, 2)

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
        x = x.view(-1, 8*55*125)
        #x = x.view(-1, 32*55*100)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x




def train_model(ds_train, ds_test):
	net = Net()
	# Cross Entropy Loss is used for classification
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.AdamW(net.parameters(), lr = 2e-5)
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
			# print(outputs)

			loss = criterion(outputs, _y)
			#acc = tr.eq(outputs.round(), _y).float().mean() # accuracy
			#print(loss.item())

			#=======backward pass=====================================
			loss.backward() # backpropagate the loss through the model
			optimizer.step() # update the gradients w.r.t the loss

			running_loss += loss.item()
			epoch_loss += loss.item()
			if i % 10 == 9:    # print every 10 mini-batches
				print('[%d, %5d] loss: %.3f' %
				(epoch + 1, i + 1, running_loss / 10))
				running_loss = 0.0
		#print('normalized training loss: %.3f' % (epoch_loss / i ))
		test_loss = 0.0

		correct = 0
		total = 0
		
		with torch.no_grad():
			for i, (_x, _y) in enumerate(ds_test):
				#print(_x.unsqueeze(1).shape)
				outputs = net(_x.unsqueeze(1))
				loss = criterion(outputs,_y)
				test_loss += loss.item()

				_, predicted = torch.max(outputs.data,1)
				total += _y.size(0)
				correct += (predicted == _y).sum().item()
		#print('Accuracy of the network on test images: %d%%' %(100*correct/total))
		#print('Accuracy of the network on test images: %.2f%%' %(100*correct/total))

		print('[epoch %d] test loss: %.3f training loss: %.3f' %
				(epoch + 1, test_loss / len(ds_test), epoch_loss / len(ds_train))) 
	
	print('Finished Training')
	return net

def test_model(ds,ds_loader, net):
	criterion = nn.CrossEntropyLoss	()
	# loss = 0
	test_no = len(ds)
	print(test_no)
	batch_size=32
	
	#correct = 0
	#total = 0
	# precision and recall values for classification threshold, thre
	thre = torch.arange(0,1.01,0.05)
	thre_no = len(thre)
	true_p = torch.zeros(thre_no)
	false_p =torch.zeros(thre_no)
	false_n = torch.zeros(thre_no)
	true_n = torch.zeros(thre_no)

	y_hat = torch.zeros(test_no,2)
	y_ori = torch.zeros(test_no)
	y_pre = torch.zeros(test_no)
	with torch.no_grad():
		for i, (_x, _y) in enumerate(ds_loader):
			outputs = net(_x.unsqueeze(1))
			
			# view output as probability and set classification threshold
			prob = F.softmax(outputs,1)
			#print(prob)

			for j in range(thre_no):
				pred_threshold = (prob>thre[j]).float()
				#print(pred_threshold)
			#_, predicted = torch.max(outputs.data,1)
				predicted = pred_threshold[:,1]
				#print(predicted)
			
				for m in range(len(_y)):
					if _y[m] == 1. and pred_threshold[m,1] == 1.:
						true_p[j] += 1.
					if _y[m] == 0. and pred_threshold[m,1] == 1.:
						false_p[j] += 1.
					if _y[m] == 1. and pred_threshold[m,1] == 0.:
						false_n[j] += 1.
					if _y[m] == 0. and pred_threshold[m,1] == 0.:
						true_n[j]  += 1.
			
			#total += _y.size(0)
			#correct += (predicted == _y).sum().item() 
			#print(outputs)
			#print(_y)
			#print(predicted)
			y_hat[batch_size*i:batch_size*(i+1),:] = outputs
			y_ori[batch_size*i:batch_size*(i+1)] = _y
			y_pre[batch_size*i:batch_size*(i+1)] = predicted
	#for k in range(test_no):
	#	if y_ori[k] != y_pre[k]:
	#		print(y_hat[k,0],y_hat[k,1],y_ori[k],y_pre[k])

	#print('Accuracy of the network on test dataset: %.2f%%' %(100*correct/total))
	# accuracy, precision, recall, true positive rate, false positive rate
	#acc = torch.zeros(thre_no)
	#pre = torch.zeros(thre_no)
	#rec = torch.zeros(thre_no)
	#tpr = torch.zeros(thre_no)
	#fpr = torch.zeros(thre_no) 
	print("Threshold, Accuracy, Precision, Recall, TPR, FPR, FScore")
	for j in range(thre_no):
		acc = 100*(true_p[j]+true_n[j])/(true_p[j]+true_n[j]+false_p[j]+false_n[j])

		if (true_p[j]+false_p[j]) > 0.:
			pre = 100*true_p[j]/(true_p[j]+false_p[j])
		else:
			pre = 100*torch.ones(1)

		if (true_p[j]+false_n[j]) > 0.:
			rec = 100*true_p[j]/(true_p[j]+false_n[j])
		else:
			rec = 100*torch.ones(1)

		tpr = 100*true_p[j]/(true_p[j]+false_n[j])
		fpr = 100*false_p[j]/(false_p[j]+true_n[j])
		fscore = 2*pre*rec/(pre+rec)
		#print(true_p[j],false_p[j],true_n[j],false_n[j])
		print(" %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f" %(thre[j].item(),acc.item(),pre.item(),rec.item(),tpr.item(),fpr.item(),fscore))

	#print('Accuracy of the network on test dataset: %.2f%%' %(100*(true_p+true_n)/(true_p+true_n+false_p+false_n)))
	#print(true_p)
	#print(false_p)
	#print(false_n)
	#print(true_n)
	#print('Precision: %.2f%%' %(100*true_p/(true_p+false_p)))
	#print('Recall: %.2f%%' %(100*true_p/(true_p+false_n)))
	#print('True Positive Rate: %.2f%%' %(100*true_p/(true_p+false_n)))
	#print('False Positive Rate: %.2f%%' %(100*false_p/(false_p+true_n)))
	
	# print the first few other true and predicted values
	#print('Examples of outputs, true classification, and predicted classification')
	#for k in range(10):
	#	print(y_hat[k,0],y_hat[k,1],y_ori[k],y_pre[k])
	# print('MSE of the network on the test set: %d %%' % (loss))


if __name__ == "__main__":

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	# # # pdb.set_trace()
	# sorted
	#ds_train = PrepareData(path = '/Volumes/jd/data.hawaii/pts/detect_train_data.pt')
	# unsorted and absolute
	#ds_train = PrepareData(path = '/Volumes/jd/data.hawaii/pts/detect_train_data_unsorted.pt')
	# unsorted and NOT absolute
	#ds_train = PrepareData(path = '/Volumes/jd/data.hawaii/pts/detect_train_data_unsortedNotAbs.pt')
	# sorted and absolute
	ds_train = PrepareData(path = '/Volumes/jd/data.hawaii/pts/detect_train_data_sortedAbs50s.pt')
	# sorted and NOT absolute
	#ds_train = PrepareData(path = '/Volumes/jd/data.hawaii/pts/detect_train_data_sortedNotAbs.pt')
	ds_train_loader = DataLoader(ds_train, batch_size=32, shuffle=True)

	# sorted
	#ds_test = PrepareData(path = '/Volumes/jd/data.hawaii/pts/detect_test_data.pt')
	# unsorted and absolute
	#ds_test = PrepareData(path = '/Volumes/jd/data.hawaii/pts/detect_test_data_unsorted.pt')
	# unsorted and NOT absolute
	#ds_test = PrepareData(path = '/Volumes/jd/data.hawaii/pts/detect_test_data_unsortedNotAbs.pt')
	# sorted and absolute
	ds_test = PrepareData(path = '/Volumes/jd/data.hawaii/pts/detect_test_data_sortedAbs50s.pt')
	# sorted and NOT absolute
	#ds_test = PrepareData(path = '/Volumes/jd/data.hawaii/pts/detect_test_data_sortedNotAbs.pt')
	ds_test_loader = DataLoader(ds_test, batch_size=32, shuffle=True)
	net = train_model(ds_train_loader, ds_test_loader)

	PATH = './SeisConvNetDetect_sortedAbs50s.pth'
	torch.save(net.state_dict(), PATH)

	# net = torch.load('./SeisConvNet.pth')
	accuracy = test_model(ds_test, ds_test_loader, net)



