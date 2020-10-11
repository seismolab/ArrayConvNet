import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

#========Preparing datasets for PyTorch DataLoader=====================================
# Custom data pre-processor to transform X and y from numpy arrays to torch tensors
class PrepareData(Dataset):
	def __init__(self, path):
		self.X, self.y = torch.load(path)
		# CE loss only accepts ints as classes
		self.y = self.y.type(torch.LongTensor) 

	def __len__(self):
		return self.X.shape[0]

	def __getitem__(self, idx):
		return self.X[idx], self.y[idx]

#========Network architecture=====================================
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # conv2d takes C_in, C_out, kernel size, stride, padding
        # input array (3,55,2500)
        self.conv1 = nn.Conv2d(3, 4, (1,9), stride = 1, padding=(0,4))
        self.pool1 = nn.MaxPool2d((1,5), stride=(1,5))
        self.conv2 = nn.Conv2d(4, 4, (5,3), stride = 1, padding=(2,1))
        self.pool2 = nn.MaxPool2d((1,2), stride=(1,2))
        self.conv3 = nn.Conv2d(4, 8, (5,3), stride = 1, padding=(2,1))

        self.fc1 = nn.Linear(8*55*125, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(torch.squeeze(x,1))))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool2(F.relu(self.conv3(x)))
        x = x.view(-1, 8*55*125)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


#========Training the model =====================================
# ds_train is the training dataset loader
# ds_test is the testing dataset loader
def train_model(ds_train, ds_test):
	net = Net()
	# Cross Entropy Loss is used for classification
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.AdamW(net.parameters(), lr = 2e-5)
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

			#=======backward pass=====================================
			loss.backward() # backpropagate the loss through the model
			optimizer.step() # update the gradients w.r.t the loss

			running_loss += loss.item()
			epoch_loss += loss.item()
			if i % 10 == 9:    # print every 10 mini-batches
				print('[%d, %5d] loss: %.3f' %
				(epoch + 1, i + 1, running_loss / 10))
				running_loss = 0.0
		
		# For each epoch, monitor test loss to ensure we are not overfitting
		test_loss = 0.0
		correct = 0
		total = 0
		
		with torch.no_grad():
			for i, (_x, _y) in enumerate(ds_test):
				outputs = net(_x.unsqueeze(1))
				loss = criterion(outputs,_y)
				test_loss += loss.item()

				_, predicted = torch.max(outputs.data,1)
				total += _y.size(0)
				correct += (predicted == _y).sum().item()

		print('[epoch %d] test loss: %.3f training loss: %.3f' %
				(epoch + 1, test_loss / len(ds_test), epoch_loss / len(ds_train))) 
	
	print('Finished Training')
	return net

#========Testing the model =====================================
# ds is the testing dataset
# ds_loader is the testing dataset loader
# net is the trained network
def test_model(ds,ds_loader, net):
	criterion = nn.CrossEntropyLoss()
	test_no = len(ds)
	batch_size=32
	
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

			for j in range(thre_no):
				pred_threshold = (prob>thre[j]).float()
				predicted = pred_threshold[:,1]
			
				for m in range(len(_y)):
					if _y[m] == 1. and pred_threshold[m,1] == 1.:
						true_p[j] += 1.
					if _y[m] == 0. and pred_threshold[m,1] == 1.:
						false_p[j] += 1.
					if _y[m] == 1. and pred_threshold[m,1] == 0.:
						false_n[j] += 1.
					if _y[m] == 0. and pred_threshold[m,1] == 0.:
						true_n[j]  += 1.
		
			y_hat[batch_size*i:batch_size*(i+1),:] = outputs
			y_ori[batch_size*i:batch_size*(i+1)] = _y
			y_pre[batch_size*i:batch_size*(i+1)] = predicted

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
		print(" %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f" %(thre[j].item(),acc.item(),pre.item(),rec.item(),tpr.item(),fpr.item(),fscore))



if __name__ == "__main__":

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	# Prepare the training dataset and loader 
	# path is where the preprocessed training event data is housed
	ds_train = PrepareData(path = '/Volumes/jd/data.hawaii/pts/detect_train_data_sortedAbs50s.pt')
	ds_train_loader = DataLoader(ds_train, batch_size=32, shuffle=True)

	# Prepare the testing dataset and loader 
	# path is where the preprocessed test event data is housed
	ds_test = PrepareData(path = '/Volumes/jd/data.hawaii/pts/detect_test_data_sortedAbs50s.pt')
	ds_test_loader = DataLoader(ds_test, batch_size=32, shuffle=True)
	net = train_model(ds_train_loader, ds_test_loader)

	# detect_net_path is where we will store our trained model
	detect_net_path = './SeisConvNetDetect_sortedAbs50s.pth'
	torch.save(net.state_dict(), detect_net_path)

	# Analyze our final model on the testing dataset 
	accuracy = test_model(ds_test, ds_test_loader, net)



