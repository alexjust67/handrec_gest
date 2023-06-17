import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
import math
import time
import os

class dataset(Dataset):
  
    def __init__(self,x,y):
        self.x = torch.tensor(x,dtype=torch.float32)
        self.y = torch.tensor(y,dtype=torch.float32)
        self.length = self.x.shape[0]  
		
    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]  
    
    def __len__(self):
        return self.length





class net(nn.Module):
	def __init__(self,input_size,output_size):
		super(net,self).__init__()
		
		self.relu = nn.LeakyReLU()
		self.l1 = nn.Linear(input_size,126)
		self.l2 = nn.Linear(126,126)
		self.l3 = nn.Linear(126,50)
		self.l4 = nn.Linear(50,output_size)
		self.sftmx = nn.Softmax(dim=1)

		self.steps = 0
		self.epochs = 0
		self.best_valdiation_loss = math.inf

	def forward(self,x):
		
		output = self.l1(x) 
		output = self.relu(output)
		output = self.l2(output)
		output = self.relu(output)
		output = self.l3(output)
		output = self.relu(output)
		output = self.l4(output)
		if self.training: output = self.sftmx(output)

		if self.training:
			self.steps += 1
		
		return output





def calculate_loss_and_accuracy(validation_loader, model, criterion, stop_at = 1200, print_every=99999):
	correct = 0
	total = 0
	steps = 0
	total_loss = 0
	sz = len(validation_loader)
	
	for images, labels in validation_loader:
	
		if total%print_every == 0 and total > 0:
			accuracy = 100 * correct / total
			print(accuracy)
		
		if total >= stop_at:
			break;
		#if torch.cuda.is_available():
		#	images = images.cuda()
		#	labels = labels.cuda()

		# Forward pass only to get logits/output
		outputs = model(images)
		
		#Get Loss for validation data
		loss = criterion(outputs, labels)
		total_loss += loss.item()


		# Get predictions from the maximum value
		_, predicted = torch.max(outputs.data, 1)
		_, labels=torch.max(labels, 1)
		# Total number of labels
		total += labels.size(0)
		steps += 1

		correct += (predicted == labels).sum().item()

		del outputs, loss, _, predicted

	accuracy = 100 * correct / total
	return total_loss/steps, accuracy


def save_model(model, use_ts=False,curr_folder='./mdl'):
	if not os.path.exists(curr_folder):
		os.makedirs(curr_folder) 
	if use_ts:
		time_stamp = time.strftime("%d_%b_%Y_%Hh%Mm", time.gmtime())
		torch.save(model, curr_folder + '/{}.ckp'.format(time_stamp))		#model.state_dict()
	else:
		torch.save(model, curr_folder + '/{}.ckp'.format('best_model'))		#model.state_dict()