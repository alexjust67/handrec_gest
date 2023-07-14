import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import module1 as m1
from tensorboardX import SummaryWriter
import os
import cv2
import matplotlib.pyplot as plt


writer = SummaryWriter()

def arrayer(trainset_path,lable_path):
    #cycle through the images in the folder and make them into arrays
    first_lable = True
    lim=50
    for i in os.listdir(lable_path):
        if first_lable:
            y = cv2.imread(lable_path+i)
            x = cv2.imread(trainset_path + i)
            rgb = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
            rgb = np.array(rgb)/255
            x = np.expand_dims(rgb,3)
            y = np.expand_dims(np.array(y),3)
            first_lable = False
        else:
            img = cv2.imread(lable_path+i)
            img = np.array(img)
            img = np.expand_dims(img,3)
            y = np.append(y,img,axis=3)
            img = cv2.imread(trainset_path + i)
            img = np.array(img)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb = np.expand_dims(rgb,3)
            rgb = rgb/255
            x = np.append(x,rgb,axis=3)
        if lim == 0:
            break
        lim-=1

            
    return np.transpose(x,(3,2,1,0)),np.transpose(y,(3,2,1,0))

batch_size = 14
steps_before_print = 50
step_size = 2

x,y=arrayer("./Trainer/CRAID1/CRAID1/images/","./Trainer/CRAID1/CRAID1/train/PointsClass/")
x1,y1=arrayer("./Trainer/CRAID1/CRAID1/images/","./Trainer/CRAID1/CRAID1/val/segmentationClass/")

dataset = m1.dataset(x,y)
valdata = m1.dataset(x1,y1)

#plt.imshow(dataset.__getitem__(1)[1].transpose(1,2,0))
#plt.show()

dataloader = DataLoader(dataset=dataset,shuffle=True,batch_size=batch_size)

validataloader = DataLoader(dataset=valdata,shuffle=True,batch_size=batch_size)

model = m1.DensityMapModel()
#criterion = nn.MSELoss()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.001)

costval = []

dataiter = iter(dataloader)
lis, labels = next(dataiter)
writer.add_graph(model, lis)

def train(epochs):
    for epoch in range(epochs):
        step = 0
        epoch_loss = 0
        epoch_acc = 0
        times_calculated = 0
        total_size = len(dataloader)
        for i, (images, labels) in enumerate(dataloader):
            model.train()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            writer.add_scalar('trainning_loss', loss.item(), model.steps)
            loss.backward()
            optimizer.step()

            step += 1
            epoch_loss += loss.item()

            if step % steps_before_print == 0:
                # Calculate Accuracy
                model.eval()
                validation_loss, accuracy = m1.calculate_loss_and_accuracy(validataloader, model, criterion, stop_at = 1200)
                writer.add_scalar('validation_loss', validation_loss, model.steps)
                writer.add_scalar('accuracy', accuracy, model.steps)
                epoch_acc += accuracy
                times_calculated += 1
                # Print Loss
                print('Iteration: {}/{} - ({:.2f}%). Loss: {}. Accuracy: {}'.format(step, total_size, step*100/total_size , loss.item(), accuracy))
                
                if validation_loss < model.best_valdiation_loss:
                    model.best_valdiation_loss = validation_loss
                    print('Saving best model')
                    m1.save_model(model)
                del validation_loss
                
            del loss, outputs, images, labels

        model.epochs += 1

        #print('Epoch({}) avg loss: {} avg acc: {}'.format(epoch, epoch_loss/step, epoch_acc/times_calculated))
        print('Epoch ', epoch)
        #save_model(model, use_ts=True)

learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
train(50)
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
train(20)
learning_rate = 0.0001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
train(20)