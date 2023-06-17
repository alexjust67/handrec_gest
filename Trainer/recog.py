import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
import module1 as m1
from tensorboardX import SummaryWriter


writer = SummaryWriter()

with open("valiset", "rb") as fp:   # Unpickling
  a = pickle.load(fp)
with open("trainset", "rb") as fp:   # Unpickling
  b = pickle.load(fp)

def arrayer(b):
    x = []
    y = []
    for m in b:
        x.append(m[1])
    for m in b:
        if m[0][0]=='ok':
            y.append([1,0,0])
        if m[0][0]=='openplm':
            y.append([0,1,0])
        if m[0][0]=='thumup':
            y.append([0,0,1])
    x=np.array(x)
    y=np.array(y)
    return x,y

batch_size = 14
steps_before_print = 50
step_size = 2

x,y=arrayer(b)
x1,y1=arrayer(a)
dataset = m1.dataset(x,y)
valdata = m1.dataset(x1,y1)

dataloader = DataLoader(dataset=dataset,shuffle=True,batch_size=batch_size)
validataloader = DataLoader(dataset=valdata,shuffle=True,batch_size=batch_size)

model = m1.net(x.shape[1],3)
#criterion = nn.MSELoss()
criterion = nn.CrossEntropyLoss()
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