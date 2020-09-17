import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import platform
import time
import copy
import random

"""
Method to train the model
"""
def train_model(model, criterion, optimizer, scheduler, train_loader, valid_loader, batch_size, device, save_path, num_epochs=25):
    
    #move model into gpu
    model = model.to(device)
    #define some lists to append important calculations
    train_acc_lst = []
    train_loss_lst = []
    val_acc_lst = []
    val_loss_lst = []
    val_min_loss = np.inf
    since = time.time()

    #loop through the epochs
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        running_loss = 0.0
        running_corrects = 0
        
        #set on train mode
        model.train()
        
        #training step(we only update the model in the training loop)
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            #calculate training loss
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        #re-adjust the learning rate as epoch increases
        scheduler.step()
        
        #calculate avg loss and accuracy for training process
        train_loss = running_loss / (len(train_loader) * batch_size)
        train_acc = running_corrects.double() / (len(train_loader) * batch_size)        
        train_acc_lst.append(train_acc)
        train_loss_lst.append(train_loss)
            
        #set to evaluation mode
        model.eval()
        
        running_loss = 0.0
        running_corrects = 0

        #validation step(we don't update the model in validation, that's why the evaluation is turned on)
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            #calculate training loss
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        #calculate avg loss and accuracy for validation process
        val_loss = running_loss / (len(valid_loader) * batch_size)
        val_acc = running_corrects.double() / (len(valid_loader) * batch_size)
        val_acc_lst.append(val_acc)
        val_loss_lst.append(val_loss)
        
        print('Train Loss: {:.4f} Acc: {:.4f}'.format(train_loss, train_acc))
        print('Validation Loss: {:.4f} Acc: {:.4f}'.format(val_loss, val_acc))
        
        #save the model if the validation loss is decreasing
        if val_loss < val_min_loss:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(val_min_loss,val_loss))
            torch.save(model.state_dict(), save_path)
            val_min_loss = val_loss
            #save the parameters for the current best model
            best_model_wts = copy.deepcopy(model.state_dict())
            
                    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('minimum val loss: {:4f}'.format(val_min_loss))

    #load the best model
    model.load_state_dict(best_model_wts)
    
    return model,train_acc_lst, train_loss_lst, val_acc_lst, val_loss_lst


"""
Method to fix the graident of the model's layers so that the layers won't be updated during training.
"""
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

"""
Calculate the accuracy of all class predictions.
"""
def calculate_accuracy(model, dataloader, name):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network all images in {} set: {:.2f}'.format(name, 100 * correct / total))


    
"""
Calculate the accuracy of specific class predictions.
"""
def calculate_accuracy_specific(model, dataloader, classes, name):
    c_length = len(classes)
    class_correct = list(0. for i in range(c_length))
    class_total = list(0. for i in range(c_length))
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(c_length):
        print('Accuracy of {} in {} set: {:.2f}'.format(classes[i], name, 100 * class_correct[i] / class_total[i])) 


