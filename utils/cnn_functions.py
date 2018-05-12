from __future__ import print_function
import torch
import torchvision
import torch.nn as nn
from torch.autograd import  Variable
from models import resnet
from torchvision import transforms
from torch.utils.data import DataLoader
import os
from torch.nn.parameter import Parameter
from PIL import Image
from utils import dataset
import pandas as pd
import matplotlib.pyplot as plt

#os.environ['CUDA_VISIBLE_DEVICES']='0,1'

batch_size=60
train_labels=pd.read_csv('./buffer/train_labels.csv')
#train_labels = pd.read_csv('./data/base/Annotations/label.csv')
test_labels=pd.read_csv('./buffer/test_labels.csv')
train_transform=torchvision.transforms.Compose(
    [
    transforms.Resize(300),
    transforms.RandomCrop((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.5,contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229,0.224,0.225])
    ])
test_transform=torchvision.transforms.Compose(
    [
    transforms.Resize(300),
    transforms.CenterCrop((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229,0.224,0.225])
    ])

def train(model,train_data_root,use_gpu=True,device_ids=None,epoches=12):
    train_data = dataset.attribute_dataset(root_dir=train_data_root, labels=train_labels, transform=train_transform)

    train_dataloader = DataLoader(dataset=train_data, shuffle=True, batch_size=batch_size, num_workers=4)

    optimizer=torch.optim.Adam(model.parameters(),lr=1e-6,weight_decay=1e-4)
    loss_function=torch.nn.NLLLoss()
    best_accuracy=0
    if use_gpu:
        model=model.cuda()
    if use_gpu and device_ids is not None:
        # model=nn.DataParallel(model,device_ids)
        # optimizer=nn.DataParallel(optimizer,device_ids)
        model=nn.DataParallel(model)
        optimizer=nn.DataParallel(optimizer)
    for epoch in range(0,epoches):
        model.train(True)
        m=0
        for images,labels in train_dataloader:
            #print labels
            if use_gpu:
                images,labels=Variable(images.cuda()),Variable(labels.cuda())
            else:
                images,labels=Variable(images),Variable(labels)
            outputs=model(images)
            output=torch.cat(outputs, dim=1)
            loss=loss_function(torch.log(output),labels)

            if (m+1)%10==0:
                plt.plot(m,loss.data[0],'ro')
                plt.savefig('./buffer/loss.jpg')
                plt.xlabel('iteration')
                plt.ylabel('loss')
                plt.ylim([0, 3])
                plt.xlim([0,1000])


            optimizer.zero_grad()
            loss.backward()
            if device_ids is not None:
                optimizer.module.step()
            else:
                optimizer.step()
            _,preds=torch.max(outputs[0].data,1)
            preds=torch.unsqueeze(preds,1)
            label_shift=[0,6,14,19,24,29,39,45]
            for i in range(1,8):
                _,tmp=torch.max(outputs[i].data,1)
                tmp=torch.unsqueeze(tmp, 1)
                tmp=tmp+label_shift[i]
                preds=torch.cat((preds,tmp),dim=1)
            accuracy=0.0
            for i in range(0,len(labels.data)):
                accuracy+=torch.sum(preds[i,:]==labels.data[i])

            #state_iter = "epoch:{}/{},train_loss:{},accuracy:{}".format(epoch, m, loss.data[0], accuracy/(len(labels.data) + 0.0))
            state_iter = "epoch:{}/{},train_loss:{}".format(epoch, m, loss.data[0])


            log_path = './buffer/log.log'
            log_file = open(log_path, mode='a')
            log_file.write(state_iter + '\n')
            log_file.close()

            print(state_iter)

            m=m+1
        model.train(False)
        accuracy=test(model,train_data_root)
        if accuracy>best_accuracy:
            best_accuracy=accuracy
            best_model=model.state_dict()
        print("####test for epoch:{},accuracy:{}".format(epoch,accuracy))
    model.train(False)
    torch.save(best_model,'./weights/parameters.pkl')

def test(model,train_data_root):
    model.eval()
    test_data = dataset.attribute_dataset(root_dir=train_data_root, labels=test_labels, transform=test_transform)
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, num_workers=4)
    accuracy=0.0
    number_sample=0
    for images,labels in test_dataloader:
        images,labels=Variable(images.cuda()),Variable(labels.cuda())
        outputs=model(images)
        _, preds = torch.max(outputs[0].data, 1)
        preds = torch.unsqueeze(preds, 1)
        number_sample+=batch_size
        label_shift = [0, 6, 14, 19, 24, 29, 39, 45]
        for i in range(1, 8):
            _, tmp = torch.max(outputs[i].data, 1)
            tmp = torch.unsqueeze(tmp, 1)
            tmp = tmp + label_shift[i]
            preds = torch.cat((preds, tmp), dim=1)
        for i in range(0, len(labels.data)):
            accuracy += torch.sum(preds[i, :] == labels.data[i])
    return accuracy/number_sample

def load_model_parameter(model,params):
    own_dict=model.state_dict()
    for name,val in params.items():
        name=name[7:]
        if name in own_dict.keys():
            if isinstance(val,Parameter):
                val=val.data
            own_dict[name].copy_(val)
    return model

tasks_all = ['skirt_length_labels', 'coat_length_labels', 'collar_design_labels', 'lapel_design_labels',
                 'neck_design_labels', 'neckline_design_labels', 'pant_length_labels', 'sleeve_length_labels']

task_index=[2,5,0,7,4,1,3,6]
def weights_init(m):
    classname=m.__class__.__name__
    if classname.find('attention') != -1 or classname.find('fc') !=-1:
        nn.init.xavier_normal(m.weight.data)
        nn.init.xavier_normal(m.bias.data)

def inference(image_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    net = resnet.AttributeClassification(pretrained=True)
    #net.load_stat_dict(torch.load('./weights/parameters.pkl'))
    net = load_model_parameter(net, torch.load('./weights/parameters.pkl'))
    net.cuda()
    image = Image.open(image_path)
    image = test_transform(image)
    image=torch.unsqueeze(image,0)
    image = Variable(image.cuda())
    output = net(image)
    predict = {}
    for i in range(0, 8):
        prob, category = torch.max(output[i].data, 1)
        predict[tasks_all[i]] = [category.cpu().numpy()[0], prob.cpu().numpy()[0]]
    return predict
