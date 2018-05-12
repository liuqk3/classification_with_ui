from utils.cnn_functions import resnet, train
import os

train_data_root = open('./buffer/train_data_root.txt').readlines()
train_data_root = train_data_root[len(train_data_root) - 1]

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
device_ids = [0,1]
model = resnet.AttributeClassification(pretrained=True)
model.train(True)
train(model, train_data_root, use_gpu=True, device_ids=device_ids, epoches=10)