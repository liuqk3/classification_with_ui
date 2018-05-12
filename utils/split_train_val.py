import os, shutil, random
import numpy as np
import pandas as pd
labels=pd.read_csv('./data/base/Annotations/label.csv')
number_sample=79572
factor=0.8
index=range(0,number_sample)
random.shuffle(index)
train_f=open('./train_labels.csv','w')
test_f=open('./test_labels.csv','w')
for i in range(0,number_sample):
    path,attribute,label= labels.iloc[index[i]]
    if(i<factor*number_sample):
        train_f.write(path+','+attribute+','+label+'\n')
    else:
        test_f.write(path+','+attribute+','+label+'\n')
train_f.close()
test_f.close()
