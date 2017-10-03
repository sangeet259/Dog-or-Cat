
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import os
import sys
from random import shuffle
from tqdm import tqdm


# In[2]:


TRAIN_DIR ='/media/sangeet/Stuff/DC Shares/Courses and Tutorials/ML DL AI/Datasets/DogsVsCats-Kaggle/train'
TEST_DIR='/media/sangeet/Stuff/DC Shares/Courses and Tutorials/ML DL AI/Datasets/DogsVsCats-Kaggle/test'
IMG_SIZE=50
LR = 1e-3

MODEL_NAME='dogsvscats-{}-{}.model'.format(LR,'2conv-basic')


# In[3]:


def label_img(img):
    word_label=img.split('.')[0]
    if word_label =='cat' : return [1,0]
    elif word_label == 'dog' : return [0,1]


# In[4]:


def create_train_data():
    training_data =[]
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE))
        shuffle(training_data)
    np.save('train_data.npy',training_data)
    return training_data


# In[5]:


def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
        img_num = img.split('.')[0]
        img = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img),img_num])
    np.save('test_data.npy',testing_data)
    return testing_data


# In[6]:
train_data=create_train_data()
print(len(train_data))
sys.exit(0)
import pickle

with open("train_data.pkl","wb") as f:
	pickle.dump(train_data,f)

# In[13]:


import tensorflow as tf

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')


# In[14]:


if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')


# In[15]:


train = train_data[:-500]
test = train_data[-500:]


# In[16]:


X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y = [i[1] for i in train]


test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y = [i[1] for i in test]


# In[19]:


try:
	model.fit({'input': X}, {'targets': Y}, n_epoch=3, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
except IndexError as e:
	print("Fucked Up")
	raise e

model.save('quicktest.model')

print("succes ? !")
# In[ ]:




