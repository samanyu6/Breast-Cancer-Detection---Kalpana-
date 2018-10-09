#Importing the required libraries
import cv2
import os
import numpy as np
from random import shuffle
from tqdm import tqdm

TRAIN_DIR = '/home/samanyu/Documents/kalpana2/ptrain'
TEST_DIR = '/home/samanyu/Documents/kalpana2/ptest'
IMG_SIZE = 500
LR = 1e-3

'''Setting up the model which will help with tensorflow models'''
MODEL_NAME = 'malignant-vs-benign-{}-{}.model'.format(LR, '6conv-basic')

#M or B as mentioned in the text file provided by the Wisconsin Dataset
def label_img(img):
	label_file = open('/home/samanyu/Documents/kalpana2/label.txt','r')
	img_name = img.split('.')[0]
	content = label_file.read().split('\n')
	for s in content:
		if img_name in s:
			s= s.split(' ')
			try:
				s[4] == 'M'
			except Exception as e:
				s_label = [0,1]
			else:
				s_label = [1,0]

			return s_label
			
'''Creating the training data'''
def create_train_data():
    #Creating an empty list where we should the store the training data
    #after a little preprocessing of the data
    training_data = []

    #tqdm is only used for interactive loading
    #loading the training data
    for img in tqdm(os.listdir(TRAIN_DIR)):
        #labeling the images
        label = label_img(img)
        path = os.path.join(TRAIN_DIR,img)
        #loading the image from the path and then converting them into
        #greyscale for easier covnet prob
        img = cv2.imread(path,0)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)])
    #shuffling of the training data to preserve the random state of our data
    shuffle(training_data)
    #saving our trained data for further uses if required(not an necessary step)
    np.save('train_data.npy', training_data)
    return training_data
'''Processing the given test data'''
#Almost same as processing the traning data but
#we dont have to label it.
def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
        img_num = img.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img), img_num])
        
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data

'''Running the training and the testing in the train and test dataset for our model'''
train_data = create_train_data()
test_data = process_test_data()
#train_data = np.load('train_data.npy')
#test_data = np.load('test_data.npy')
'''Creating the neural network using tensorflow'''
#Importing the required libraries
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

import tensorflow as tf
tf.reset_default_graph()
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

#Loading the saved model
if os.path.exists('/home/samanyu/Documents/kalpana2/Cov_Net{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')

#Splitting the testing data and training data
train = train_data[:220]
test = train_data[220:]

'''Setting up the features and lables'''
#X-Features & Y-Labels

X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y = [i[1] for i in train]
test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y = [i[1] for i in test]

'''Fitting the data into our model'''
#epoch = 5 taken
model.fit({'input': X}, {'targets': Y}, n_epoch=5, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
model.save(MODEL_NAME)

'''Testing the data'''
import matplotlib.pyplot as plt
# if you need to create the data:
#test_data = process_test_data()
# if you already have some saved:
test_data = np.load('test_data.npy')

fig=plt.figure()

for num,data in enumerate(test_data[:20]):
    
    img_num = data[1]
    img_data = data[0]
    
    y = fig.add_subplot(4,5,num+1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
    #model_out = model.predict([data])[0]
    model_out = model.predict([data])[0]
    print(model_out)
    if np.argmax(model_out) == 1:
        str_label='M'
    else:
        str_label='B'
        
    y.imshow(orig,cmap='gray')
    plt.title(str_label+','+img_num)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
    print()
plt.show()













"""
x = 'mdb001 G CIRC B 535 425 197'
y = 'mdb006 F NORM'
w = 'mb134 F MISC M 469 728 49'

x= x.split(' ')
y = y.split(' ')

z = w
try:
	z[4] == 'M'
except Exception as e:
	z_label = 0
else:
	z_label = 1

print(z_label)
"""
 


