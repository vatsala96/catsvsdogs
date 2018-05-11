import numpy as np 
import os 
import cv2
from random import shuffle
from tqdm import tqdm
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

train_dir = "train"
test_dir = "test1"

img_size = 50
lr = 1e-3

MODEL_NAME = "dog_cat_{}_{}.model".format(lr,"2convbasic")

def label_img(img):
	word_label = img.split('.')[-3]

	if word_label == "cat":
		return [1,0]
	if word_label == "dog":
		return [0,1]


def create_train_data():
	training_data=[]
	for img in tqdm(os.listdir(train_dir)):
		label = label_img(img)
		img_path = os.path.join(train_dir,img)
		img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
		img = cv2.resize(img, (img_size,img_size))
		training_data.append([np.array(img),np.array(label)])

	shuffle(training_data)
	np.save('training_data.npy',training_data)

	return training_data

def create_test_data():
	testing_data=[]
	for img in tqdm(os.listdir(test_dir)):
		img_path = os.path.join(test_dir,img)
		img_num = img.split('.')[0]
		img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
		img = cv2.resize(img, (img_size,img_size))
		testing_data.append([np.array(img), img_num])

	shuffle(testing_data)
	np.save('testing_data.npy',testing_data)

	return testing_data


#train_data = create_train_data()

train_data = np.load('training_data.npy')

convnet = input_data(shape=[None,img_size,img_size,1],name = 'input')

convnet = conv_2d(convnet,32,5,activation = 'relu')
convnet = max_pool_2d(convnet,5)

convnet = conv_2d(convnet,64,5,activation='relu')
convnet = max_pool_2d(convnet,5)

convnet = fully_connected(convnet,1024,activation='relu')
convnet = dropout(convnet,0.7)

convnet = fully_connected(convnet,2,activation = 'softmax')
convnet = regression(convnet, optimizer = 'adam', learning_rate = lr, loss = 'categorical_crossentropy', name = 'targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')


train = train_data[:-500]
test = train_data[-500:]

X = np.array([i[0] for i in train]).reshape(-1,img_size,img_size,1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,img_size,img_size,1)
test_y = [i[1] for i in test]

print(model.fit({'input':X},{'targets':Y},n_epoch = 3,validation_set=({'input':test_x}, {'targets':test_y}), snapshot_step=500,show_metric = True, run_id = MODEL_NAME))

pred = model.predict(test_x[:50])
print(pred)
