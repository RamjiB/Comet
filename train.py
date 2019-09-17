import os,glob,cv2
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Convolution2D,Activation,Dense,MaxPooling2D,Flatten,Dropout,LeakyReLU
from keras.callbacks import ModelCheckpoint,CSVLogger,ReduceLROnPlateau
from keras.optimizers import Adam,RMSprop
from sklearn.metrics import accuracy_score,confusion_matrix

#train and valid paths
train_path = 'train/'
valid_path = 'valid/'
test_csv_path = 'csv/test_df.csv'
prediction_csv = 'csv/trail_5_prediction.csv'

IMG_SIZE = 128
CHANNEL = 3
ALPHA = 0.002
OUTPUT_LAYER = 1
CHECKPOINT = 'models/trail_5.h5'
CSVFILE = 'csv/trail_5.csv'
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001

#read and store images along with its correcting labels
#damaged - 0, undamaged - 1
def read_images(path = train_path,mode = 'train'):
	X = []
	Y = []
	class_names = os.listdir(path)
	for cl in class_names:
		img_paths = glob.glob(path+cl+'/*.png')
		for img_path in img_paths:
			img = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)
			img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))

			if mode == 'train':
				X.append(img)
				if cl =='damaged':
					Y.append(0)
				else:
					Y.append(1)
			elif mode == 'valid':
				X.append(img)
				if cl =='damaged':
					Y.append(0)
				else:
					Y.append(1)
	X = np.reshape(np.array(X),(len(X),IMG_SIZE,IMG_SIZE,CHANNEL))
	Y = np.reshape(np.array(Y),(len(Y),1))
	return X,Y

#reading train and valid images
x_train,y_train = read_images(train_path,'train')
x_valid,y_valid = read_images(valid_path,'valid')

#normalization
x_train = x_train/255
x_valid = x_valid/255

print(x_valid.shape,x_train.shape,y_train.shape,y_valid.shape)

#model creation
model = Sequential()
model.add(Convolution2D(64,kernel_size=(3,3),padding = 'Same',input_shape=(IMG_SIZE,IMG_SIZE,CHANNEL)))
model.add(LeakyReLU(alpha = ALPHA))
model.add(Convolution2D(64,kernel_size=(3,3),padding = 'valid'))
model.add(LeakyReLU(alpha = ALPHA))
model.add(Flatten())
model.add(Dense(256,activation = 'relu'))
model.add(Dense(128,activation='relu'))
# model.add(Dropout(0.3))
model.add(Dense(OUTPUT_LAYER,activation = 'sigmoid'))

model.summary()

#callbacks

checkpoint = ModelCheckpoint(CHECKPOINT,monitor = 'val_acc',save_best_only = True,verbose = 1)
csv = CSVLogger(CSVFILE,separator = ",",append= True)
lr = ReduceLROnPlateau(monitor = 'val_acc',factor = 0.1,patience = 2,verbose = 1,mode = 'max')

model.compile(optimizer = Adam(lr = LEARNING_RATE),loss='binary_crossentropy',metrics = ['accuracy'])
model.fit(x_train,y_train,batch_size = BATCH_SIZE,epochs=EPOCHS,validation_data = (x_valid,y_valid),verbose=1,callbacks = [checkpoint,csv,lr])

#save the last model
model.save_weights('models/trail_5_' + str(EPOCHS) + '.h5')	

print('------------------------------- model  trained ----------------------------')
#TEST DATA

#read test data
test_df = pd.read_csv(test_csv_path)
x_test = []
for i in test_df['img_path']:
	img = cv2.cvtColor(cv2.imread(i),cv2.COLOR_BGR2RGB)
	img = cv2.resize(img,(128,128))
	x_test.append(img)
x_test = np.reshape(np.array(x_test),(len(x_test),IMG_SIZE,IMG_SIZE,CHANNEL))

#normalization
x_test = x_test/255

y_test = test_df['class_name']
print('------------- test data shape ----------------------')
print(x_test.shape,y_test.shape)

#predictions
prediction = model.predict_classes(x_test,batch_size= BATCH_SIZE,verbose = 1)

print(prediction.shape)
print('--------------------- prediction dataframe -----------------------------')
df = pd.DataFrame(columns = ['img_path','predictions'])
df['img_path'] = test_df['img_path']
df['predictions'] = prediction

df.to_csv(prediction_csv,index = False)

print(accuracy_score(y_test,prediction))
print(confusion_matrix(y_test,prediction))

