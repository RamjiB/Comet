import os,glob,cv2
import numpy as np

#train and valid paths
train_path = 'train/'
valid_path = 'valid/'

IMG_SIZE = 128
CHANNEL = 3

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


print(x_train.shape,x_valid.shape,y_train.shape,y_valid.shape)		
