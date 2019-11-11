import glob
from keras.applications import inception_v3,mobilenet,vgg19,resnet50,xception,densenet
from keras.layers import Conv2D,MaxPool2D,Dropout,Flatten,Dense,
from keras import models
from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,CSVLogger
from keras.optimizers import Adam

total_train_images = len(glob.glob('train/damaged/*')) + len(glob.glob('train/undamaged/*'))
total_valid_images = len(glob.glob('valid/damaged/*')) + len(glob.glob('valid/undamaged/*'))
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 100

datagen = ImageDataGenerator(rescale=1.0/255,
                horizontal_flip=True,
                vertical_flip=True)

#resizing all the images to 128 x 128
train_gen = datagen.flow_from_directory('train/' ,
                                        target_size = (IMG_SIZE,IMG_SIZE) ,
                                        batch_size = BATCH_SIZE,
                                       class_mode ='binary',
                                       shuffle = True)
valid_gen = datagen.flow_from_directory('valid/' ,
                                        target_size =(IMG_SIZE,IMG_SIZE) , 
                                        batch_size = BATCH_SIZE,
                                       class_mode ='binary',
                                       shuffle = True)
def pretrained_model(model):
    if model == 'densenet':
        base_model = densenet.DenseNet121(include_top=False,weights='imagenet',input_shape = (IMG_SIZE,IMG_SIZE,3))
    elif model == 'inception':
        base_model = inception_v3.InceptionV3(include_top=False,weights='imagenet',input_shape = (IMG_SIZE,IMG_SIZE,3))
    elif model == 'mobilenet':
        base_model = mobilenet.MobileNet(include_top=False,weights='imagenet',input_shape = (IMG_SIZE,IMG_SIZE,3))
    elif model == 'vgg':
        base_model = vgg19.VGG19(include_top=False,weights='imagenet',input_shape = (IMG_SIZE,IMG_SIZE,3))
    elif model == 'resnet':
        base_model = resnet50.ResNet50(include_top=False,weights='imagenet',input_shape = (IMG_SIZE,IMG_SIZE,3))
    elif model == 'xception':
        base_model = xception.Xception(include_top=False,weights='imagenet',input_shape = (IMG_SIZE,IMG_SIZE,3))
        
    for layer in base_model.layers:
        layer.trainable = True 
    x = base_model.output
    x = Flatten()(x)
    x = Dense(150,activation='relu')(x)
    x = Dropout(0.2)(x)
    predictions = Dense(1,activation='sigmoid')(x)

    return models.Model(base_model.input,predictions)

main_model = pretrained_model('vgg')
main_model.summary()

csv_logger = CSVLogger("vgg_model.csv",separator = ",",append=True)

checkpoint_fp = "vgg_model_best.h5"
checkpoint = ModelCheckpoint(checkpoint_fp,monitor='val_acc',
                             verbose=1,
                            save_best_only= True,mode='max')

learning_rate = ReduceLROnPlateau(monitor='acc',
                                 factor = 0.1,
                                 patience = 2,
                                 verbose = 1,
                                 mode = 'max',
                                 min_lr = 0.00001)

callback = [checkpoint,learning_rate,csv_logger]

steps_p_ep_tr = total_train_images//BATCH_SIZE
steps_p_ep_va = total_valid_images//BATCH_SIZE

main_model.compile(optimizer = Adam(lr=0.0001),
              loss = 'binary_crossentropy', metrics=['accuracy'])

my_model = main_model.fit_generator(train_gen,
                                   steps_per_epoch = steps_p_ep_tr,
				   validation_data = valid_gen,
                                   validation_steps = steps_p_ep_va,
                                   verbose = 1,
                                   epochs = 100,
                                   callbacks = callback)

my_model.save_weights('vgg_model_epoch_100.h5')
