import keras
from keras.applications import inception_v3,nasnet,mobilenet,vgg19,resnet50,xception,densenet
import keras.backend as K

from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,SeparableConv2D,Dropout,Flatten,Dense,BatchNormalization,GlobalAveragePooling2D
from keras import layers,models
from keras import initializers
from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,CSVLogger
from keras.optimizers import Adam,RMSprop

datagen = ImageDataGenerator(rescale=1.0/255,
                horizontal_flip=True,
                vertical_flip=True)

#resizing all the images to 96 x 96
train_gen = datagen.flow_from_directory('train/' , 
                                        target_size = (128,128) , 
                                        batch_size = 32,
                                       class_mode ='binary',
                                       shuffle = True)
def pretrained_model(model):
    if model == 'densenet':
        base_model = densenet.DenseNet121(include_top=False,weights='imagenet',input_shape = (128,128,3))
    elif model == 'inception':
        base_model = inception_v3.InceptionV3(include_top=False,weights='imagenet',input_shape = (128,128,3))
    elif model == 'mobilenet':
        base_model = mobilenet.MobileNet(include_top=False,weights='imagenet',input_shape = (128,128,3))
    elif model == 'vgg':
        base_model = vgg19.VGG19(include_top=False,weights='imagenet',input_shape = (128,128,3))
    elif model == 'resnet':
        base_model = resnet50.ResNet50(include_top=False,weights='imagenet',input_shape = (128,128,3))
    elif model == 'xception':
        base_model = xception.Xception(include_top=False,weights='imagenet',input_shape = (128,128,3))
        
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
checkpoint = ModelCheckpoint(checkpoint_fp,monitor='acc',
                             verbose=1,
                            save_best_only= True,mode='max')

learning_rate = ReduceLROnPlateau(monitor='acc',
                                 factor = 0.1,
                                 patience = 2,
                                 verbose = 1,
                                 mode = 'max',
                                 min_lr = 0.00001)

callback = [checkpoint,learning_rate,csv_logger]

steps_p_ep_tr = 1013//32

main_model.compile(optimizer = Adam(lr=0.0001), 
              loss = 'binary_crossentropy', metrics=['accuracy'])

my_model = main_model.fit_generator(train_gen,
                                   steps_per_epoch = steps_p_ep_tr,
                                   verbose = 1,
                                   epochs = 100,
                                   callbacks = callback)

my_model.save_weights('vgg_model_epoch_100.h5')
