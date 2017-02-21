from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.preprocessing import image

import cv2
import json
import numpy as np


class Vgg:
    def __init__(self):
        print "Hello from Vgg()"
        self.create()  # create model

    def create(self):
        model = self.model = Sequential()

        model.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))
        model.add(Convolution2D(64, 3, 3, activation='relu'))  # layer 1
        model.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))
        model.add(Convolution2D(64, 3, 3, activation='relu'))  # layer2
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))  #

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))  # layer3
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))  # layer4
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))  # layer5
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))  # layer6
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))  # layer7
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))  # layer8
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))  # layer9
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))  # layer10
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))  # layer11
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))  # layer12
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))  # layer13
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))  # layer14 fully connected (inner_product)
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))  # layer15 (inner_product)
        model.add(Dropout(0.5))
        model.add(Dense(1000, activation='softmax'))  # layer16 fully connected (inner_product)
        self.compile(lr=0.01)

    def load_imagenet_class_index(self, imagenet_file):
        with open(imagenet_file) as data_file:
            self.imagenet_index = json.load(data_file)

    def load_weights(self, weights_file):
        self.model.load_weights(weights_file)

    def predict(self, img):
        im = cv2.resize(img, (224, 224)).astype(np.float32)
        im[:, :, 0] -= 103.939
        im[:, :, 1] -= 116.779
        im[:, :, 2] -= 123.68
        im = im.transpose((2, 0, 1))
        im = np.expand_dims(im, axis=0)
        return self.model.predict(im)

    def get_batches(self, path, gen=image.ImageDataGenerator(), shuffle=True, batch_size=8, class_mode='categorical'):
        return gen.flow_from_directory(path,target_size=(224,224), class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)

    def finetune(self, num_classes):
        model = self.model
        model.pop()
        for layer in model.layers: layer.trainable=False
        model.add(Dense(num_classes, activation='softmax'))
        self.compile()

    def compile(self, lr=0.001):
        sgd = SGD(lr, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    def fit(self, train_batches, val_batches, nb_epoch=1):
        self.model.fit_generator(train_batches, samples_per_epoch=train_batches.nb_sample, nb_epoch=nb_epoch,
                                 validation_data=val_batches, nb_val_samples=val_batches.nb_sample)

    def predict(self, path, batch_size=8):
        test_batches = self.get_batches(path, shuffle=False, batch_size=batch_size, class_mode=None)
        return test_batches, self.model.predict_generator(test_batches, test_batches.nb_sample)
    