import os
from sklearn.ensemble import VotingRegressor
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Adadelta
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
import datetime

import json

from utils import setGPUConfigurations, saveAccAndLossFigures
from constants import *

SPLIT_DATASET_PATH = '/content/sample_data/splitdataset/'

train_path = os.path.join(SPLIT_DATASET_PATH, 'train')
val_path = os.path.join(SPLIT_DATASET_PATH, 'val')

# https://github.com/i3p9/deepfake-detection-with-xception/blob/main/train_dateset.py

CHECKPOINT_PATH = '/content/gdrive/MyDrive/ColabStuff/xai-deepfake-detection/benchmarks_model_checkpoints'

# Creating directory to hold best models


def createModelCheckpointsFolder():
    print('Creating Directory: ' + CHECKPOINT_PATH)
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

# Printing history params to file


def printAndSaveHistoryToFile(history):
    print(history)
    history_dict = history.history
    json.dump(history_dict,
              open(os.path.join(CHECKPOINT_PATH, "vgg19_history.json"), 'w'))

# Build model structure


def buildModel():
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=30,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       brightness_range=[0.2, 0.9],
                                       shear_range=0.2,
                                       zoom_range=0.3,
                                       horizontal_flip=True,
                                       fill_mode='nearest')

    train_generator = train_datagen.flow_from_directory(
        directory=train_path,
        target_size=(224, 224),
        class_mode="categorical",  # "categorical", "binary", "sparse", "input"
        batch_size=BATCH_SIZE,
        shuffle=True,
        # save_to_dir = 'scratchall\\augmentations'
    )

    val_datagen = ImageDataGenerator(rescale=1. /
                                     255  # rescale the tensor values to [0,1]
                                     )

    val_generator = val_datagen.flow_from_directory(
        directory=val_path,
        target_size=(224, 224),
        color_mode="rgb",
        class_mode="categorical",  # "categorical", "binary", "sparse", "input"
        batch_size=BATCH_SIZE,
        shuffle=True
        #save_to_dir = tmp_debug_path
    )

    base_model = VGG19(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3)
    )

    # create a custom top classifier
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)
    model = Model(inputs=base_model.inputs, outputs=predictions)
    model.summary()
    print("Compiling Model...")
    model.compile(optimizer=Adam(learning_rate=5e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # model.compile(optimizer=Adadelta(learning_rate=0.01, rho=0.95, epsilon=1e-07, name="Adadelta"))
    return model, train_generator, val_generator

    # process_eval = multiprocessing.Process(target=evaluate, args=(...))
    # process_eval.start()
    # process_eval.join()

# Custom callbacks to stop training when model not improving and save best model to directory


def defineCustomCallbacks():
    log_dir = "/content/gdrive/MyDrive/ColabStuff/xai-deepfake-detection/model_checkpoints/logs/vgg19"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1)

    custom_callbacks = [
        EarlyStopping(monitor='val_loss', mode='min', patience=3, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                          patience=2, min_lr=0.00001),
        tensorboard_callback,
        ModelCheckpoint(filepath=os.path.join(CHECKPOINT_PATH,
                                              'vgg19_model.h5'),
                        monitor='val_loss',
                        mode='min',
                        verbose=1,
                        save_best_only=True)
    ]
    return custom_callbacks

# Training the neural network


def trainCNN(model, train_generator, val_generator, custom_callbacks, initial_epoch):
    history = model.fit(train_generator,
                        epochs=NUM_EPOCHS,
                        steps_per_epoch=len(train_generator),
                        validation_data=val_generator,
                        validation_steps=len(val_generator),
                        callbacks=custom_callbacks,
                        initial_epoch=initial_epoch)
    # printAndSaveHistoryToFile(history)
    saveAccAndLossFigures(history)


setGPUConfigurations()
createModelCheckpointsFolder()

custom_callbacks = defineCustomCallbacks()

# Load checkpoint:
checkpoint_path = os.path.join(CHECKPOINT_PATH, "vgg19_model.h5")
model, train_generator, val_generator = buildModel()

if os.path.exists(checkpoint_path) is True:
    # Load model:
    model = load_model(checkpoint_path)
    # Finding the epoch index from which we are resuming
    initial_epoch = 16
else:
    initial_epoch = 0

# Start/resume training
trainCNN(model, train_generator, val_generator,
         custom_callbacks, initial_epoch)
