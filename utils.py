import os
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from efficientnet.tfkeras import EfficientNetB0

from constants import *


def isPathAvailableAndNonEmpty(path):
    dir_is_existing = False
    if os.path.exists(path):
        if len(os.listdir(path)) == 0:
            dir_is_existing = False
        else:
            dir_is_existing = True
    else:
        dir_is_existing = False

    return dir_is_existing


def setGPUConfigurations():
    import tensorflow as tf
    # from tensorflow.compat.v1 import ConfigProto
    # from tensorflow.compat.v1 import InteractiveSession

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus),
                  "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    # config = ConfigProto()
    # config.gpu_options.allow_growth = True
    # session = InteractiveSession(config=config)
    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
    os.environ["TF_CPP_VMODULE"] = "gpu_process_state=10,gpu_cudamallocasync_allocator=10"

# Generate Accuracy and Loss Curve plots and save to directory


def saveAccAndLossFigures(history):
    # Plot results
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    sns.set_style("darkgrid")
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, acc, label='Training Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend(loc='lower right')
    plt.grid("on")
    plt.savefig(os.path.join(BENCHMARKS_CHECKPOINT_PATH,
                "figure_accuracy_curve.png"), bbox_inches='tight', dpi=100)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, loss, label='Training loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend(loc='upper right')
    plt.grid("on")
    plt.savefig(os.path.join(BENCHMARKS_CHECKPOINT_PATH,
                "figure_loss_curve.png"), bbox_inches='tight', dpi=100)


# Loading the best saved model from directory
def loadSavedModel():
    best_model = load_model(os.path.join(
        BENCHMARKS_CHECKPOINT_PATH, 'best_model.h5'))
    return best_model
