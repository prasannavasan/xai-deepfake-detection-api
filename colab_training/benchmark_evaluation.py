import os
from tensorflow.keras.models import load_model
from efficientnet.tfkeras import EfficientNetB0
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import datetime


from utils import setGPUConfigurations, loadSavedModel
from constants import *

SPLIT_DATASET_PATH = '/content/sample_data/splitdataset/'
BENCHMARKS_CHECKPOINT_PATH = '/content/gdrive/MyDrive/ColabStuff/xai-deepfake-detection/benchmarks_model_checkpoints'
test_path = os.path.join(SPLIT_DATASET_PATH, 'test')

# Function to evaluate the model on test data


def evaluateModel(model, model_name):
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(directory=test_path,
                                                      target_size=(INPUT_SIZE,
                                                                   INPUT_SIZE),
                                                      color_mode="rgb",
                                                      class_mode="categorical",
                                                      batch_size=1,
                                                      shuffle=False)

    log_dir = "/content/gdrive/MyDrive/ColabStuff/xai-deepfake-detection/model_checkpoints/logs/" + \
        model_name + "_evaluate/"

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1, write_images=True)
    loss, accuracy = model.evaluate(test_generator,
                                    steps=None,
                                    workers=3,
                                    verbose=1,
                                    callbacks=[tensorboard_callback])

    print("Loss:", loss, "; Accuracy:", accuracy)
    return test_generator


# Function generates predictions on the test data and uses info to
# plot confusion matrix and printing classification report
def predictOnTestData(model, test_generator, model_name):
    preds = model.predict(test_generator)

    predicted_labels = np.argmax(preds, axis=-1)  # find highest class
    actual_labels = test_generator.classes

    print("Label indices:", test_generator.class_indices)

    plotConfusionMatrix(actual_labels, predicted_labels, False, model_name)
    printClassificationReport(actual_labels, predicted_labels, False)

    return preds

# Generates predictions for unique videos in the test set


def viewPredictionsOnVideoBasis(test_generator, preds, model_name):
    # Identifying unique videos by looking at the frame name
    uniquevideos = []
    for videoframe in test_generator.filenames:
        frame_basename = videoframe.split("-")
        if frame_basename[0] not in uniquevideos:
            uniquevideos.append(frame_basename[0])

    actual_labels = []
    predicted_labels = []
    for video in uniquevideos:
        total_pred = 0
        count = 0
        # Aggregates scores of all frames of the video and averages to give the final prediction for the video
        for idx, videoframe in enumerate(test_generator.filenames):
            if video in videoframe:
                total_pred += preds[idx]
                count += 1

        avg_pred = total_pred / count
        actual_label = video.split("/")[0]
        actual_label = actual_label.title()
        pred_label = LABELS[np.argmax(avg_pred, axis=-1)]

        # print("Video -", video, " Avg prediction:", max(avg_pred),
        #       ": Predicted label:", pred_label, '[ Actual Label:', actual_label, ']')

        predicted_labels.append(np.argmax(avg_pred, axis=-1))
        actual_labels.append(LABELS.index(actual_label))

    print("-------------------------------------------------------------------------------")
    plotConfusionMatrix(actual_labels, predicted_labels, True, model_name)
    printClassificationReport(actual_labels, predicted_labels, True)
    return actual_labels, predicted_labels


# Classification report including f1 score, recall, precision etc are printed
def printClassificationReport(actual_labels, predicted_labels, isFullVideo):
    if isFullVideo:
        text = "(Video wise)"
    else:
        text = "(Frame wise)"
    print("----------------------------------------------------------------------------------")
    print("Classification report:", text)
    print("----------------------------------------------------------------------------------")
    print(
        classification_report(actual_labels,
                              predicted_labels,
                              target_names=['Fake', 'Real']))


# Code adapted fom https://www.stackvidhya.com/plot-confusion-matrix-in-python-and-why/
def plotConfusionMatrix(actual_labels, predicted_labels, isFullVideo, model_name):
    cm = confusion_matrix(y_true=actual_labels, y_pred=predicted_labels)
    cm_plot_labels = ['fake', 'real']

    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]

    group_percentages = [
        "{0:.2%}".format(value) for value in cm.flatten() / np.sum(cm)
    ]

    labels = [
        f"{v1}\n{v2}\n{v3}"
        for v1, v2, v3 in zip(group_names, group_counts, group_percentages)
    ]

    labels = np.asarray(labels).reshape(2, 2)
    ax = sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')

    if (isFullVideo):
        filename = "figure_confusion_matrix_" + model_name + "video.png"
        ax.set_title('Confusion Matrix (Video wise) \n\n')
    else:
        filename = "figure_confusion_matrix_" + model_name + "frames.png"
        ax.set_title('Confusion Matrix (Frames wise) \n\n')

    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')

    ax.xaxis.set_ticklabels(cm_plot_labels)
    ax.yaxis.set_ticklabels(cm_plot_labels)

    fig = ax.get_figure()

    fig.savefig(os.path.join(BENCHMARKS_CHECKPOINT_PATH,
                filename), bbox_inches='tight', dpi=100)
    fig.clf()
    # plt.show()


setGPUConfigurations()

# Xception Model
print("Xception Model")
print("-------------------------------------------------")
model = load_model(os.path.join(
    BENCHMARKS_CHECKPOINT_PATH, 'xception_model.h5'))
test_gen = evaluateModel(model, "xception")
preds = predictOnTestData(model, test_gen, "xception")
actual_labels_xception, predicted_labels_xception = viewPredictionsOnVideoBasis(
    test_gen, preds, "xception")

# ResNet50 Model
print("ResNet50 Model")
print("-------------------------------------------------")
model = load_model(os.path.join(
    BENCHMARKS_CHECKPOINT_PATH, 'resnet_model.h5'))
test_gen = evaluateModel(model, "resnet50")
preds = predictOnTestData(model, test_gen, "resnet50")
actual_labels_resnet, predicted_labels_resnet = viewPredictionsOnVideoBasis(
    test_gen, preds, "resnet50")

# VGG16 Model
print("VGG16 Model")
print("-------------------------------------------------")
model = load_model(os.path.join(
    BENCHMARKS_CHECKPOINT_PATH, 'vgg16_model.h5'))
test_gen = evaluateModel(model, "vgg16")
preds = predictOnTestData(model, test_gen, "vgg16")
actual_labels_vgg16, predicted_labels_vgg16 = viewPredictionsOnVideoBasis(
    test_gen, preds, "vgg16")

# Main Model
print("Main Model")
print("-------------------------------------------------")
MODEL_CHECKPOINT_PATH = '/content/gdrive/MyDrive/ColabStuff/xai-deepfake-detection/model_checkpoints'
model = load_model(os.path.join(MODEL_CHECKPOINT_PATH, 'best_model.h5'))
test_gen = evaluateModel(model, "efficientnet")
preds = predictOnTestData(model, test_gen, "efficientnet")
actual_labels_ours, predicted_labels_ours = viewPredictionsOnVideoBasis(
    test_gen, preds, "efficientnet")


# PLOT ROC CURVE

# For Xception
fpr_xception, tpr_xception, thresholds_xception = roc_curve(
    actual_labels_xception, predicted_labels_xception)
auc_xception = auc(fpr_xception, tpr_xception)

# For ResNet50
fpr_resnet, tpr_resnet, thresholds_resnet = roc_curve(
    actual_labels_resnet, predicted_labels_resnet)
auc_resnet = auc(fpr_resnet, tpr_resnet)

# For VGG16
fpr_vgg16, tpr_vgg16, thresholds_vgg16 = roc_curve(
    actual_labels_vgg16, predicted_labels_vgg16)
auc_vgg16 = auc(fpr_vgg16, tpr_vgg16)

# For main efficient net model
fpr_ours, tpr_ours, thresholds_ours = roc_curve(
    actual_labels_ours, predicted_labels_ours)
auc_ours = auc(fpr_ours, tpr_ours)


plt.figure()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_xception, tpr_xception,
         label='Xception (Baseline) (area = {:.3f})'.format(auc_xception))
plt.plot(fpr_resnet, tpr_resnet,
         label='ResNet50 (Baseline) (area = {:.3f})'.format(auc_resnet))
plt.plot(fpr_vgg16, tpr_vgg16,
         label='VGG16 (Baseline) (area = {:.3f})'.format(auc_vgg16))
plt.plot(fpr_ours, tpr_ours,
         label='EfficientNet (Ours) (area = {:.3f})'.format(auc_ours))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.savefig(os.path.join(BENCHMARKS_CHECKPOINT_PATH,
            "ROC Curve (Baseline_Ours).png"), bbox_inches='tight', dpi=100)
plt.close()
# plt.show()
