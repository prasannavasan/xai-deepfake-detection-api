import cv2
import math
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.preprocessing import image
from tqdm import tqdm
import shutil
from mtcnn import MTCNN
from alibi.explainers import AnchorImage
import matplotlib.pyplot as plt
import json
from PIL import Image
import numpy as np
from skimage import transform
import time

from tensorflow.keras.models import load_model
from efficientnet.tfkeras import EfficientNetB0

from constants import *
from utils import setGPUConfigurations, loadSavedModel, isPathAvailableAndNonEmpty

TEST_PATH = "test\\"

# Function accepts a single video name and internally extracts frames and makes predictions on all frames


def predictVideo(video, model, FACE_FRAMES_PATH, TEMP_FRAMES_PATH):

    extractFrames(
        video, FACE_FRAMES_PATH, TEMP_FRAMES_PATH)

    prediction_face_frames = []
    frames_paths = []
    prediction_face_paths = []

    faces = os.listdir(FACE_FRAMES_PATH)
    faces.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    print('Prepraring for prediction on', video, '. Hang on tight ...')
    for face in tqdm(faces):
        img_path = os.path.join(FACE_FRAMES_PATH, face)

        img = image.load_img(img_path, target_size=(224, 224, 3))
        img = image.img_to_array(img)
        img = img / 255

        # img=cv2.imread(img_path)  # where f_path is the path to the image file
        # img=cv2.resize(img, (224,224))
        # img = image.img_to_array(img)
        # CV2 inputs images in BGR format in general when you train a model you may have
        # trained it with images in rgb format. If so you need to convert the cv2 image.
        # uncomment the line below if that is the case.
        # img *= 1./255
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # img= np.float32(img) /255

        # print(img.shape)
        # arr = np.array([img])
        # predictions=model.predict(arr)
        # print("single-", predictions)

        # this will give you an integer value
        # im = Image.open(img_path)
        # im = im.convert('RGB')
        # #Resizing into 128x128 because we trained the model with this image size.
        # im = im.resize((224,224))
        # img_array = np.array(im)
        # img_array = np.expand_dims(img_array, axis=0)
        # img_array = img_array /255

        # img = np.expand_dims(img, axis=0)
        # classes = model.predict(img)
        # result = np.argmax(classes)
        # print(result)

        # img2 = tf.keras.utils.load_img(
        #     img_path, target_size=(224, 224)
        # )
        # img_array = tf.keras.utils.img_to_array(img2)
        # img_array = img_array.astype('float32')/255.0
        # img_array = tf.expand_dims(img_array, 0) # Create a batch

        # predictions = model.predict(img_array)
        # prediction = int(model.predict(img_array)[0][0])
        # score = tf.nn.softmax(predictions[0])
        # print('sincle',predictions)
        # print(prediction)

        # input = tf.convert_to_tensor(img)
        # input = tf.image.resize(input,(224,224))
        # input = input[:,:,:]
        # input = input/255.0

        # img = input

        # img = transform.resize(img, (224, 224))
        # img = np.expand_dims(img, axis=0)

        # img = image.load_img(img_path, target_size=(224, 224))
        # img = image.img_to_array(img)
        # # img = img / 255
        # img = np.expand_dims(img, axis=0)

        # img = Image.open(img_path)
        # img = img.convert('RGB')
        # x = np.asarray(img, dtype='float32')
        # x = x.im
        # x = np.expand_dims(x, axis=0)

        prediction_face_frames.append(img)
        prediction_face_paths.append(img_path)

        img_path = os.path.join(TEMP_FRAMES_PATH, face)
        frames_paths.append(img_path)

    prediction_face_frames = np.array(prediction_face_frames)
    # prediction_face_frames = prediction_face_frames.astype(np.float32).reshape((-1,1))
    print(prediction_face_frames.shape)
    prediciton_arr = predictOnAllFramesInVideo(
        model, prediction_face_frames, frames_paths, prediction_face_paths)
    return prediciton_arr, prediction_face_frames

# Extract frames given a video


def extractFrames(video, FACE_FRAMES_PATH, TEMP_FRAMES_PATH):
    
    face_time = 0
    frame_time = 0
    
    start_frame = time.time()
    cap = cv2.VideoCapture(video)
    frameRate = math.floor(cap.get(
        cv2.CAP_PROP_FPS))  # extracting the frame rate of the video
    print('Extracting frames ...')  # eg: 29.97

    total_frames = math.floor(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(total_frames)
    if (total_frames > 300):
        total_frames = 300
    
    frame_time += time.time() - start_frame
    for i in range(total_frames):
        cap.grab()
        if i % math.floor(frameRate) == 0:
            start_frame = time.time()
            success, frame = cap.retrieve()
            if not success:
                continue

            if frame.shape[1] < 300:
                scale_factor = 2
            elif frame.shape[1] > 1000 and frame.shape[1] <= 1900:
                scale_factor = 0.5
            elif frame.shape[1] > 1900:
                scale_factor = 0.33
            else:
                scale_factor = 1

            width = int(frame.shape[1] * scale_factor)
            height = int(frame.shape[0] * scale_factor)
            dszie = (width, height)
            new_frame = cv2.resize(frame,
                                   dszie,
                                   interpolation=cv2.INTER_AREA)

            frame_img = str(i) + '.png'
            new_filename = os.path.join(TEMP_FRAMES_PATH, frame_img)
            # print(new_filename)
            cv2.imwrite(new_filename, new_frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 100])
            new_filename2 = os.path.join(FACE_FRAMES_PATH, frame_img)
            
            end_frame = time.time()
            per_frame_time = end_frame - start_frame
            frame_time += per_frame_time

            start_face = time.time()
            extractFaces(new_frame, new_filename, new_filename2)
            end_face = time.time()
            per_face_time = end_face-start_face
            face_time += per_face_time
    
    print("Frame extraction time", frame_time)
    print("Face extraction time", face_time)

# Extract face from a given frame into the destination path


def extractFaces(image, src_path, destpath):
    detector = MTCNN()
    image = cv2.cvtColor(cv2.imread(src_path), cv2.COLOR_BGR2RGB)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(image)
    # count = 0
    biggest = 0
    y1, y2, x1, x2 = 0, 0, 0, 0
    face_identied = False
    for face in faces:
        x, y, width, height = face['box']
        confidence = face['confidence']
        if confidence > 0.97:
            face_identied = True
            area = width * height
            # to select face with biggest no of pixels
            if area > biggest:
                biggest = area
                margin_x = int(width * 0.2)  # scale by a factor of 0.2
                margin_y = int(height * 0.2)  # scale by a factor of 0.2
                x1 = x - margin_x
                x2 = x + width + margin_x
                y1 = y - margin_y
                y2 = y + height + margin_y
                # prevent new coord point being outside image bounds due to addition of margin
                if x1 < 0:
                    x1 = 0
                if x2 > image.shape[1]:
                    x2 = image.shape[1]
                if y1 < 0:
                    y1 = 0
                if y2 > image.shape[0]:
                    y2 = image.shape[0]

        if (face_identied):
            crop_image = image[y1:y2, x1:x2]
            cv2.imwrite(destpath, cv2.cvtColor(crop_image, cv2.COLOR_RGB2BGR))

# Given a list of face frames, find avergae prediction for the video as a whole


def predictOnAllFramesInVideo(model, prediction_face_frames, frames_paths, faces_paths):
    start_predict = time.time()
    print('Prediction in progress...')
    frame_preds = []

    preds = model.predict(prediction_face_frames)
    # print(predicted_class)
    total_pred = 0
    i = 0
    for pred in preds:
        # print(pred)
        total_pred += pred
        frame_pred = {"label": np.argmax(
            pred, axis=-1), "pred": pred[np.argmax(pred, axis=-1)], "frame_path": frames_paths[i], "face_path": faces_paths[i]}
        frame_preds.append(frame_pred)
        i += 1

    avg_pred = total_pred / len(preds)
    pred_label = LABELS[np.argmax(avg_pred, axis=-1)]

    print("Avg prediction (Fake): ", avg_pred[0], "(Real):", avg_pred[1],
          ": Predicted label:", pred_label)
    end_predict = time.time()
    predict_time = end_predict - start_predict
    print("Prediction time", predict_time)
    arr = list(map(str, [avg_pred[0], avg_pred[1], pred_label, frame_preds]))
    return arr

# Find anchors for a given list of face frames


def getAnchorExplainations(model, prediction_face_frames, ANCHORS_PATH, SEGMENTS_PATH):
    start_anchor = time.time()

    def predict_fn(x): return model.predict(x)
    image_shape = (224, 224, 3)

    # segmentation_fn = 'slic'
    # kwargs = {'n_segments': 20, 'compactness': 20, 'sigma': .5}
    # explainer = AnchorImage(predict_fn,
    #                         image_shape,
    #                         segmentation_fn=segmentation_fn,
    #                         segmentation_kwargs=kwargs,
    #                         images_background=None)

    segmentation_fn = 'slic'
    args = {'n_segments': 35, 'compactness': 18, 'sigma': .5}
    # segmentation_fn = 'quickshift'
    # args = {'kernel_size': 4, 'max_dist': 200, 'ratio': 0.2}

    explainer = AnchorImage(predict_fn,
                            image_shape,
                            segmentation_fn=segmentation_fn,
                            segmentation_kwargs=args,
                            images_background=None)

    print("Generating anchor images ... ")

    anchor_paths = []

    for idx, image in enumerate(tqdm(prediction_face_frames)):
        explanation_img = explainer.explain(image, threshold=0.80, p_sample=0.5, tau=0.25, delta= 0.9, batch_size= 100, verbose="True", coverage_samples=500)

        explanation_anchor = explanation_img.anchor
        print("Precision:", explanation_img.precision,
              "Coverage:", explanation_img.coverage)
        all_zeros = not np.any(explanation_anchor)

        if not all_zeros:
            explanation_anchor = explanation_anchor.astype(np.float32)
            explanation_anchor /= 255

            anchor_file = str(idx) + '.png'
            img_path = os.path.join(
                ANCHORS_PATH, anchor_file)
            plt.imsave(img_path, explanation_anchor)
            plt.imsave(os.path.join(SEGMENTS_PATH, anchor_file),
                       explanation_img.segments)
            anchor_paths.append(img_path)
        else:
            anchor_paths.append("None")
    end_anchor = time.time()
    anchor_time = end_anchor- start_anchor
    print("Anchors time", anchor_time)
    return anchor_paths


def getPredictionForVideo(videoName, videoUrl):
    # setGPUConfigurations()
    model = load_model(
        "E:\\Deepfake\\xai-deepfake-detection-web\\api\\model_checkpoints\\new.h5")
    video = videoName

    ANCHORS_PATH = os.path.join(TEST_PATH, video, "anchor")
    TEMP_FRAMES_PATH = os.path.join(TEST_PATH, video, "frames")
    FACE_FRAMES_PATH = os.path.join(TEST_PATH, video, "faces")
    os.makedirs(ANCHORS_PATH, exist_ok=True)
    os.makedirs(TEMP_FRAMES_PATH, exist_ok=True)
    os.makedirs(FACE_FRAMES_PATH, exist_ok=True)

    print("-----------------------------------")
    print(video)
    print("-----------------------------------")

    prediciton_arr, predicition_face_frames = predictVideo(
        videoUrl, model, FACE_FRAMES_PATH, TEMP_FRAMES_PATH)
    # anchor_paths = getAnchorExplainations(
    #     model, predicition_face_frames, ANCHORS_PATH)
    # return json.dumps(prediciton_arr), anchor_paths


video = 'fake1'
videoname = video
video_url = video + '.mp4'
getPredictionForVideo(videoname, video_url)
model = load_model(
    "E:\\Deepfake\\xai-deepfake-detection-web\\api\\model_checkpoints\\new.h5")
ANCHORS_PATH = os.path.join(TEST_PATH, videoname, "anchor")
FACE_FRAMES_PATH = os.path.join(TEST_PATH, videoname, "faces")
SEGMENTS_PATH = os.path.join(TEST_PATH, videoname, "segments")

os.makedirs(ANCHORS_PATH, exist_ok=True)
os.makedirs(SEGMENTS_PATH, exist_ok=True)

faces = os.listdir(FACE_FRAMES_PATH)
faces.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
prediction_face_frames = []
for face in tqdm(faces):
    img_path = os.path.join(FACE_FRAMES_PATH, face)

    img = image.load_img(img_path, target_size=(224, 224, 3))
    img = image.img_to_array(img)
    img = img / 255
    prediction_face_frames.append(img)

prediction_face_frames = np.array(prediction_face_frames)

getAnchorExplainations(model, prediction_face_frames,
                       ANCHORS_PATH, SEGMENTS_PATH)
