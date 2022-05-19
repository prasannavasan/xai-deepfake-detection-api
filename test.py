import cv2
import math
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.preprocessing import image
from tqdm import tqdm
import shutil
from mtcnn import MTCNN
from alibi.explainers import AnchorImage
import matplotlib.pyplot as plt
import json

from constants import *
from utils import setGPUConfigurations, loadSavedModel, isPathAvailableAndNonEmpty

TEST_PATH = "test\\"

# Function accepts a single video name and internally extracts frames and makes predictions on all frames


def predictVideo(video, model, FACE_FRAMES_PATH, TEMP_FRAMES_PATH):
    extractFrames(
        video, FACE_FRAMES_PATH, TEMP_FRAMES_PATH)

    prediction_frames = []

    prediction_frames_paths = []
    prediction_face_paths = []

    faces = os.listdir(FACE_FRAMES_PATH)
    faces.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    print('Prepraring for prediction on', video, '. Hang on tight ...')
    for face in tqdm(faces):
        img_path = os.path.join(FACE_FRAMES_PATH, face)
        img = image.load_img(img_path, target_size=(224, 224, 3))
        img = image.img_to_array(img)
        img = img / 255
        prediction_frames.append(img)
        prediction_face_paths.append(img_path)

        img_path = os.path.join(TEMP_FRAMES_PATH, face)
        prediction_frames_paths.append(img_path)

    prediction_frames = np.array(prediction_frames)
    prediciton_arr = predictOnAllFramesInVideo(
        model, prediction_frames, prediction_frames_paths, prediction_face_paths)
    return prediciton_arr, prediction_frames

# Extract frames given a video


def extractFrames(video, FACE_FRAMES_PATH, TEMP_FRAMES_PATH):
    cap = cv2.VideoCapture(video)
    frameRate = math.floor(cap.get(
        cv2.CAP_PROP_FPS))  # extracting the frame rate of the video
    print('Extracting frames ...')  # eg: 29.97

    total_frames = math.floor(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if (total_frames > 300):
        total_frames = 300

    for i in range(total_frames):
        cap.grab()
        if i % math.floor(frameRate) == 0:
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
            new_frame = cv2.resize(frame, dszie, interpolation=cv2.INTER_AREA)

            frame_img = str(i) + '.png'
            new_filename = os.path.join(TEMP_FRAMES_PATH, frame_img)
            cv2.imwrite(new_filename, new_frame, [
                        cv2.IMWRITE_JPEG_QUALITY, 100])
            new_filename2 = os.path.join(FACE_FRAMES_PATH, frame_img)
            extractFaces(new_frame, new_filename2)


# Extract face from a given frame into the destination path


def extractFaces(image, destpath):
    detector = MTCNN()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(image)
    # count = 0
    biggest = 0
    y1, y2, x1, x2 = 0, 0, 0, 0
    face_identied = False
    for face in faces:
        x, y, width, height = face['box']
        confidence = face['confidence']
        if confidence > 0.97:  # to select the face with the biggest no of pixels
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


def predictOnAllFramesInVideo(model, prediction_frames, frames_paths, faces_paths):
    print('Prediction in progress...')
    frame_preds = []
    preds = model.predict(prediction_frames)
    total_pred = 0
    i = 0
    for pred in preds:
        total_pred += pred
        frame_pred = {"label": np.argmax(
            pred, axis=-1), "label_text": LABELS[np.argmax(
                pred, axis=-1)], "pred": pred[np.argmax(pred, axis=-1)], "frame_path": frames_paths[i], "face_path": faces_paths[i]}
        frame_preds.append(frame_pred)
        i += 1

    avg_pred = total_pred / len(preds)
    pred_label = np.argmax(avg_pred, axis=-1)
    pred_label_text = LABELS[np.argmax(avg_pred, axis=-1)]

    print("Avg prediction (Fake): ", avg_pred[0], "(Real):", avg_pred[1],
          ": Predicted label:", pred_label_text)
    arr = list(map(str, [avg_pred[0], avg_pred[1],
               pred_label, pred_label_text, frame_preds]))
    return arr

# Find anchors for a given list of face frames


def getAnchorExplainations(model, prediction_frames, ANCHORS_PATH, SEGMENTS_PATH):

    def predict_fn(x): return model.predict(x)
    image_shape = (224, 224, 3)
    segmentation_fn = 'slic'
    args = {'n_segments': 35, 'compactness': 18, 'sigma': .5}
    explainer = AnchorImage(predict_fn,
                            image_shape,
                            segmentation_fn=segmentation_fn,
                            segmentation_kwargs=args,
                            images_background=None)

    print("Generating anchor images ... ")

    anchor_paths = []
    segment_paths = []
    anchor_infos = []

    for idx, image in enumerate(tqdm(prediction_frames)):
        explanation_img = explainer.explain(
            image, threshold=0.95, p_sample=0.5, tau=0.25, verbose="True")
        explanation_anchor = explanation_img.anchor
        explaintion_segments = explanation_img.segments
        explaintion_precision = explanation_img.precision
        explaintion_coverage = explanation_img.coverage

        print("Precision:", explaintion_precision,
              "Coverage:", explaintion_coverage)
        anchor_infos.append(
            {"precision": explaintion_precision, "coverage":  explaintion_coverage})
        all_zeros = not np.any(explanation_img)
        print(explanation_img)

        if not all_zeros:
            explanation_anchor = explanation_anchor.astype(np.float32)
            explanation_anchor /= 255

            anchor_file = str(idx) + '.png'
            img_path = os.path.join(ANCHORS_PATH, anchor_file)
            segment_path = os.path.join(SEGMENTS_PATH, anchor_file)
            plt.imsave(img_path, explanation_anchor)
            plt.imsave(segment_path, explaintion_segments)
            anchor_paths.append(img_path)
            segment_paths.append(segment_path)
        else:
            anchor_paths.append("None")
            segment_paths.append("None")

    return anchor_paths, segment_paths, anchor_infos


def getPredictionForVideo(videoName, videoUrl):
    # setGPUConfigurations()
    model = loadSavedModel()
    video = videoName

    ANCHORS_PATH = os.path.join(TEST_PATH, video, "anchor")
    SEGMENTS_PATH = os.path.join(TEST_PATH, video, "segments")
    TEMP_FRAMES_PATH = os.path.join(TEST_PATH, video, "frames")
    FACE_FRAMES_PATH = os.path.join(TEST_PATH, video, "faces")
    os.makedirs(ANCHORS_PATH, exist_ok=True)
    os.makedirs(TEMP_FRAMES_PATH, exist_ok=True)
    os.makedirs(FACE_FRAMES_PATH, exist_ok=True)
    os.makedirs(SEGMENTS_PATH, exist_ok=True)

    print("-----------------------------------")
    print(video)
    print("-----------------------------------")

    prediciton_arr, predicition_frames = predictVideo(
        videoUrl, model, FACE_FRAMES_PATH, TEMP_FRAMES_PATH)
    anchor_paths, segment_paths, anchor_infos = getAnchorExplainations(
        model, predicition_frames, ANCHORS_PATH, SEGMENTS_PATH)

    return json.dumps(prediciton_arr), anchor_paths, segment_paths, anchor_infos
