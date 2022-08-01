from flask import make_response, jsonify
from werkzeug.exceptions import HTTPException, BadRequest, InternalServerError
import ast
import smtplib
from email.mime import base
import firebase_admin
from firebase_admin import credentials, firestore, storage, initialize_app
from pathlib import Path
from flask import Flask, jsonify, make_response, request, send_from_directory
import json
from test import getPredictionForVideo
import os
import shutil
import ast
import ffmpeg
import urllib.parse
import subprocess
from flask_swagger_ui import get_swaggerui_blueprint
app = Flask(__name__)


@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory("static", path)


SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'
swaggerur_blueprint = get_swaggerui_blueprint(SWAGGER_URL, API_URL, config={
                                              'app_name': "Xaivier Forensic Deepfake Detector API"})
app.register_blueprint(swaggerur_blueprint, url_prefix=SWAGGER_URL)

cred = credentials.Certificate('key.json')
firebase_admin.initialize_app(cred, {
    "serviceAccount": "key.json",
})

db = firestore.client()


def sendMail():
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login('xaiviertoolkit@gmail.com', 'intel@321')
    server.sendmail('xaiviertoolkit@gmail.com',
                    'kjayregis@gmail.com', "Video is done processing")
    print("Mail Sent")


def getMetadata(video_url):
    metadata = []
    dd = ffmpeg.probe(
        video_url, cmd=r'ffprobe.exe')

    for stream in dd["streams"]:
        for pair in stream:
            if pair == "r_frame_rate":
                metadata.append({"frame_rate": stream[pair]})
            elif pair == "duration":
                metadata.append({"duration": stream[pair]})
            elif pair == "duration_ts":
                metadata.append({"duration_ts": stream[pair]})
            elif pair == "bit_rate":
                metadata.append({"bitrate": stream[pair]})
            elif pair == "codec_long_name":
                metadata.append({"codec_name": stream[pair]})
            elif pair == "tags":
                metadata.append({"tags": stream[pair]})

    print(metadata)
    return metadata


# def custom_error(message, status_code):
#     return make_response(jsonify(message), status_code)

@ app.route("/", methods=['GET'])
def home():
    return "Welcome to XAIVIER!"

@ app.route("/prediction", methods=['GET'])
def prediction():
    try:
        video_name = request.args.get('videoname')
        video_name = os.path.splitext(video_name)[0]
        print("Getting stored prediction for", video_name)

        doc_ref = db.collection(u'predictions').document(video_name)
        doc = doc_ref.get()
        doc = doc.to_dict()
        prediciton_arr = [doc["fake"], doc["real"],
                          doc["label"], doc["label_text"], doc["frame_preds"]]
        metadata = doc["metadata"]
        return {"metadata": metadata, "fake": prediciton_arr[0], "real": prediciton_arr[1], "Predicted Label": prediciton_arr[2], "Predicted Label Text": prediciton_arr[3], "Frame preds": prediciton_arr[4]}
    except:
        err = "Something went wrong on the server. Try again"
        return err,  500


@ app.route("/predict", methods=['POST'])
def predict():
    try:
        data = request.get_json()
        video_url = data['url']
        video_name = data['videoName']
        video_name = os.path.splitext(video_name)[0]
        print("Starting Prediction On Video", video_name)

        doc_ref = db.collection(u'predictions').document(video_name)

        prediciton_arr, anchor_paths, segment_paths, anchor_infos = getPredictionForVideo(
            video_name, video_url)
        prediciton_arr = json.loads(prediciton_arr)

        print("Uploading frames and faces to Cloud storage")

        frames = ast.literal_eval(prediciton_arr[4])
        i = 0
        bucket = storage.bucket()
        for frame in frames:
            frame["precision"] = anchor_infos[i]["precision"]
            frame["coverage"] = anchor_infos[i]["coverage"]
            remote_path = "frames/" + video_name + "_" + str(i)
            blob = bucket.blob(remote_path)
            outfile = frame["frame_path"]
            blob.upload_from_filename(
                filename=outfile, num_retries=3, timeout=120)
            blob.make_public()

            frame["frame_path"] = blob.public_url

            remote_path = "faces/" + video_name + "_" + str(i)
            blob = bucket.blob(remote_path)
            outfile = frame["face_path"]
            blob.upload_from_filename(
                filename=outfile, num_retries=3, timeout=120)
            blob.make_public()

            frame["face_path"] = blob.public_url

            if anchor_paths[i] != "None":
                remote_path = "anchors/" + video_name + "_" + str(i)
                blob = bucket.blob(remote_path)
                print(anchor_paths[i])
                outfile = anchor_paths[i]
                blob.upload_from_filename(
                    filename=outfile, num_retries=3, timeout=120)
                blob.make_public()
                frame["anchor_path"] = blob.public_url

                remote_path = "segments/" + video_name + "_" + str(i)
                blob = bucket.blob(remote_path)
                print(segment_paths[i])
                outfile = segment_paths[i]
                blob.upload_from_filename(
                    filename=outfile, num_retries=3, timeout=120)
                blob.make_public()
                frame["segment_path"] = blob.public_url

            i += 1

        prediciton_arr[4] = frames

        base_path = os.path.join("test", video_name)

        shutil.rmtree(base_path)

        print(prediciton_arr)

        metadata = getMetadata(data['url'])
        print("Got the metadata")
        # sendMail()
        print("Saving in firestore... ")
        doc_ref.set({
            u'fake': prediciton_arr[0],
            u'real': prediciton_arr[1],
            u'label': prediciton_arr[2],
            u'label_text': prediciton_arr[3],
            u'metadata': metadata,
            u'frame_preds': prediciton_arr[4]
        })
        print( {"metadata": metadata, "fake": prediciton_arr[0], "real": prediciton_arr[1], "Predicted Label": prediciton_arr[2], "Predicted Label Text": prediciton_arr[3], "Frame preds": prediciton_arr[4]})
        return {"metadata": metadata, "fake": prediciton_arr[0], "real": prediciton_arr[1], "Predicted Label": prediciton_arr[2], "Predicted Label Text": prediciton_arr[3], "Frame preds": prediciton_arr[4]}
    except:
        err = "Something went wrong on the server. Try again"
        return err,  500


if __name__ == "__main__":
    app.run(debug=True)
