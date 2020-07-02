import argparse
import cv2
from pathlib import Path
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras
import keras


def put_text(img, class_name):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 500)
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2

    cv2.putText(img, class_name,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)
    return img


def detectFaceOpenCVDnn(net, frame, conf_threshold):
    # frameOpencvDnn = frame.copy()
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = max(int(detections[0, 0, i, 3] * frameWidth), 0)
            y1 = max(int(detections[0, 0, i, 4] * frameHeight), 0)
            x2 = max(int(detections[0, 0, i, 5] * frameWidth), 0)
            y2 = max(int(detections[0, 0, i, 6] * frameHeight), 0)
            bboxes.append([x1, y1, x2-x1, y2-y1])
    return bboxes


def predict(vid_path):
    face_model_file = Path("face_model.caffemodel")
    config_file = Path("config.prototxt")
    path_to_primary_model = Path("model.h5")
    classes = {'away': 0, 'left': 1, 'right': 2}
    reverse_dict = {0: 'away', 1: 'left', 2: 'right'}
    sequence_length = 9
    loc = -5
    face_model = cv2.dnn.readNetFromCaffe(str(config_file), str(face_model_file))
    primary_eye_model = keras.models.load_model(str(path_to_primary_model))
    print("predicting on:", vid_path)
    answers = []
    image_sequence = []
    frames = []
    cap = cv2.VideoCapture(str(vid_path))
    ret_val, frame = cap.read()
    while ret_val:
        frames.append(frame)
        bbox = detectFaceOpenCVDnn(face_model, frame, 0.7)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # DO THIS or else degraded performance.
        if not bbox:
            answers.append(classes['away'])  # if face detector fails, treat as away
            image = np.zeros((1, 75, 75, 3), np.float64)
            image_sequence.append((image, True))
        else:
            face = min(bbox, key=lambda x: x[3] - x[1])  # select lowest face, probably belongs to kid
            crop_img = frame[face[1]:face[1] + face[3], face[0]:face[0] + face[2]]
            if crop_img.size == 0:
                answers.append(classes['away'])  # if face detector fails, treat as away
                image = np.zeros((1, 75, 75, 3), np.float64)
                image_sequence.append((image, True))
            else:
                answers.append(classes['left'])  # if face detector succeeds, treat as left
                image = cv2.resize(crop_img, (75, 75)) * 1. / 255
                image = np.expand_dims(image, axis=0)
                image_sequence.append((image, False))
        if len(image_sequence) == sequence_length:
            if not image_sequence[sequence_length // 2][1]:  # if middle image is not blacked
                to_predict = [x[0] for x in image_sequence[0::2]]
                prediction = primary_eye_model.predict(to_predict)
                predicted_classes = np.argmax(prediction, axis=1)
                answers[loc] = np.int16(predicted_classes[0]).item()
            image_sequence.pop(0)
            popped_frame = frames.pop(0)
            class_text = reverse_dict[answers[-sequence_length]]
            popped_frame = put_text(popped_frame, class_text)
            cv2.imshow('image', popped_frame)
            print(class_text)
            cv2.waitKey(0)
        ret_val, frame = cap.read()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='-1', help='gpu id to use, use -1 for CPU')
    parser.add_argument('--video-file', help='path to video file to process')
    parser.add_argument('--webcam-id', type=int, default=0, help='webcam id to use as input source')
    parser.add_argument('--output-path', help='if present, results will be dumped to this file')
    opt = parser.parse_args()
    video_file = Path('example.mp4')
    predict(video_file)
