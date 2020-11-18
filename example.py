import argparse
import cv2
from pathlib import Path
import numpy as np


def put_text(img, class_name):
    font = cv2.FONT_HERSHEY_SIMPLEX
    top_left_corner_text = (10, 30)
    font_scale = 1
    font_color = (255, 255, 255)
    line_type = 2

    cv2.putText(img, class_name,
                top_left_corner_text,
                font,
                font_scale,
                font_color,
                line_type)
    return img


def put_arrow(img, class_name, face):
    arrow_start_x = int(face[0] + 0.5 * face[2])
    arrow_end_x = int(face[0] + 0.1 * face[2] if class_name == "left" else face[0] + 0.9 * face[2])
    arrow_y = int(face[1] + 0.8 * face[3])
    img = cv2.arrowedLine(img,
                          (arrow_start_x, arrow_y),
                          (arrow_end_x, arrow_y),
                          (0, 255, 0),
                          thickness=3,
                          tipLength=0.4)
    return img


def put_rectangle(popped_frame, face):
    color = (0, 255, 0)  # green
    thickness = 2
    popped_frame = cv2.rectangle(popped_frame,
                                 (face[0], face[1]), (face[0] + face[2], face[1] + face[3]),
                                 color,
                                 thickness)
    return popped_frame


def detect_face_opencv_dnn(net, frame, conf_threshold):
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


def predict(opt):
    # initialize
    face_model_file = Path("face_model.caffemodel")
    config_file = Path("config.prototxt")
    path_to_primary_model = Path("model.h5")
    classes = {'away': 0, 'left': 1, 'right': 2}
    reverse_dict = {0: 'away', 1: 'left', 2: 'right'}
    sequence_length = 9
    loc = -5
    answers = []
    image_sequence = []
    frames = []
    frame_count = 0
    last_class_text = ""  # Initialize so that we see the first class assignment as an event to record
    dataset_mean = opt.per_channel_mean if opt.per_channel_mean else [0.41304266, 0.34594961, 0.27693587]
    print("using the following values for per-channel mean:", dataset_mean)
    dataset_std = opt.per_channel_std if opt.per_channel_std else [0.28606387, 0.2466201, 0.20393684]
    print("using the following values for per-channel std:", dataset_mean)
    # load deep models
    face_model = cv2.dnn.readNetFromCaffe(str(config_file), str(face_model_file))
    primary_model = tf.keras.models.load_model(str(path_to_primary_model))
    # set video source
    if opt.source_type == 'file':
        video_file = Path(opt.source)
        print("predicting on file:", video_file)
        cap = cv2.VideoCapture(str(video_file))
    else:
        print("predicting on webcam:", opt.source)
        cap = cv2.VideoCapture(int(opt.source))
    # Get some basic info about the video
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    resolution = (int(width), int(height))
    framerate = int(cap.get(cv2.CAP_PROP_FPS))
    # If creating annotated video output, set up now
    if opt.output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*"MP4V")  # may need to be adjusted per available codecs & OS
        video_output = cv2.VideoWriter(str(opt.output_video_path.resolve()), fourcc, framerate, resolution, True)
    if opt.output_annotation:
        output_file = open(opt.output_annotation, "w", newline="")
        if opt.output_format == "PrefLookTimestamp":
            # Write header
            output_file.write("Tracks: left, right, away, codingactive, outofframe\nTime,Duration,TrackName,comment\n\n")

    # iterate over frames
    ret_val, frame = cap.read()
    while ret_val:
        frames.append(frame)
        bbox = detect_face_opencv_dnn(face_model, frame, 0.7)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # network was trained on RGB images.
        if not bbox:
            answers.append(classes['away'])  # if face detector fails, treat as away and mark invalid
            image = np.zeros((1, 75, 75, 3), np.float64)
            image_sequence.append((image, True))
        else:
            face = min(bbox, key=lambda x: x[3] - x[1])  # select lowest face in image, probably belongs to kid
            crop_img = frame[face[1]:face[1] + face[3], face[0]:face[0] + face[2]]
            if crop_img.size == 0:
                answers.append(classes['away'])  # if face detector fails, treat as away and mark invalid
                image = np.zeros((1, 75, 75, 3), np.float64)
                image_sequence.append((image, True))
            else:
                answers.append(classes['left'])  # if face detector succeeds, treat as left and mark valid
                image = cv2.resize(crop_img, (75, 75)) * 1. / 255
                image = np.expand_dims(image, axis=0)
                image -= np.array(dataset_mean)
                image /= (np.array(dataset_std) + 1e-6)
                image_sequence.append((image, False))
        if len(image_sequence) == sequence_length:
            if not image_sequence[sequence_length // 2][1]:  # if middle image is valid
                to_predict = [x[0] for x in image_sequence[0::2]]
                prediction = primary_model.predict(to_predict)
                predicted_classes = np.argmax(prediction, axis=1)
                answers[loc] = np.int16(predicted_classes[0]).item()
            image_sequence.pop(0)
            popped_frame = frames.pop(0)
            class_text = reverse_dict[answers[-sequence_length]]
            # If save_annotated_video is true, add text label, bounding box for face, and arrow showing direction
            if opt.output_video_path:
                popped_frame = put_text(popped_frame, class_text)
                if bbox:
                    popped_frame = put_rectangle(popped_frame, face)
                    if not class_text == "away":
                        popped_frame = put_arrow(popped_frame, class_text, face)
                video_output.write(popped_frame)
                # Record "event" for change of direction if code has changed
            if opt.output_annotation:
                if opt.output_format == "PrefLookTimestamp":
                    if class_text != last_class_text:
                        frame_ms = int(1000./framerate * frame_count)
                        output_file.write("{},0,{}\n".format(frame_ms, class_text))
                        last_class_text = class_text
            print("frame: {}, class: {}".format(str(frame_count-sequence_length+1), class_text))
        ret_val, frame = cap.read()
        frame_count += 1
        
    if opt.output_video_path:
        video_output.release()
    if opt.output_annotation:  # write footer to file
        if opt.output_format == "PrefLookTimestamp":
            frame_ms = int(1000./framerate * frame_count)
            output_file.write("0,{},codingactive\n".format(frame_ms))
            output_file.close()
    cap.release()


def configure_environment(gpu_id):
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id  # set gpu visibility prior to importing tf and keras
    global tf
    import tensorflow as tf


def parse_arguments():
    parser = argparse.ArgumentParser(description='Baby Eye Tracker')
    parser.add_argument('--gpu_id', type=str, default='-1', help='GPU id to use, use -1 for CPU.')
    parser.add_argument('--source_type', type=str, default='file', choices=['file', 'webcam'],
                        help='selects source of stream to use.')
    parser.add_argument('source', type=str, help='the source to use (path to video file or webcam id).')
    parser.add_argument('--output_annotation', type=str, help='filename for text output')
    # Set up text output file, using https://osf.io/3n97m/ - PrefLookTimestamp coding standard
    parser.add_argument('--output_format', type=str, default="PrefLookTimestamp", choices=["PrefLookTimestamp"])
    parser.add_argument('--output_video_path', help='if present, annotated video will be saved to this path')
    parser.add_argument('--per_channel_mean', nargs=3, metavar=('Channel1_mean', 'Channel2_mean', 'Channel3_mean'),
                        type=float, help='supply custom per-channel mean of data for normalization')
    parser.add_argument('--per_channel_std', nargs=3, metavar=('Channel1_std', 'Channel2_std', 'Channel3_std'),
                        type=float, help='supply custom per-channel std of data for normalization')
    args = parser.parse_args()

    if args.output_annotation:
        args.output_filepath = Path(args.output_annotation)
    if args.output_video_path:
        args.output_video_path = Path(args.output_video_path)
    return args


if __name__ == '__main__':
    args = parse_arguments()
    configure_environment(args.gpu_id)
    predict(args)
