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
    face_model_file = Path("face_model.caffemodel")
    config_file = Path("config.prototxt")
    path_to_primary_model = Path("model.h5")
    classes = {'away': 0, 'left': 1, 'right': 2}
    reverse_dict = {0: 'away', 1: 'left', 2: 'right'}
    sequence_length = 9
    loc = -5
    dataset_mean = opt.per_channel_mean if opt.per_channel_mean else [0.41304266, 0.34594961, 0.27693587]
    print("using the following values for per-channel mean:", dataset_mean)
    dataset_std = opt.per_channel_std if opt.per_channel_std else [0.28606387, 0.2466201, 0.20393684]
    print("using the following values for per-channel std:", dataset_mean)
    face_model = cv2.dnn.readNetFromCaffe(str(config_file), str(face_model_file))
    primary_model = keras.models.load_model(str(path_to_primary_model))
    answers = []
    image_sequence = []
    frames = []
    frame_count = 0
    if opt.source_type == 'file':
        video_file = Path(opt.source)
        print("predicting on file:", video_file)
        cap = cv2.VideoCapture(str(video_file))
    else:
        print("predicting on webcam:", opt.source)
        cap = cv2.VideoCapture(int(opt.source))
        
    # Get the first frame
    ret_val, frame = cap.read()
    last_class_text = "" # Initialize so that we see the first class assignment as an event to record
    
    # Get some basic info about the video
    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    resolution = (int(width), int(height))
    framerate = int(cap.get(cv2.CAP_PROP_FPS))
    
    # If creating annotated video output, set up now!
    if opt.save_annotated_video:
        if opt.output_video_path:
            video_output_filepath = Path(opt.output_video_path)
        else:
            # Default output filename: same as input with "_babyeyetracker" appended,
            # in same directory
            if opt.source_type == "file":
                video_output_filepath = Path(video_file.parent, video_file.stem + "_babyeyetracker.mp4")          
            else:
                video_output_filepath = Path(f"webcam_{opt.source}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"MP4V") # may need to be adjusted per available codecs & OS
            video_output = cv2.VideoWriter(str(video_output_filepath.resolve()), fourcc, framerate, resolution, True)       
    
    # Set up text output file, using https://osf.io/3n97m/ - PrefLookTimestamp coding standard
    if opt.output_path:
        output_filepath = Path(opt.output_path)
    else:
        output_filepath = Path(video_file.parent, video_file.stem + "_babyeyetracker.csv") if opt.source_type == "file" else Path(f"webcam_{opt.source}.csv")
    output_file = open(output_filepath, "w", newline="")
    # Write header
    output_file.write("Tracks: left, right, away, codingactive, outofframe\nTime,Duration,TrackName,comment\n\n")

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
            # If showing result, add text label, bounding box for face, and arrow showing
            # direction
            if opt.show_result or opt.save_annotated_video:
                popped_frame = put_text(popped_frame, class_text)
                if bbox:
                    color = (0, 255, 0) # green
                    thickness = 2
                    popped_frame = cv2.rectangle(popped_frame, (face[0], face[1]), (face[0] + face[2], face[1] + face[3]), color, thickness)
                    if not class_text == "away":
                        arrow_start_x = int(face[0] + 0.5 * face[2])
                        arrow_end_x = int(face[0] + 0.1 * face[2] if class_text == "left" else face[0] + 0.9 * face[2])
                        arrow_y = int(face[1] + 0.8 * face[3])
                        popped_frame = cv2.arrowedLine(popped_frame, (arrow_start_x, arrow_y), (arrow_end_x, arrow_y), (0, 255, 0), thickness=3, tipLength=0.4)
                cv2.imshow("frame", popped_frame)
                cv2.waitKey(1) # Make sure display is updated
                if opt.save_annotated_video:
                    video_output.write(popped_frame)
                # Record "event" for change of direction if code has changed
                if class_text != last_class_text:
                    frame_ms = int(1000./framerate * frame_count)
                    output_file.write(f"{frame_ms},0,{class_text}\n")
                    last_class_text = class_text
            print("frame: {}, class: {}".format(str(frame_count-sequence_length+1), class_text))
        ret_val, frame = cap.read()
        frame_count += 1
        
    if opt.save_annotated_video:
        video_output.release()
        
    frame_ms = int(1000./framerate * frame_count)
    output_file.write(f"0,{frame_ms},codingactive\n")
    output_file.close()
    
    cap.release()
    cv2.destroyAllWindows()


def configure_environment(gpu_id, use_tensorflow_2):
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id  # set gpu visibility prior to importing tf and keras
    from keras.backend.tensorflow_backend import set_session
    
    if use_tensorflow_2:
        import tensorflow.compat.v1 as tf
        tf.disable_v2_behavior() 
        from tensorflow import keras # for compatibility with tensorflow 2.x - see https://github.com/keras-team/keras/releases
    else:
        import keras
        import tensorflow as tf

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='-1', help='GPU id to use, use -1 for CPU.')
    parser.add_argument('--source_type', type=str, default='file', choices=['file', 'webcam'],
                        help='selects source of stream to use.')
    parser.add_argument('source', type=str, help='the source to use (path to video file or webcam id).')
    parser.add_argument('--output_path', help='filename for text output')
    parser.add_argument('--show_result', help='frames and class will be displayed on screen.', action ='store_true')
    parser.add_argument('--save_annotated_video', help='video with class annotations will be saved', action ='store_true')
    parser.add_argument('--output_video_path', help='filename for annotated video output')
    parser.add_argument('--per_channel_mean', nargs=3, metavar=('Channel1_mean', 'Channel2_mean', 'Channel3_mean'),
                        type=float, help='supply custom per-channel mean of data for normalization')
    parser.add_argument('--per_channel_std', nargs=3, metavar=('Channel1_std', 'Channel2_std', 'Channel3_std'),
                        type=float, help='supply custom per-channel std of data for normalization')
    parser.add_argument('--use_tensorflow_2', action='store_true', help='Use Tensorflow 2.x, e.g. via Anaconda')
    opt = parser.parse_args()
    configure_environment(opt.gpu_id, opt.use_tensorflow_2)
    predict(opt)
