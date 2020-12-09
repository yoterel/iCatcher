from pathlib import Path
import numpy as np
import re
import random
import shutil
import cv2
import predict
import parsers
import logging


def get_frame_number(frame):
    number = re.findall(r'\d+', frame)[-1]
    return int(number)


def is_legal_spacing(series, consecutive_frames, is_single_frame):
    if is_single_frame:
        return True
    else:
        if consecutive_frames:
            length = len(series) - 1
            if length == 0:
                return True
            else:
                return get_frame_number(series[-1][0]) - get_frame_number(series[0][0]) == length
        else:
            for i, x in enumerate(series):
                if i > 0:
                    if get_frame_number(series[i][0]) != get_frame_number(series[i - 1][0]) + 2:
                        return False
            return True


def select_folder(holdout_folder,
                  validation_folder,
                  train_folder,
                  file_counter,
                  ten_percent,
                  holdout_vid,
                  disjoint_validation_videos=False,
                  training_size=0.9):
    if holdout_vid:  # holdout
        selected_folder = holdout_folder
    else:
        if disjoint_validation_videos:
            if file_counter < ten_percent:  # holdout
                selected_folder = validation_folder
            else:
                selected_folder = train_folder
        else:
            if random.uniform(0, 1) <= training_size:  # training
                selected_folder = train_folder
            else:
                selected_folder = validation_folder  # validation
    return selected_folder


def is_series_diverse(series):
    classes = [x[1] for x in series]
    unique = set(classes)
    if len(unique) > 1:
        return True
    else:
        return False


def check_terminate(sample, vid_sample, total_sample_number, sample_thresh_per_video):
    if sample >= total_sample_number:
        logging.info("sample>=total_sample_number")
        return True
    if sample_thresh_per_video:
        if vid_sample >= sample_thresh_per_video:
            logging.info("reached vid threshold.")
            return True
    return False


def get_class(series, frames_per_data_point):
    my_dst_class = series[frames_per_data_point // 2][1]
    return my_dst_class


def get_vid_name(file_name):
    hyphen_number = file_name.count("_")
    if hyphen_number == 1:
        return re.split(r'_', file_name)[0]
    else:
        if hyphen_number == 2:
            my_list = re.split(r'_', file_name)
            return my_list[0]+"_"+my_list[1]
        else:
            return None


def copy_series(series, db_root_folder, sample_number, dst_folder):
    vid_name = get_vid_name(series[0][0])
    for j, img in enumerate(series):
        src = Path.joinpath(db_root_folder, img[1], img[0])
        new_name = '{}_{:07d}_xx{:02d}xx.png'.format(vid_name, sample_number, j)
        dst = Path.joinpath(dst_folder, new_name)  # select middle member as the class indicator
        shutil.copyfile(str(src), str(dst))


def create_folder(parent_folder, name):
    folder = Path(parent_folder, name)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def create_train_val_test_folders(root_folder, create_sub_class_folders=True):
    train_folder = create_folder(root_folder, "train")
    validation_folder = create_folder(root_folder, "validation")
    holdout_folder = create_folder(root_folder, "holdout")
    if create_sub_class_folders:
        create_folder(train_folder, "away")
        create_folder(train_folder, "left")
        create_folder(train_folder, "right")
        create_folder(validation_folder, "away")
        create_folder(validation_folder, "left")
        create_folder(validation_folder, "right")
        create_folder(holdout_folder, "away")
        create_folder(holdout_folder, "left")
        create_folder(holdout_folder, "right")
    return train_folder, validation_folder, holdout_folder


def get_vid_number(vid_name):
    numbers = re.findall(r'\d+', vid_name)
    return int(numbers[0])


def create_custom_dataset(db_folder,
                          output_folder,
                          consecutive_frames=False,
                          is_single_frame=False,
                          diversify=False,
                          frames_per_data_point=5,
                          total_sample_number=1000000,
                          disjoint_validation_videos=True,
                          holdout_is_experiment=True,
                          holdout_exp_name="Canine2",
                          sample_thresh_per_video=None,
                          filter_file=None):
    """
    creates a custom dataset from a full dataset
    :param db_folder: the full dataset folder (structure of folder must be /db_folder/class/vidname_framenumber.png)
    :param output_folder: the output folder
    :param consecutive_frames: whether the frames in the new dataset will be consecutive (if not, skips every 1 frame)
    :param is_single_frame: whether the dataset is a single frame per data-point or multi frame per data-point
    :param diversify: use only "interesting" points for dataset (i.e. classes are diverse within a data-point)
    :param frames_per_data_point: how many frames per data point. must be odd number.
    :param total_sample_number: total number of data-points in the new dataset
    :param disjoint_validation_videos: whether validation set is taken from a disjoint set of videos or not
    :param holdout_is_experiment: whether holdout set is an entire experiment (i.e. a set of videos with same prefix)
    :param holdout_exp_name: the name of the prefix of videos to use as holdout (test) set
    :param sample_thresh_per_video: if set, maximum amount of frames to take per video
    :param filter_file: if present, uses only images from video names mentioned in this list
    :return: -
    """
    db_folder = Path(db_folder)
    output_folder = Path(output_folder)
    exclude_videos = []
    train_folder, validation_folder, holdout_folder = create_train_val_test_folders(output_folder)
    output_stats_file = Path.joinpath(output_folder, "db_stats.txt")
    f = open(str(output_stats_file), "w+")
    sample = 0
    file_counter = 0
    no_holdout_file_counter = 0
    database_file = Path.joinpath(db_folder, "db_stats.txt")
    files = [line.rstrip('\n') for line in open(database_file, "r")]
    if filter_file:
        filters = [line.rstrip('\n') for line in open(filter_file, "r")]
        file_names = [x.split(",")[0] for x in files]
        files = [files[i] for i, x in enumerate(file_names) if x in filters]
    if exclude_videos:
        files = [x for x in files if x.split(",")[0] not in exclude_videos]
    files_holdout = [x for x in files if holdout_exp_name in x]
    files_not_holdout = [x for x in files if holdout_exp_name not in x]
    perm = np.random.permutation(len(files))
    ten_percent = len(files_not_holdout) // 10
    while sample < total_sample_number and file_counter < len(perm):
        old_sample = sample
        vid_sample = 0
        file_data = files[perm[file_counter]]
        if file_data in files_holdout:
            holdout_vid = True
        else:
            holdout_vid = False
        fields = file_data.split(",")
        logging.info("file: {}".format(fields[0]))
        logging.info("file counter: {}".format(file_counter))
        logging.info("no_holdout_counter: {}".format(no_holdout_file_counter))
        logging.info("sample: {}".format(sample))
        query = '**/{}*.png'.format(fields[0])
        images = []
        for image in db_folder.glob(query):
            images.append((image.name, image.parent.name))
        images.sort(key=lambda tup: tup[0])  # sorts in place
        if is_single_frame:
            my_image_range = images
        else:
            if consecutive_frames:  # using consecutive images every time
                my_image_range = images[0:-frames_per_data_point+1]
            else:
                my_image_range = images[0:-(frames_per_data_point*2 - 1) + 1]  # using jumps of 2 between images every time
        for i, image in enumerate(my_image_range):  # run through required images
            skip_frames = 1 if (consecutive_frames or is_single_frame) else 2
            scope = frames_per_data_point if (consecutive_frames or is_single_frame) else frames_per_data_point*2 - 1
            series = images[i:i + scope:skip_frames]  # image series
            if is_legal_spacing(series, consecutive_frames, is_single_frame):  # images are "legal" according to spacing
                if is_series_diverse(series) or not diversify:  # select only series with 2 or more classes
                    selected_folder = select_folder(holdout_folder,
                                                    validation_folder,
                                                    train_folder,
                                                    no_holdout_file_counter,
                                                    ten_percent,
                                                    holdout_vid,
                                                    disjoint_validation_videos)
                    my_dst_class = get_class(series, frames_per_data_point)
                    dst_folder = Path.joinpath(output_folder, selected_folder, my_dst_class)
                    copy_series(series, db_folder, sample, dst_folder)
                    vid_sample += 1
                    sample += 1
                    if check_terminate(sample, vid_sample, total_sample_number, sample_thresh_per_video):
                        break
        usable_frames = int(fields[1]) - int(fields[2]) - int(fields[3])
        logging.info("used: {} out of {} in: {}".format(sample - old_sample,usable_frames, fields[0]))
        f.write(file_data + "," + str(sample - old_sample) + "\n")
        f.flush()
        if not holdout_vid:
            no_holdout_file_counter += 1
        file_counter += 1
    f.close()


def create_dataset_from_videos(raw_video_folder,
                               output_folder,
                               raw_labels=None,
                               face_model_file=Path("models", "face_model.caffemodel"),
                               config_file=Path("models", "config.prototxt"),
                               ):
    """
    Given a folder of video files and a label parser, outputs a folder with 3 sub folders: away, left, right
    places in them all frames belonging to that class from the videos.
    NOTE: this is not the final input to iCatcher! use "create_custom_dataset" with default values for the final input.

    frames names are important, we use "videoname_framenumber.png"
    this is used to get 5-tuples of consecutive frames in "create_custom_dataset"

    :param raw_video_folder: the path to the videos folder
    :param output_folder: the path to output the dataset
    :param raw_labels: the path to the labels folder or labels in any format (np, pandas, etc)
    :param face_model_file: the face extractor model file
    :param config_file: the face extractor model config file
    :return: -
    """
    raw_video_folder = Path(raw_video_folder)
    output_folder = Path(output_folder)
    database_file = Path(output_folder, "db_stats.txt")
    right_folder = Path.joinpath(output_folder, "right")
    left_folder = Path.joinpath(output_folder, "left")
    away_folder = Path.joinpath(output_folder, "away")
    net = cv2.dnn.readNetFromCaffe(str(config_file), str(face_model_file))
    right_folder.mkdir(parents=True, exist_ok=True)
    left_folder.mkdir(parents=True, exist_ok=True)
    away_folder.mkdir(parents=True, exist_ok=True)
    try:  # open db_stats file if exists
        file = open(database_file, 'r')
    except IOError:
        file = open(database_file, 'w')
    file.close()
    vid_ext = ["*.mov", "*.mp4"]
    video_files = []
    for ext in vid_ext:
        video_files.extend(raw_video_folder.glob(ext))
    for video_file in video_files:
        f = open(database_file, "r")
        if video_file.stem in f.read():
            logging.info("Skipping: {} since it has already been processed".format(video_file.stem))
            f.close()
            continue
        logging.info("proccessing: {}".format(str(video_file)))
        parser = parsers.XmlParser(".vcx", raw_labels)
        responses = parser.parse(video_file.stem)
        frame_counter = 0
        no_face_counter = 0
        no_annotation_counter = 0
        cap = cv2.VideoCapture(str(video_file))
        ret_val, frame = cap.read()
        while ret_val:
            if responses:
                if frame_counter >= responses[0][0]:  # skip until reaching first annotated frame
                    # find closest (previous) response this frame belongs to
                    q = [index for index, val in enumerate(responses) if frame_counter >= val[0]]
                    response_index = max(q)
                    if responses[response_index][1] != 0:  # make sure response is valid
                        bbox = predict.detect_face_opencv_dnn(net, frame, 0.7)
                        if not bbox:
                            no_face_counter += 1
                            logging.info("Face not detected in frame: " + str(frame_counter))
                        else:
                            face = min(bbox, key=lambda x: x[3]-x[1])  # select lowest face, probably belongs to kid
                            crop_img = frame[face[1]:face[1] + face[3], face[0]:face[0] + face[2]]
                            face_class = responses[response_index][2]
                            file_name = '{}_{:05d}.png'.format(video_file.stem, frame_counter)
                            full_path_to_save = Path.joinpath(output_folder, face_class, file_name)
                            cv2.imwrite(str(full_path_to_save), crop_img)
                    else:
                        no_annotation_counter += 1
                        logging.info("Skipping since frame is invalid")
                else:
                    no_annotation_counter += 1
                    logging.info("Skipping since no annotation (yet)")
            else:
                no_annotation_counter += 1
                logging.info("Skipping frame since parser reported no annotation")
            ret_val, frame = cap.read()
            frame_counter += 1
            logging.info("Processing frame: {}".format(frame_counter))
        f = open(database_file, "a+")
        my_string = '{},{:05d},{:05d},{:05d}\n'.format(str(video_file.stem), frame_counter, no_face_counter,
                                                       no_annotation_counter)
        f.write(my_string)
        f.close()
