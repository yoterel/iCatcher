import argparse
import cv2
import keras


def predict():
    print("predicting on:", vid_path)
    answers = []
    image_sequence = []
    frame_counter = 0
    cap = cv2.VideoCapture(str(vid_path))
    ret_val, frame = cap.read()
    while ret_val:
        # print(frame_counter)
        # if any(start <= frame_counter <= end for start,end in self.trial_times):
        bbox = detectFaceOpenCVDnn(self.face_model, frame, 0.7)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # DO THIS or else degraded performance.
        if not bbox:
            answers.append(self.classes['away'])  # if face detector fails, treat as away
            image = np.zeros((1, 75, 75, 3), np.float64)
            image_sequence.append((image, True))
        # image_sequence.clear()
        else:
            face = min(bbox, key=lambda x: x[3] - x[1])  # select lowest face, probably belongs to kid
            crop_img = frame[face[1]:face[1] + face[3], face[0]:face[0] + face[2]]
            if crop_img.size == 0:
                answers.append(self.classes['away'])  # if face detector fails, treat as away
                image = np.zeros((1, 75, 75, 3), np.float64)
                image_sequence.append((image, True))
            else:
                answers.append(self.classes['left'])  # if face detector succeeds, treat as left
                image = cv2.resize(crop_img, (75, 75)) * 1. / 255
                image = np.expand_dims(image, axis=0)
                image_sequence.append((image, False))
        if len(image_sequence) == self.sequence_length:
            if not image_sequence[self.sequence_length // 2][1]:  # if middle image is not blacked
                if self.is_single_image_model:
                    to_predict = image_sequence[0][0]
                else:
                    if self.skip_frames:
                        to_predict = [x[0] for x in image_sequence[0::2]]
                    else:
                        to_predict = [x[0] for x in image_sequence]
                prediction = self.primary_eye_model.predict(to_predict)
                predicted_classes = np.argmax(prediction, axis=1)
                if predicted_classes[0] == 1 and self.is_2class_model:
                    prediction = self.secondary_eye_model.predict(to_predict)
                    predicted_classes = np.argmax(prediction, axis=1)
                    predicted_classes[0] += 1
                if self.is_single_image_model:
                    loc = -1
                else:
                    loc = -5 if self.skip_frames else -3
                answers[loc] = np.int16(predicted_classes[0]).item()
            image_sequence.pop(0)
        ret_val, frame = cap.read()


if __name__ == "main":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='-1', help='gpu id to use, use -1 for CPU')
    parser.add_argument('--video-file', help='path to video file to process')
    parser.add_argument('--webcam-id', type=int, default=0, help='webcam id to use as input source')
    parser.add_argument('--output-path', help='if present, results will be dumped to this file')
    opt = parser.parse_args()
