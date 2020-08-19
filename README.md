# Baby-Eye-Tracker
A CNN that classifies discrete eye gaze direction ("Left", "Right", "Away") from low-res in-the-wild infant videos.
Based on "Automatic, Real-Time Coding of Looking-While-Listening Children Videos Using Neural Networks" presented in [ICIS 2020](https://infantstudies.org/congress-2020).

# Step 1:
Install requirements (python >= 3.6):

`pip install -r requirements.txt`

# Step 2:
Download the latest network model & weights file [here](https://www.cs.tau.ac.il/~yotamerel/baby_eye_tracker/model.h5).
This is a keras model h5 file which contains both the architecture and the weights.

Download the face extraction model files (opencv dnn):

[prototxt (contains architecture)](https://www.cs.tau.ac.il/~yotamerel/baby_eye_tracker/config.prototxt)

[caffemodel (contains weights)](https://www.cs.tau.ac.il/~yotamerel/baby_eye_tracker/face_model.caffemodel)

Put files in the same directory as "example.py".

# Step 3:
To run the example file with the webcam (id for default webcam is usually 0):

`python example.py --source_type webcam my_webcam_id`

To run the example file with a video file:

`python example.py --source_type file /path/to/my/video.mp4`

An example video file can be found [here](https://www.cs.tau.ac.il/~yotamerel/baby_eye_tracker/example.mp4).

Feel free to contribute code.
