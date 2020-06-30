# Baby-Eye-Tracker
A CNN that classifies discrete eye gaze directions

Step 1:
Install requirements (python >= 3.6):

`pip install -r requirements.txt`

Step 2:
Download the network model & weights file [here](https://www.cs.tau.ac.il/~yotamerel/eye_discrete_model_and_weights.h5)

Step 3:
Run the example file with the webcam:

`python example.py --webcam mywebcam_id`

Run the example file with a video file:

`python example.py --video_file /path/to/my/video.avi`
