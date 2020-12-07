# iCatcher
A CNN that classifies discrete eye gaze direction ("Left", "Right", "Away") from low-res in-the-wild infant videos (per-frame classification).
Based on "Automatic, Real-Time Coding of Looking-While-Listening Children Videos Using Neural Networks" presented in [ICIS 2020](https://infantstudies.org/congress-2020).


# Step 1: Clone this repository to get a copy of the code to run locally.

`git clone https://github.com/yoterel/iCatcher.git`

# Step 2: Navigate to the Baby-Eye-Tracker directory, then create a virtual environment.

## Using virtual env:

Create the virtual environment:

`python3 -m venv env` (Linux and Mac) 

`py -m venv env` (Windows)

See https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/

Make sure it is using Python 3.6+ (by default it will use the most recent version available, but you can specify). 
If you don't have an appropriate version of python installed, you can install it at https://www.python.org/downloads/

You will need to make sure you have 64-bit python installed in order to use tensorflow-gpu (see https://github.com/tensorflow/tensorflow/issues/8251)

Activate the environment:

`source venv/bin/activate`

Finally intall requirements using the requirements.txt file in this repository:

`pip install -r requirements.txt`

## Using conda

We recommend installing [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for this, but you can also [Install Anaconda](https://www.anaconda.com/products/individual/get-started) if needed, then create an environment using the environment.yml file in this repository:

`conda env create -n env -f environment.yml`

Activate the environment

`conda activate env`

# Step 3:

- Download the latest network model & weights file [here](https://www.cs.tau.ac.il/~yotamerel/baby_eye_tracker/model.h5).
This is a keras model h5 file which contains both the architecture and the weights.

- Download the face extraction model files (opencv dnn):

  [prototxt (contains architecture)](https://www.cs.tau.ac.il/~yotamerel/baby_eye_tracker/config.prototxt)

  [caffemodel (contains weights)](https://www.cs.tau.ac.il/~yotamerel/baby_eye_tracker/face_model.caffemodel)

Put files in the [models](models) directory.

# Step 4:

To run the example file with the webcam (id for default webcam is usually 0):

`python example.py --source_type webcam my_webcam_id`

To run the example file with a video file:

`python example.py --source_type file /path/to/my/video.mp4`

You can save a labeled video by adding:

`--output_video_path /path/to/output_video.mp4`

If you want tooutput annotations to a file, use:

`--output_annotation /path/to/output_file.csv`

By default, this will save a file in the format described [here](https://osf.io/3n97m/) describing the 
output of the automated coding. Other formats will be added in the future.

An example video file can be found [here](https://www.cs.tau.ac.il/~yotamerel/baby_eye_tracker/example.mp4).

Feel free to contribute code.
