# Baby-Eye-Tracker
A CNN that classifies discrete eye gaze direction ("Left", "Right", "Away") from low-res in-the-wild infant videos (per-frame classification).
Based on "Automatic, Real-Time Coding of Looking-While-Listening Children Videos Using Neural Networks" presented in [ICIS 2020](https://infantstudies.org/congress-2020).


# Step 1: Clone this repository to get a copy of the code to run locally.

`git clone https://github.com/yoterel/Baby-Eye-Tracker`

# Step 2: Navigate to the Baby-Eye-Tracker directory, then create a virtual environment.

## Windows and Linux

Create the virtual environment:

`python3 -m venv env` (Linux and Mac) 

`py -m venv env` (Windows)

See https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/

Make sure it is using Python 3.6+ (by default it will use the most recent version available, but you can specify). 
If you don't have an appropriate version of python installed, you can install it at https://www.python.org/downloads/

You will need to make sure you have 64-bit python installed in order to use tensorflow-gpu (see https://github.com/tensorflow/tensorflow/issues/8251)

Activate the environment:

`source venv/bin/activate`

## MacOS using Anaconda

In principle you should just be able to create the virtual environment with python3 as on Linux. But installing the requirements is more straightforward using Anaconda:

[Install Anaconda](https://www.anaconda.com/products/individual/get-started) if needed, then create a virtual environment using conda, including pip (see [this article]( https://datumorphism.com/til/programming/python/python-anaconda-install-requirements/):

`conda create -n env python=3.8 anaconda pip`

Activate the environment

`conda activate env`

# Step 3: Install the requirements

From the activated virtual environment, run

## Regular Python environment:

`pip install -r requirements.txt`

If you see an error like the following on MacOS:

```
ERROR: Could not find a version that satisfies the requirement pkg-resources==0.0.0 (from -r requirements.txt (line 14)) (from versions: none)
ERROR: No matching distribution found for pkg-resources==0.0.0 (from -r requirements.txt (line 14))
```

you can safely remove the line `pkg-resources==0.0.0` from requirements.txt and try again. pkg-resources is already included in setuptools. 

## Conda environment

`pip install -r requirements_conda.txt`

# Step 4:

- Download the latest network model & weights file [here](https://www.cs.tau.ac.il/~yotamerel/baby_eye_tracker/model.h5).
This is a keras model h5 file which contains both the architecture and the weights.

- Download the face extraction model files (opencv dnn):

  [prototxt (contains architecture)](https://www.cs.tau.ac.il/~yotamerel/baby_eye_tracker/config.prototxt)

  [caffemodel (contains weights)](https://www.cs.tau.ac.il/~yotamerel/baby_eye_tracker/face_model.caffemodel)

Put files in the same directory as "example.py".

# Step 5:

To run the example file with the webcam (id for default webcam is usually 0):

`python example.py --source_type webcam my_webcam_id`

To run the example file with a video file:

`python example.py --source_type file /path/to/my/video.mp4`

If you're using Tensorflow 2.x (e.g. due to using Anaconda above), add the flag `--use_tensorflow_2`.

To display an annotated video during processing, showing the face bounding box and label,
add `--show_result`. You can also save this video using `--save_annotated_video`.

This will save a file in the format described [here](https://osf.io/3n97m/) describing the 
output of the automated coding. You can specify the location of this file using `--output_path <output path>`
and the location of the annotated video (if saving) using `--output_video_path <video path>`.

An example video file can be found [here](https://www.cs.tau.ac.il/~yotamerel/baby_eye_tracker/example.mp4).

Feel free to contribute code.
