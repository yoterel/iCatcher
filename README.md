# iCatcher
A CNN that classifies discrete eye gaze direction ("Left", "Right", "Away") from low-res in-the-wild infant videos (per-frame classification).
Based on "Automatic, Real-Time Coding of Looking-While-Listening Children Videos Using Neural Networks" presented in [ICIS 2020](https://infantstudies.org/congress-2020), and in [BUCLD 46](https://www.bu.edu/bucld/).

Preprint of icatcher available [here](https://doi.org/10.31234/osf.io/6ysu9).

Preprint of icatcher+, which this repository uses (partially) available [here](https://doi.org/10.31234/osf.io/s8c4m).

# Step 1: Clone this repository to get a copy of the code to run locally.

`git clone https://github.com/yoterel/iCatcher.git`

# Step 2: Navigate to the iCatcher directory, then create a virtual environment.

## Using conda

We recommend installing [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for this, but you can also [Install Anaconda](https://www.anaconda.com/products/individual/get-started) if needed, then create an environment using the environment.yml file in this repository:

`conda env create -n env -f environment.yml`

Activate the environment

`conda activate env`

# Step 3: Download the latest network model & weights file:

Download all required model & weight files [here](https://www.cs.tau.ac.il/~yotamerel/baby_eye_tracker/model_files.zip).

Extract the files from this step to the [models](models) directory (the models directory needs to directly contain the extracted files).

This zip contains:

- The original iCatcher model (tf model) trained on the Princeton look-while-listening dataset (model.h5).
  
- A proposed improvement of iCatcher (torch model) trained on the lookit dataset (icatcher+.pt).
  
  **Note1**: the improvments were proposed by [these guys](https://github.com/XinchengTan/infant-gaze-coding).
  
  **Note2**: this model does not include the face selection mechanism described by the original authors (yet).

- Face extraction model files from opencv-dnn (config.prototxt & face_model.caffemodel)


# Step 4:

To run icatcher with the webcam (id for default webcam is usually 0):

`python icatcher.py --source_type webcam my_webcam_id --show_output`

To run icatcher with a video file:

`python icatcher.py --source_type file /path/to/my/video.mp4 --show_output`

You can save a labeled video by adding:

`--output_video_path /path/to/output_folder`

If you want to output annotations to a file, use:

`--output_annotation /path/to/output_annotation_folder`

By default, this will save a file in the format described [here](https://osf.io/3n97m/) describing the 
output of the automated coding. Other formats will be added upon request.

An example video file can be found [here](https://www.cs.tau.ac.il/~yotamerel/baby_eye_tracker/example.mp4).

For all command line options, use:

`python icatcher.py --help`


# Training:

If you want to retrain the model from scratch / finetune it, use [train.py](train.py).

**Note**: this script expects a dataset orginized in a particular way. To create such dataset follow these steps:

- Gather raw video files into some folder
- Gather label files into some other folder (these can be in any format you choose, but a parser is required - see below)
- Use "create_dataset_from_videos" in [dataset.py](dataset.py) script to automatically extract faces from each frame into a output folder (with subfolders away, left and right). Notice this requires creating your own parser - see [parsers.py](parsers.py) for examples.
- Use "create_custom_dataset" in [dataset.py](dataset.py) script to further process the dataset into the final form (we recommend using default values unless architectural changes are made to the network). The final dataset structure will be a folder containing the subfolders {train, validation, holdout} each with their own subfolders {away, left, right}, consisting of 5-tuples of non-consecutive frames from the original videos in the appropriate class.
- Finally, use [train.py](train.py) to train the network.

For more detailed information, see function documentation in code.

# Contributions

Feel free to contribute code by submitting a pull request.
