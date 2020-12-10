import utils
import models
from pathlib import Path
import numpy as np
import data_generators
import sys
import pickle
import logging
import argparse
import tensorflow as tf


def train(args):
    """
    trains a network using cmd line arguments args
    :param args: cmd line arguments
    :return: -
    """
    pretrained_model = args.pretrained_model_path
    image_size = args.image_size
    batch_size = args.batch_size
    learning_rate = args.lr
    epochs = args.epochs
    seed = args.seed
    train_data_dir = Path(args.data_path, "train")
    validation_data_dir = Path(args.data_path, "validation")
    dataset_mean_and_std_file_path = Path("cache", "mean_and_std.pickle")
    early_stopping_patience = 8
    reduce_lr_patience = 3
    mask_training_images = True  # better generalization & robust to image missing
    mask_validation_images = False  # for some reason affects performance greatly if set to False
    normalize_data = True
    rescale, zoom_range, width_shift, height_shift, rotation, brightness_range = (1. / 255, 0, 0, 0, 0, [0.5, 1.5])
    verbosity = 1  # set verbosity of fit_generator
    class_weights = {0: 1, 1: 1.5, 2: 1.5}  # change for class imbalances (0: away, 1: left, 2: right)

    train_datagen = data_generators.ImageDataGenerator(
        rescale=rescale,
        # zoom_range=0.2,
        # width_shift_range=0.2,
        # height_shift_range=0.2,
        featurewise_center=normalize_data,
        featurewise_std_normalization=normalize_data,
        brightness_range=brightness_range,
        mask_image=mask_training_images)

    # Use custom validation image generator. rescale all images by 1./255 and apply image augmentation.
    valid_datagen = data_generators.ImageDataGenerator(rescale=rescale,
                                                       mask_image=mask_validation_images,
                                                       featurewise_center=normalize_data,
                                                       featurewise_std_normalization=normalize_data)

    # initialize envelope of 5 generators
    train_five_datagen = data_generators.FiveImageGenerator(
        generator=train_datagen,
        source_dir=train_data_dir,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        reverse_aug=False,
        seed=seed)
    # initialize envelope of 5 generators
    valid_five_datagen = data_generators.FiveImageGenerator(
        generator=valid_datagen,
        source_dir=validation_data_dir,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        reverse_aug=False,
        seed=seed)

    train_generator = train_five_datagen.generate()  # the iterator itself
    valid_generator = valid_five_datagen.generate()  # the iterator itself

    # find mean and std of a big chunk of the dataset if not already cached
    if not dataset_mean_and_std_file_path.is_file():
        n_batches = 3000
        temp_data = []
        for k in range(n_batches):
            logging.info("normalizing: {} / {}".format(str(k), str(n_batches)))
            x, y = next(train_generator)
            x = np.reshape(np.array(x), (len(x) * x[0].shape[0], 75, 75, 3))
            temp_data.append(x)
        temp_data = np.concatenate(temp_data)
        train_datagen.fit(temp_data)
        my_mean = train_datagen.mean
        my_std = train_datagen.std
        f = open(dataset_mean_and_std_file_path, 'wb')
        pickle.dump([my_mean, my_std], f)
        f.close()
    else:
        f = open(dataset_mean_and_std_file_path, 'rb')
        my_mean, my_std = pickle.load(f)
        f.close()
    logging.info("normalizing with mean: {}".format(my_mean))
    logging.info("normalizing with std: {}".format(my_std))
    train_datagen.mean = valid_datagen.mean = my_mean
    train_datagen.std = valid_datagen.std = my_std
    if pretrained_model:
        model = models.fine_tune_baseline_model(pretrained_model, learning_rate)
    else:
        model = models.build_baseline_model_5image(learning_rate)

    model_output = Path.joinpath(args.output_path, "{}.h5".format(args.model_name))
    checkpoint = tf.keras.callbacks.ModelCheckpoint(str(model_output),
                                                    monitor='val_loss',
                                                    verbose=1,
                                                    save_best_only=True,
                                                    mode='min',
                                                    period=1)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  min_delta=0.001,
                                                  patience=early_stopping_patience,
                                                  mode='min',
                                                  verbose=1)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                     factor=0.2,
                                                     patience=reduce_lr_patience,
                                                     verbose=1,
                                                     mode='min',
                                                     min_delta=0.001,
                                                     cooldown=0,
                                                     min_lr=1e-7)
    callbacks = [early_stop, checkpoint, reduce_lr]
    if args.tensorboard:
        from tensorflow.python.keras.callbacks import TensorBoard
        tensor_board = TensorBoard(log_dir=args.tensorboard,
                                   write_graph=True,
                                   write_images=True)
        callbacks.append(tensor_board)

    steps_per_epoch = np.math.ceil(train_five_datagen.samples / batch_size)
    validation_steps = np.math.ceil(valid_five_datagen.samples / batch_size)

# start training
    H = model.fit(train_generator,
                  shuffle=True,
                  steps_per_epoch=steps_per_epoch,
                  epochs=epochs,
                  validation_data=valid_generator,
                  validation_steps=validation_steps,
                  class_weight=class_weights,
                  verbose=verbosity,
                  callbacks=callbacks)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Script to train iCatcher')
    parser.add_argument("model_name", help="The name to give the newly trained model (without extension).")
    parser.add_argument("data_path", help="The path to the folder containing the data")
    parser.add_argument("--output_path", default="models", help="The trained model will be saved to this folder")
    parser.add_argument("--pretrained_model_path", help="A path to the pretrained model file")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size to train with")
    parser.add_argument("--image_size", type=int, default=75, help="All images will be resized to this size")
    parser.add_argument("--lr", type=int, default=1e-5, help="Initial learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Max number of epochs to train model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed to train with")
    parser.add_argument("--gpu_id", type=int, default=-1, help="Which GPU to use (or -1 for cpu)")
    parser.add_argument("--tensorboard",
                        help="If present, writes training stats to this path (readable with tensorboard)")
    parser.add_argument("--log",
                        help="If present, writes training log to this path")
    parser.add_argument("-v", "--verbosity", type=str, choices=["debug", "info", "warning"], default="info", help="Selects verbosity level")
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    if args.pretrained_model_path:
        args.pretrained_model_path = Path(args.pretrained_model_path)
    else:
        args.pretrained_model_path = None
    args.output_path = Path(args.output_path)
    if args.log:
        args.log = Path(args.log)
    if args.tensorboard:
        args.tensorboard = Path(args.tensorboard)
    args.data_path = Path(args.data_path)
    return args


if __name__ == "__main__":
    args = parse_arguments()
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=args.log, filemode='w', level=args.verbosity.upper())
    else:
        logging.basicConfig(level=args.verbosity.upper())
    utils.configure_environment(args.gpu_id)
    train(args)
    logging.info("Done!")
