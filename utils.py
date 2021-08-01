import logging


def configure_compute_environment(gpu_id, model):
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)  # set gpu visibility prior to importing autograd library
    if model == "icatcher":
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                logging.info("Physical GPUs: {}, Logical GPUs: {}".format(len(gpus), len(logical_gpus)))
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                logging.info(e)
        return gpu_id
    elif model == "icatcher+":
        if gpu_id == -1:
            return "cpu"
        else:
            return "cuda:0"
    else:
        raise NotImplementedError