import tensorflow as tf


def create_shared_model():
    image_size = 75  # All images will be resized to this value
    img_shape = (image_size, image_size, 3)
    img_input = tf.keras.layers.Input(shape=img_shape)
    conv1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', name='conv1')(img_input)
    maxpool1 = tf.keras.layers.MaxPooling2D(2, 2)(conv1)
    conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', name='conv2')(maxpool1)
    conv22 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', name='conv22')(conv2)
    maxpool2 = tf.keras.layers.MaxPooling2D(2, 2)(conv22)
    conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', name='conv3')(maxpool2)
    conv32 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', name='conv32')(conv3)
    maxpool3 = tf.keras.layers.MaxPooling2D(2, 2)(conv32)
    flatten = tf.keras.layers.Flatten()(maxpool3)
    model = tf.keras.Model(inputs=img_input, outputs=flatten, name='shared_cnn')
    return model


def build_baseline_model_5image(learning_rate, is_2class_model=False):
    image_size = 75  # All images will be resized to this value
    img_shape = (image_size, image_size, 3)
    input1 = tf.keras.layers.Input(shape=img_shape)
    input2 = tf.keras.layers.Input(shape=img_shape)
    input3 = tf.keras.layers.Input(shape=img_shape)
    input4 = tf.keras.layers.Input(shape=img_shape)
    input5 = tf.keras.layers.Input(shape=img_shape)
    shared_model = create_shared_model()
    x1 = shared_model(input1)
    x2 = shared_model(input2)
    x3 = shared_model(input3)
    x4 = shared_model(input4)
    x5 = shared_model(input5)
    merge_layer = tf.keras.layers.Concatenate()([x1, x2, x3, x4, x5])
    hidden1 = tf.keras.layers.Dense(32, activation='relu', name='fc1')(merge_layer)
    hidden2 = tf.keras.layers.Dense(16, activation='relu', name='fc2')(hidden1)
    if is_2class_model:
        out = tf.keras.layers.Dense(2, activation='softmax', name='fc3')(hidden2)
    else:
        out = tf.keras.layers.Dense(3, activation='softmax', name='fc3')(hidden2)
    model = tf.keras.Model(
        inputs=[input1, input2, input3, input4, input5],
        outputs=out,
        name='my_small_vgg')

    opt = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # model.summary()
    # for layer in model.layers:
    #     print("layer:", layer, "trainable:", layer.trainable)
    # internal_model = model.get_layer('shared_cnn')
    # print("shared_cnn:")
    # for layer in internal_model.layers:
    #     print("layer:", layer, "trainable:", layer.trainable)
    return model


def fine_tune_baseline_model(weight_location, learning_rate):
    model = tf.keras.models.load_model(str(weight_location))
    opt = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    # model.summary()
    return model