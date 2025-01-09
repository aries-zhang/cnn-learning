import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from pathlib import Path
from PIL import Image
import glob
import os
import numpy as np
import cv2

model_path = "models/regular/basic.h5"
num_pixels = 784


def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    x_train = x_train / 255
    x_test = x_test / 255

    x_train = x_train.reshape(x_train.shape[0], num_pixels)
    x_test = x_test.reshape(x_test.shape[0], num_pixels)

    return (x_train, y_train)


def create_model():
    model = Sequential()
    model.add(Dense(10, input_dim=num_pixels, activation="relu"))
    model.add(Dense(30, activation="relu"))
    model.add(Dense(10, activation="softmax"))
    model.compile(Adam(lr=0.01),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    print(model.summary())

    return model


def train_model():
    (x_train, y_train) = load_data()

    model = create_model()
    history = model.fit(x_train,
                        y_train,
                        validation_split=0.1,
                        epochs=10,
                        batch_size=200,
                        verbose=1,
                        shuffle=1)

    model.save(model_path)

    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.legend(["loss", "val_loss"])
    plt.show()

    return model


def prepare_model():
    model_file = Path(model_path)
    if model_file.exists():
        return load_model(model_path)
    else:
        return train_model()


def predict():
    model = prepare_model()
    for f in glob.iglob("./test_data/*"):
        test_image_name = os.path.basename(f)
        image = normalize_image(f)
        prediction = model.predict(image)
        index = np.argmax(prediction, axis=1)
        print("Prediction for image {} is: {}, probability: {}"
              .format(test_image_name, index, prediction[0][index]))


def normalize_image(path):
    image = Image.open(path)

    image_array = np.asarray(image)
    resized = cv2.resize(image_array, (28, 28))
    gray_scaled = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    image = cv2.bitwise_not(gray_scaled) / 255
    # plt.imshow(image)
    # plt.show()
    image = image.reshape(1, 784)

    return image


predict()
