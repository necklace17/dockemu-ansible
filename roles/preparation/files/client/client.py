import os
import pickle
import socket
import time

import flwr as fl
import tensorflow as tf
import logging
import sys

MOUNTED_PATH = "/var/mounted"

root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
root.addHandler(handler)

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

HOSTNAME = socket.gethostname()
logging.info(f"Hostname :{HOSTNAME}")

SERVER_SOCKET = f"{os.getenv('SERVERNAME')}:{os.getenv('SERVERPORT')}"

dataset_split_string = pickle.load(
    open(os.path.join(MOUNTED_PATH, "pickle_split_string"), "rb")
)
HOST_NUMBER = int(HOSTNAME.split("-")[1])

if HOST_NUMBER == 0:
    SPLIT_BEFORE = 0
else:
    SPLIT_BEFORE = dataset_split_string[HOST_NUMBER - 1]
MY_SPLIT_STRING = dataset_split_string[HOST_NUMBER]
logging.info(
    f"My host number is {str(HOST_NUMBER)}. I will take dataset part [{SPLIT_BEFORE}:{MY_SPLIT_STRING}]"
)

if __name__ == "__main__":
    # Load and compile Keras model
    model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = pickle.load(
        open(os.path.join(MOUNTED_PATH, "../mounted/pickle_cifar"), "rb")
    )

    # Define Flower client
    class CifarClient(fl.client.NumPyClient):
        def get_parameters(self):  # type: ignore
            return model.get_weights()

        def fit(self, parameters, config):  # type: ignore
            model.set_weights(parameters)
            model.fit(
                x_train[SPLIT_BEFORE:MY_SPLIT_STRING],
                y_train[SPLIT_BEFORE:MY_SPLIT_STRING],
                epochs=1,
                batch_size=32,
            )
            return model.get_weights(), len(x_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            model.set_weights(parameters)
            loss, accuracy = model.evaluate(x_test, y_test)
            return loss, len(x_test), {"accuracy": accuracy}

    # Start Flower client
    counter = 0
    while True:
        try:
            logging.info(f"Try connection attempt number {counter}")
            fl.client.start_numpy_client(SERVER_SOCKET, client=CifarClient())
            exit()
        except Exception as error:
            logging.error(f"with error: {error}")
            wait_time = 5
            logging.info(f"Wait {wait_time} seconds")
            time.sleep(wait_time)
        counter += 1
    logging.info("Connection was not successful.")
    logging.info("Pause script.")
    time.sleep(3000)
