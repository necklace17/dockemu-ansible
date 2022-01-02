import os
import pickle
import socket
import time

import flwr as fl
import tensorflow as tf
import logging
import sys

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
DATASET_PART = HOSTNAME.split("-")[1]
logging.info(f"Part of dataset to use: {DATASET_PART}")


SERVER_SOCKET = f"{os.getenv('SERVERNAME')}:{os.getenv('SERVERPORT')}"

if __name__ == "__main__":
    # Load and compile Keras model
    model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = pickle.load(open("pickle_cifar", "rb"))

    # Define Flower client
    class CifarClient(fl.client.NumPyClient):
        def get_parameters(self):  # type: ignore
            return model.get_weights()

        def fit(self, parameters, config):  # type: ignore
            model.set_weights(parameters)
            model.fit(x_train, y_train, epochs=1, batch_size=32)
            return model.get_weights(), len(x_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            model.set_weights(parameters)
            loss, accuracy = model.evaluate(x_test, y_test)
            return loss, len(x_test), {"accuracy": accuracy}

    # Start Flower client
    for i in range(0, 5):
        try:
            logging.info(f"Try connection attempt number {i}")
            fl.client.start_numpy_client(SERVER_SOCKET, client=CifarClient())
            exit()
        except Exception as error:
            logging.error(f"with error: {error}")
            wait_time = 5
            logging.info(f"Wait {wait_time} seconds")
            time.sleep(wait_time)
    logging.info("Connection was not successful.")
    logging.info("Pause script.")
    time.sleep(3000)
