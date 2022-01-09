"""Script for experiment execution."""
import logging
import os
import pickle
import random
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import yaml


# Set logging configuration
root = logging.getLogger()
root.setLevel(logging.DEBUG)
stdout_handler = logging.StreamHandler(sys.stdout)
file_handler = logging.FileHandler(
    f"execution_logs/{datetime.now().strftime('%Y-%m-%d-%H-%M')}-execution.logs"
)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
for handler in [stdout_handler, file_handler]:
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    root.addHandler(handler)

# Parameters
YAML_CONFIG_FILE = "group_vars/all.yaml"
DOCKEMU_SCRIPT = "./dockemu_execution.sh"
DOCKEMU_CLEANUP_SCRIPT = "./dockemu_cleanup.sh"
START_NUMBER_OF_CLIENTS = 2
END_NUMBER_OF_CLIENTS = 2
NUMBER_OF_EXECUTIONS = 1
NS3_NETWORK_SCRIPT = "tap-csma-virtual-machine-client-server"
NUMBER_OF_LEARNING_ROUNDS = 3
ERROR_RATE_FACTORS = [
    1,
    # 5,
    # 10
]
# Set integer for reproducibility
FIXED_SEED = None  # 42
TRAIN_DATASET_ENTRIES = 50000
TEST_DATASET_ENTRIES = 10000
BASE_CONTAINER_NAME = "fliot"
SRC_FOLDER = "/home/dockemu/src/dockemu/"
TIME_LOGGING_FORMAT = "%Y-%m-%d %H:%M:%S,%f"
MOUNTED_DOCKER_PREP_PATH = (
    "/home/dockemu/PycharmProjects/dockemu-ansible-fl/roles/preparation/files/mounted/"
)
TRAIN_DATASET_SPLIT_STRING_PICKLE_NAME = "pickle_train_split_string"
TEST_DATASET_SPLIT_STRING_PICKLE_NAME = "pickle_test_split_string"


def yaml_modification(**kwargs):
    """Opens and updates the YAML_CONFIG_FILE based on keyword arguments."""
    with open(YAML_CONFIG_FILE) as file:
        yaml_file = yaml.load(file, Loader=yaml.FullLoader)

    yaml_file.update(**kwargs)

    with open(YAML_CONFIG_FILE, "w") as file:
        yaml.dump(yaml_file, file)


def follow_file(file_name, happy_break_string, bad_break_string=None):
    """Follows a file and breaks if the happy or bad break_string appears."""
    while not os.path.isfile(file_name):
        # Wait till file is created...
        time.sleep(1)
    while True:
        file = open(file_name, "r")
        content = file.read()
        if happy_break_string in content:
            return
        elif bad_break_string and bad_break_string in content:
            raise Exception(f"Found '{happy_break_string}' in {file_name}")
        elif "Exception" in content:
            raise f"Found Exception in {file_name}"
        file.close()
        time.sleep(1)


def generate_dataset_split_string(clients_count, dataset_entries_count):
    """Generates a string with random integers in the number of clients, which together sum up to the number of entries
    in the dataset."""
    # Generate random float values
    random_numbers = [np.random.random_sample() for _ in range(clients_count)]
    # Divide each value by the sum of the random numbers and multiply it with the count of database entries. Finally,
    # cast it to int
    sum_random_numbers = np.sum(random_numbers)
    split_string = [
        int(np.round(i / sum_random_numbers * dataset_entries_count))
        for i in random_numbers
    ]
    # Get the rounding deviation which occurred in the previous operation, select a random value in the dataset and
    # equalize the deviation
    rounding_deviation = np.sum(split_string) - dataset_entries_count
    while True:
        random_position = random.randint(0, clients_count - 1)
        if split_string[random_position] > rounding_deviation:
            split_string[random_position] = (
                split_string[random_position] - rounding_deviation
            )
            break
    sum_up_split_string = []
    for i, v in enumerate(split_string):
        if i == 0:
            sum_up_split_string.append(v)
        else:
            sum_up_split_string.append(v + sum_up_split_string[i - 1])
    return sum_up_split_string


logging.info("-" * 30)
logging.info("Start experiment run")
logging.info("-" * 30)
for number_of_clients in range(START_NUMBER_OF_CLIENTS, END_NUMBER_OF_CLIENTS + 1):
    logging.info("-" * 30)
    logging.info(f"Start experiment with {number_of_clients} clients")

    for error_rate_factor in ERROR_RATE_FACTORS:

        for execution_number in range(NUMBER_OF_EXECUTIONS):
            logging.info("-" * 10)
            logging.info(f"Start execution number {execution_number + 1}")
            # Collect settings for experiment run
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            if not FIXED_SEED:
                seed = random.randint(1, 1000)
            else:
                seed = FIXED_SEED
            random.seed(seed)
            np.random.seed(seed)
            experiment_name = (
                f"{timestamp}_"
                f"{execution_number + 1}_"
                f"{NS3_NETWORK_SCRIPT}_"
                f"{number_of_clients}_"
                f"{seed}_"
                f"{str(error_rate_factor)}"
            )
            logging.info(f"Name of experiment {experiment_name}")
            experiment_parameters = {
                "experimentName": experiment_name,
                "baseContainerName": BASE_CONTAINER_NAME,
                "ns3NetworkScript": NS3_NETWORK_SCRIPT,
                "numberOfClientNodes": number_of_clients,
                "numberOfRounds": NUMBER_OF_LEARNING_ROUNDS,
                "seed": seed,
                "errorRateFactor": error_rate_factor,
            }
            log_folder = os.path.join(SRC_FOLDER, "logs", experiment_name, "logs")
            # Modify experiment configuration
            yaml_modification(**experiment_parameters)

            # Save pickle of train and test split string in the client folder.
            for pickle_name, dataset in zip(
                [
                    TRAIN_DATASET_SPLIT_STRING_PICKLE_NAME,
                    TEST_DATASET_SPLIT_STRING_PICKLE_NAME,
                ],
                [
                    TRAIN_DATASET_ENTRIES,
                    TEST_DATASET_ENTRIES,
                ],
            ):
                with open(
                    os.path.join(MOUNTED_DOCKER_PREP_PATH, pickle_name),
                    "wb",
                ) as f:
                    pickle.dump(
                        generate_dataset_split_string(number_of_clients, dataset),
                        f,
                    )

            logging.info("Script configuration finished. Execute Dockemu..")
            # Execute experiment
            subprocess.call(DOCKEMU_SCRIPT)

            # Observe successful connection from all clients to server
            seconds_for_client_start = 60
            logging.info(
                f"Wait {seconds_for_client_start} seconds for clients to startup"
            )
            time.sleep(60)
            logging.info("Scan client logs...")
            for client in range(START_NUMBER_OF_CLIENTS):
                client_name = f"{BASE_CONTAINER_NAME}-{client}"
                client_log_file = os.path.join(log_folder, client_name, "client.log")
                follow_file(client_log_file, "ChannelConnectivity.READY")
                logging.info(
                    f"Client {client_name} has established a connection to the server"
                )

            # Check for server logs
            logging.info("Scan server logs...")
            server_log_file = os.path.join(
                log_folder, f"{BASE_CONTAINER_NAME}-server-0", "server.log"
            )
            # Watch server logs
            follow_file(server_log_file, "FL finished")
            logging.info("Server collected all data from the clients")
            # Save the server logs
            server_log_file = open(server_log_file)
            # Wait till 'loss' is inside the file and experiment is finished
            file_content = server_log_file.readlines()
            for line in file_content:
                if "FL finished" in line:
                    time_from_flwr = line.split(" ")[-1]
                elif "Flower server running " in line:
                    start_time = line.split(" - ")[0]
                elif "losses_distributed" in line:
                    losses = line.split(" - ")[-1]
                elif "metrics_centralized" in line:
                    end_time = line.split(" - ")[0]

            total_time_needed = datetime.strptime(
                end_time, TIME_LOGGING_FORMAT
            ) - datetime.strptime(start_time, TIME_LOGGING_FORMAT)
            # Log experiment parameters
            logging.info(
                f"Experiment {experiment_name} finished with the following parameters: \n"
                f"start_time: {start_time}, \n"
                f"end_time: {end_time}, \n"
                f"total_time_needed: {total_time_needed}, \n"
                f"time_from_flwr {time_from_flwr}, \n"
                f"losses: {losses}."
            )

            # Log experiment parameters for each client
            for client in range(number_of_clients):
                client_name = f"{BASE_CONTAINER_NAME}-{client}"
                client_log_file = os.path.join(log_folder, client_name, "client.log")
                client_log_file = open(client_log_file)
                file_content = client_log_file.readlines()
                round_no = 1
                for line in file_content:
                    if "step" in line:
                        logging.info(
                            f"Client {client} finished run number {round_no} with following parameters: \n"
                            f"{line}"
                        )
                        round_no += 1

            # Cleanup environment
            # subprocess.call(DOCKEMU_CLEANUP_SCRIPT)
