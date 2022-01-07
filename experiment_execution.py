"""Script for experiment execution."""
import logging
import os
import random
import subprocess
import sys
import time
from datetime import datetime
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
ERROR_RATE_FACTORS = [1, 5, 10]
BASE_CONTAINER_NAME = "fliot"
SRC_FOLDER = "/home/dockemu/src/dockemu/"
TIME_LOGGING_FORMAT = "%Y-%m-%d %H:%M:%S,%f"


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
            seed = random.randint(1, 1000)
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
            follow_file(server_log_file, "Finish")
            logging.info("Server collected all data from the clients")
            # Save the server logs
            server_log_file = open(server_log_file)
            file_content = server_log_file.readlines()
            for line in file_content:
                if "Flower server running " in line:
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

            # TODO: Collect host statistics and save to experiment folder
            # 1. TODO: Collect CPU usage
            # 2. TODO: Collect GPU usage
            # 3. TODO: Collect memory usage

            # Cleanup environment
            subprocess.call(DOCKEMU_CLEANUP_SCRIPT)
