import os

import flwr as fl
import logging
import sys

root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
root.addHandler(handler)

# Start Flower server for three rounds of federated learning

strategy = fl.server.strategy.FedAvg(fraction_fit=0.5, fraction_eval=0.5)

if __name__ == "__main__":
    try:
        fl.server.start_server(
            f"0.0.0.0:{os.getenv('SERVERPORT')}",
            config={"num_rounds": int(os.getenv("NUMBER_OF_ROUNDS"))},
            strategy=strategy,
            force_final_distributed_eval=True,
        )
    except Exception as error:
        os.error(error)
