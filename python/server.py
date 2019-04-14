from model import KerasModel
from utils import load_dummy
from random import sample
import argparse
import os
import sys
import json
import tensorflow as tf
tf.config.gpu.set_per_process_memory_fraction(0.5)
tf.config.gpu.set_per_process_memory_growth(True)

parser = argparse.ArgumentParser(description="Parse Server Arguments")
parser.add_argument("-c", "--clients", metavar='Clients Per Iteration', type=int, nargs="?",
                    dest='clients', help='Clients Per Iteration', default=1)
parser.add_argument("-t", "--total", metavar='Number of Clients', type=int, nargs="?",
                    dest='total', help='Number of Clients')
parser.add_argument("-i", "--iterations", metavar='Iterations', type=int, nargs="?",
                    dest='iterations', help='Iterations', default=10000)
parser.add_argument("-n", "--name", metavar='Simulator Name', type=str, nargs="?",
                    dest='name', help='Name of the simulator run', default="default")
args = parser.parse_args()


class Server():
    def __init__(self):
        optimizer = tf.keras.optimizers.Adam()
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        loss_metrics = tf.keras.metrics.Mean(name='loss')
        acc_metrics = tf.keras.metrics.SparseCategoricalAccuracy(name='acc')

        # Generate the Keras Model
        dummy_data = load_dummy(args.name)
        self.model = KerasModel()
        self.model(dummy_data)
        self.current_iteration = 1
        self.total_iterations = args.iterations
        self.client_count = args.total
        self.clients_per_round = args.clients
        self.client_history = list()

    def iterate(self, iteration: int):
        weights_file_path = os.path.join("temp", args.name, "server_weights_step_{}.h5".format(iteration))
        # Runs one iteration
        self.model.save_weights(weights_file_path)
        chosen_clients = sample(range(self.client_count), self.clients_per_round)
        message = {
            "type": "train",
            "clients": chosen_clients,
            "weights_file_path": weights_file_path,
            "step": iteration
        }
        sys.stdout.write(json.dumps(message))
        sys.stdout.flush()
        response = sys.stdin.readline()

    def listen(self):
        command = self.readline()
        while(command != "exit"):
            if command == "start":
                self.train()
            command = self.readline()
        sys.exit(0)

    def readline(self):
        return sys.stdin.readline().strip()

    def train(self):
        for i in range(1, self.total_iterations+1):
            self.iterate(i)
            self.current_iteration += 1


Server().listen()
