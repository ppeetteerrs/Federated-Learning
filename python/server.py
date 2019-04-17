import argparse
from random import sample
import sys
import json
import tensorflow as tf
from model import KerasModel
from utils import load_dummy, load_gradients, load_test_dataset
import os

parser = argparse.ArgumentParser(description="Parse Server Arguments")
parser.add_argument("-c", "--clients", metavar='Clients Per Iteration', type=int, nargs="?",
                    dest='clients', help='Clients Per Iteration', default=1)
parser.add_argument("-t", "--total", metavar='Number of Clients', type=int, nargs="?",
                    dest='total', help='Number of Clients')
parser.add_argument("-i", "--iterations", metavar='Iterations', type=int, nargs="?",
                    dest='iterations', help='Iterations', default=10000)
parser.add_argument("-n", "--name", metavar='Simulator Name', type=str, nargs="?",
                    dest='name', help='Name of the simulator run', default="default")
parser.add_argument("-d", "--datasetname", metavar='Dataset Name', type=str, nargs="?",
                    dest='datasetname', help='Name of the dataset', default="default")
parser.add_argument("-f", "--fraction", metavar='GPU Fraction', type=float, nargs="?",
                    dest='gpu_fraction', help='GPU Fraction', default=0.15)
args = parser.parse_args()

tf.config.gpu.set_per_process_memory_fraction(args.gpu_fraction)


class Server():
    def __init__(self):
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy()
        self.loss_metrics = tf.keras.metrics.Mean(name='loss')
        self.acc_metrics = tf.keras.metrics.SparseCategoricalAccuracy(name='acc')
        log_dir = "logs/{}".format(args.name)
        self.summary_writer = tf.summary.create_file_writer(logdir=log_dir)
        self.summary_writer.set_as_default()

        # Generate the Keras Model
        dummy_data = load_dummy(args.datasetname)
        self.model = KerasModel()
        self.model(dummy_data)
        self.current_iteration = 1
        self.total_iterations = args.iterations
        self.client_count = args.total
        self.clients_per_round = args.clients
        self.client_history = list()
        self.test_data = load_test_dataset("cifar/test_data.h5")

    def iterate(self, iteration: int):
        weights_file_path = os.path.join("temp", args.name, "weights_server.h5")
        # Output weights
        self.model.save_weights(weights_file_path)

        # Choose clients
        chosen_clients = sample(range(self.client_count), self.clients_per_round)

        # Inform simulator to train clients
        message = {
            "type": "train",
            "clients": chosen_clients,
            "weights_file_path": weights_file_path,
            "step": iteration
        }
        print(json.dumps(message), flush=True)

        # Get successful clients
        response = json.loads(self.readline())["ids"]
        # print(json.dumps({
        #     "type": "update",
        #     "message": "Received data from clients {}".format(str(response)),
        #     "step": iteration
        # }), flush=True)
        self.client_history.append(response)

        # Apply client updates
        self.update_weights(iteration, response)
        self.test(iteration)

    def update_weights(self, iteration: int, clients: [int]):
        gradients = load_gradients(args.name, iteration, clients)
        # print(json.dumps({
        #     "type": "log",
        #     "message": str(gradients[0].shape),
        #     "step": iteration
        # }), flush=True)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def test(self, iteration: int):
        for batch in iter(self.test_data):
            test_predictions = self.model(batch["x"])
            loss = self.loss(batch["y"], test_predictions)
            self.loss_metrics(loss)
            self.acc_metrics(batch["y"], test_predictions)
        average_loss = self.loss_metrics.result()
        average_acc = self.acc_metrics.result()
        print(json.dumps({
            "type": "update",
            "message": ("Test Loss: {:.5f}, Test Accuracy: {:.3f}%"
                        .format(average_loss, average_acc * 100)),
            "step": iteration
        }), flush=True)
        tf.summary.scalar("loss", average_loss, step=iteration)
        tf.summary.scalar("accuracy", average_acc, step=iteration)
        self.loss_metrics.reset_states()
        self.acc_metrics.reset_states()

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
