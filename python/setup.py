import argparse
import os

parser = argparse.ArgumentParser(description="Parse Setup Arguments")
parser.add_argument("-n", "--name", metavar='Simulator Name', type=str, nargs="?",
                    dest='name', help='Name of the simulator run', default="default")
parser.add_argument("-b", "--batch", metavar='Batch Size', type=int, nargs="?",
                    dest='batch_size', help='Local Batch Size', default=32)
parser.add_argument("-e", "--epochs", metavar='Epochs', type=int, nargs="?",
                    dest='epochs', help='Local Epochs', default=1)
parser.add_argument("-t", "--total", metavar='Number of Clients', type=int, nargs="?",
                    dest='total', help='Number of Clients')
parser.add_argument("-g", "--gpu", metavar='GPU ID', type=str, nargs="?",
                    dest='gpu_id', help='GPU ID', default="0")
args = parser.parse_args()

import os
os.environ['TF_GPU_ID'] = args.gpu_id

from utils import load_h5, load_labels, separate_data_by_class, generate_client_dataset_files, load_dataset

trainX, trainY = load_h5("cifar/train_data.h5")
labels = load_labels("cifar/labels.h5")
separated_data = separate_data_by_class(trainX, trainY, labels)
datasets = generate_client_dataset_files(dataset=separated_data,
                                         directory=args.name,
                                         epochs=args.epochs,
                                         batch_size=args.batch_size,
                                         n_clients=args.total,
                                         n_samples_min=200,
                                         n_samples_max=1000,
                                         n_classes_min=3,
                                         n_classes_max=10,
                                         no_repeat=False)
