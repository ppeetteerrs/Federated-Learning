import argparse
from utils import load_h5, load_labels, separate_data_by_class, generate_client_dataset_files, load_dataset

parser = argparse.ArgumentParser(description="Parse Setup Arguments")
parser.add_argument("-n", "--name", type=str, nargs="?",
                    dest='name', help='Name of the dataset')
parser.add_argument("-t", "--total",  type=int, nargs="?",
                    dest='total', help='Total Number of Clients')
parser.add_argument("-s", "--minsample", type=int, nargs="?",
                    dest='minsample', help='Minimum Number of Samples')
parser.add_argument("-u", "--maxsample", type=int, nargs="?",
                    dest='maxsample', help='Maximum Number of Samples')
parser.add_argument("-v", "--minclass", type=int, nargs="?",
                    dest='minclass', help='Minimum Number of Classes')
parser.add_argument("-w", "--maxclass", type=int, nargs="?",
                    dest='maxclass', help='Maximum Number of Classes')
parser.add_argument("-r", "--repeat", type=bool, nargs="?",
                    dest='repeat', help='Allow Repeated Data Among Clients')
args = parser.parse_args()

trainX, trainY = load_h5("cifar/train_data.h5")
labels = load_labels("cifar/labels.h5")
separated_data = separate_data_by_class(trainX, trainY, labels)
generate_client_dataset_files(dataset=separated_data,
                              directory=args.name,
                              n_clients=args.total,
                              n_samples_min=args.minsample,
                              n_samples_max=args.maxsample,
                              n_classes_min=args.minclass,
                              n_classes_max=args.maxclass,
                              no_repeat=not (args.repeat))
