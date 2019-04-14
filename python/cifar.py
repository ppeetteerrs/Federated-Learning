import pickle
cifar_labels = ["airplane", "automobile", "bird", "cat",
                "deer", "dog", "frog", "horse", "ship", "truck"]

with open("cifar/labels.h5", "wb") as file:
    pickle.dump(cifar_labels, file)
