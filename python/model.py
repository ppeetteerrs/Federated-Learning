import tensorflow as tf
from tensorflow.keras import layers, Model


class KerasModel(Model):
    def __init__(self):
        super(KerasModel, self).__init__()
        self.conv1 = layers.Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3))
        self.relu1 = layers.ReLU()
        self.conv2 = layers.Conv2D(32, (3, 3))
        self.relu2 = layers.ReLU()
        self.pool1 = layers.MaxPool2D(pool_size=(2, 2))
        self.dropout1 = layers.Dropout(rate=0.25)

        self.conv3 = layers.Conv2D(64, (3, 3), padding='same')
        self.relu3 = layers.ReLU()
        self.conv4 = layers.Conv2D(64, (3, 3))
        self.relu4 = layers.ReLU()
        self.pool2 = layers.MaxPool2D(pool_size=(2, 2))
        self.dropout2 = layers.Dropout(rate=0.25)

        self.flatten1 = layers.Flatten()
        self.dense1 = layers.Dense(512)
        self.relu5 = layers.ReLU()
        self.dropout3 = layers.Dropout(rate=0.5)
        self.dense2 = layers.Dense(10)
        self.softmax = layers.Softmax()

    def call(self, inputs, training: bool = True):
        x = self.conv1(inputs)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.flatten1(x)
        x = self.dense1(x)
        x = self.relu5(x)
        x = self.dropout3(x)
        x = self.dense2(x)
        return self.softmax(x)
