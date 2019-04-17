from tensorflow.python.keras import layers, Model, regularizers


class KerasModel(Model):
    def __init__(self):
        super(KerasModel, self).__init__()
        weight_decay = 1e-4;
        self.conv1 = layers.Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3),
                                   kernel_regularizer=regularizers.l2(weight_decay))
        self.elu1 = layers.ELU()
        self.bn1 = layers.BatchNormalizationV2()
        self.conv2 = layers.Conv2D(32, (3, 3), kernel_regularizer=regularizers.l2(weight_decay))
        self.elu2 = layers.ELU()
        self.bn2 = layers.BatchNormalizationV2()
        self.pool1 = layers.MaxPool2D(pool_size=(2, 2))
        self.dropout1 = layers.Dropout(rate=0.2)

        self.conv3 = layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))
        self.elu3 = layers.ELU()
        self.bn3 = layers.BatchNormalizationV2()
        self.conv4 = layers.Conv2D(64, (3, 3), kernel_regularizer=regularizers.l2(weight_decay))
        self.elu4 = layers.ELU()
        self.bn4 = layers.BatchNormalizationV2()
        self.pool2 = layers.MaxPool2D(pool_size=(2, 2))
        self.dropout2 = layers.Dropout(rate=0.3)

        self.conv5 = layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))
        self.elu5 = layers.ELU()
        self.bn5 = layers.BatchNormalizationV2()
        self.conv6 = layers.Conv2D(128, (3, 3), kernel_regularizer=regularizers.l2(weight_decay))
        self.elu6 = layers.ELU()
        self.bn6 = layers.BatchNormalizationV2()
        self.pool3 = layers.MaxPool2D(pool_size=(2, 2))
        self.dropout3 = layers.Dropout(rate=0.4)

        self.flatten1 = layers.Flatten()
        self.dense1 = layers.Dense(512)
        self.elu7 = layers.ELU()
        self.dropout4 = layers.Dropout(rate=0.5)
        self.dense2 = layers.Dense(10)
        self.softmax = layers.Softmax()

    def call(self, inputs, training: bool = True):
        x = self.conv1(inputs)
        x = self.elu1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.elu2(x)
        x = self.bn2(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = self.elu3(x)
        x = self.bn3(x)
        x = self.conv4(x)
        x = self.elu4(x)
        x = self.bn4(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.conv5(x)
        x = self.elu5(x)
        x = self.bn5(x)
        x = self.conv6(x)
        x = self.elu6(x)
        x = self.bn6(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        x = self.flatten1(x)
        x = self.dense1(x)
        x = self.elu7(x)
        x = self.dropout4(x)
        x = self.dense2(x)
        return self.softmax(x)
