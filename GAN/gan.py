import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential


# Generator network
class Generator(keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        filter = 64
        # Transposed conv layer 1, output channel = filter * 8, kernel = 4, stride = 1, no padding, no bias
        self.conv1 = layers.Conv2DTranspose(filter * 8, (4, 4), strides=1, padding='valid', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        # Transposed conv layer 2
        self.conv2 = layers.Conv2DTranspose(filter * 4, (4, 4), strides=2, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        # Transposed conv layer 3
        self.conv3 = layers.Conv2DTranspose(filter * 2, (4, 4), strides=2, padding='same', use_bias=False)
        self.bn3 = layers.BatchNormalization()
        # Transposed conv layer 4
        self.conv4 = layers.Conv2DTranspose(filter * 1, (4, 4), strides=2, padding='same', use_bias=False)
        self.bn4 = layers.BatchNormalization()
        # Transposed conv layer 5
        self.conv5 = layers.Conv2DTranspose(1, (4, 4), strides=2, padding='same', use_bias=False)

    def call(self, inputs, training=None):
        x = inputs  # [z, 100]
        # Reshape to 4D tensor for transposed conv: (b, 1, 1, 100)
        x = tf.reshape(x, (x.shape[0], 1, 1, x.shape[1]))
        x = tf.nn.relu(x)
        # Transposed conv + BN + activation: (b, 4, 4, 512)
        x = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        # (b, 8, 8, 256)
        x = tf.nn.relu(self.bn2(self.conv2(x), training=training))
        # (b, 16, 16, 128)
        x = tf.nn.relu(self.bn3(self.conv3(x), training=training))
        # (b, 32, 32, 64)
        x = tf.nn.relu(self.bn4(self.conv4(x), training=training))
        # (b, 64, 64, 1)
        x = self.conv5(x)
        x = tf.tanh(x)  # Output x: [-1,1], matching preprocessing

        return x


# Discriminator network
class Discriminator(keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        filter = 64
        # Conv layer 1
        self.conv1 = layers.Conv2D(filter, (4, 4), strides=2, padding='valid', use_bias=False)
        self.bn1 = layers.BatchNormalization()

        # Conv layer 2
        self.conv2 = layers.Conv2D(filter * 2, (4, 4), strides=2, padding='valid', use_bias=False)
        self.bn2 = layers.BatchNormalization()

        # Conv layer 3
        self.conv3 = layers.Conv2D(filter * 4, (4, 4), strides=2, padding='valid', use_bias=False)
        self.bn3 = layers.BatchNormalization()

        # Conv layer 4
        self.conv4 = layers.Conv2D(filter * 8, (3, 3), strides=1, padding='valid', use_bias=False)
        self.bn4 = layers.BatchNormalization()

        # Conv layer 5
        self.conv5 = layers.Conv2D(filter * 16, (3, 3), strides=1, padding='valid', use_bias=False)
        self.bn5 = layers.BatchNormalization()

        # Global average pooling
        self.pool = layers.GlobalAveragePooling2D()
        # Flatten layer
        self.faltten = layers.Flatten()
        # Binary classification dense layer
        self.fc = layers.Dense(1)

        self.dropout = layers.Dropout(0.3)

    def call(self, inputs, training=None):
        # Conv-BN-activation: (4, 31, 31, 64)
        x = tf.nn.leaky_relu(self.bn1(self.conv1(inputs), training=training))
        # Conv-BN-activation: (4, 14, 14, 128)
        x = tf.nn.leaky_relu(self.bn2(self.conv2(x), training=training))
        # Conv-BN-activation: (4, 6, 6, 256)
        x = tf.nn.leaky_relu(self.bn3(self.conv3(x), training=training))
        # Conv-BN-activation: (4, 4, 4, 512)
        x = tf.nn.leaky_relu(self.bn4(self.conv4(x), training=training))
        # Conv-BN-activation: (4, 2, 2, 1024)
        x = tf.nn.leaky_relu(self.bn5(self.conv5(x), training=training))

        # Apply dropout
        x = self.dropout(tf.nn.leaky_relu(self.bn1(self.conv1(inputs), training=training)), training=training)
        x = self.dropout(tf.nn.leaky_relu(self.bn2(self.conv2(x), training=training)), training=training)
        x = self.dropout(tf.nn.leaky_relu(self.bn3(self.conv3(x), training=training)), training=training)
        x = self.dropout(tf.nn.leaky_relu(self.bn4(self.conv4(x), training=training)), training=training)
        x = self.dropout(tf.nn.leaky_relu(self.bn5(self.conv5(x), training=training)), training=training)

        # Global pooling and flatten
        x = self.pool(x)
        x = self.faltten(x)

        # Output: [b,1024] => [b,1]
        logits = self.fc(x)

        return logits


def main():
    d = Discriminator()
    g = Generator()

    x = tf.random.normal([2, 64, 64, 1])
    z = tf.random.normal([2, 100])

    prob = d(x)
    print(prob)
    x_hat = g(z)
    print(x_hat.shape)


if __name__ == '__main__':
    main()