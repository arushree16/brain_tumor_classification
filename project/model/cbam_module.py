import tensorflow as tf
from tensorflow.keras import layers

class ChannelAttention(layers.Layer):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden_units = max(1, channels // reduction)
        self.avg_pool = layers.GlobalAveragePooling2D(keepdims=True)
        self.max_pool = layers.GlobalMaxPooling2D(keepdims=True)
        self.fc1 = layers.Dense(hidden_units, activation='relu', use_bias=False)
        self.fc2 = layers.Dense(channels, use_bias=False)

    def call(self, x):
        avg_out = self.fc2(self.fc1(self.avg_pool(x)))
        max_out = self.fc2(self.fc1(self.max_pool(x)))
        scale = tf.nn.sigmoid(avg_out + max_out)
        return x * scale


class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = layers.Conv2D(
            1, kernel_size, padding='same', use_bias=False, activation='sigmoid'
        )

    def call(self, x):
        avg_out = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_out = tf.reduce_max(x, axis=-1, keepdims=True)
        concat = tf.concat([avg_out, max_out], axis=-1)
        scale = self.conv(concat)
        return x * scale


class CBAM(layers.Layer):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def call(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x
#small test to check
# x = tf.random.normal((2, 32, 32, 64))  # (batch, H, W, C)
# cbam = CBAM(channels=64)
# y = cbam(x)
# print("input:", x.shape, "output:", y.shape)
