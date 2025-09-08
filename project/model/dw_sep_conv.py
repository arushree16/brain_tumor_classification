#MobileNetV1 Style
import tensorflow as tf
from tensorflow.keras import layers

class DepthwiseSeparableConv(layers.Layer):
    def __init__(self, filters, kernel_size=3, strides=1, padding='same', dropout_rate=0.0):
        super().__init__()
        self.dw = layers.DepthwiseConv2D(
            kernel_size, strides=strides, padding=padding, use_bias=False,
            depthwise_initializer="he_normal"
        )
        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.ReLU()

        self.pw = layers.Conv2D(
            filters, 1, use_bias=False, kernel_initializer="he_normal"
        )
        self.bn2 = layers.BatchNormalization()
        self.act2 = layers.ReLU()

        self.dropout = layers.SpatialDropout2D(dropout_rate) if dropout_rate > 0 else None

    def call(self, x, training=False):
        x = self.dw(x)
        x = self.bn1(x, training=training)
        x = self.act1(x)

        x = self.pw(x)
        x = self.bn2(x, training=training)
        x = self.act2(x)

        if self.dropout:
            x = self.dropout(x, training=training)

        return x
#small test to check
# x = tf.random.normal((2, 64, 64, 32))
# block = DepthwiseSeparableConv(filters=64, dropout_rate=0.2)
# y = block(x, training=True)
# print("input:", x.shape, "output:", y.shape)
