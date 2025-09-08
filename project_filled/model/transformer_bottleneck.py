import tensorflow as tf
from tensorflow.keras import layers

class TransformerBottleneck(layers.Layer):
    def __init__(self, embed_dim, num_heads=4, ff_dim=128, rate=0.2):
        super().__init__()
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads
        )
        self.dropout1 = layers.Dropout(rate)

        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dropout(rate),
            layers.Dense(embed_dim),
        ])
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        # inputs: (B, H, W, C)
        B, H, W, C = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], tf.shape(inputs)[3]
        x = tf.reshape(inputs, (B, H*W, C))  # flatten spatial dims

        # Multi-head attention
        attn_output = self.att(x, x)
        attn_output = self.dropout1(attn_output, training=training)
        x = self.norm1(x + attn_output)

        # Feed-forward network
        ffn_output = self.ffn(x, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        x = self.norm2(x + ffn_output)

        # reshape back to (B, H, W, C)
        return tf.reshape(x, (B, H, W, C))
# x = tf.random.normal((2, 32, 32, 64))  # (batch, H, W, C)
# block = TransformerBottleneck(embed_dim=64, num_heads=4, ff_dim=128)
# y = block(x, training=True)
# print("input:", x.shape, "output:", y.shape)
