# Summary of your model:
# Backbone: Lightweight CNN (Depthwise Separable Convs).
# Attention: CBAM at multiple scales.
# Global Context: Transformer bottleneck layers.
# Classifier: Dense layer → 4 output classes.
import tensorflow as tf
from tensorflow.keras import layers, Model
from model.cbam_module import CBAM
from model.dw_sep_conv import DepthwiseSeparableConv
from model.transformer_bottleneck import TransformerBottleneck

def build_hybrid_model(
    input_shape=(224,224,3), 
    num_classes=2, 
    base_ch=32, 
    d_model=128, 
    num_heads=4, 
    trans_layers=1,
    dropout_rate=0.2
):
    inputs = layers.Input(shape=input_shape)

    # Stem conv
    x = layers.Conv2D(base_ch, 3, strides=2, padding='same', use_bias=False, kernel_initializer="he_normal")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Stage 1
    x = DepthwiseSeparableConv(base_ch, dropout_rate=dropout_rate)(x)
    x = DepthwiseSeparableConv(base_ch, dropout_rate=dropout_rate)(x)
    x = CBAM(base_ch)(x)

    # Stage 2
    x = DepthwiseSeparableConv(base_ch*2, strides=2, dropout_rate=dropout_rate)(x)
    x = DepthwiseSeparableConv(base_ch*2, dropout_rate=dropout_rate)(x)
    x = CBAM(base_ch*2)(x)

    # Stage 3
    x = DepthwiseSeparableConv(base_ch*4, strides=2, dropout_rate=dropout_rate)(x)
    x = DepthwiseSeparableConv(base_ch*4, dropout_rate=dropout_rate)(x)
    x = CBAM(base_ch*4)(x)

    # Project channels to d_model (1x1 conv is better than Dense on flattened tokens)
    x = layers.Conv2D(d_model, 1, use_bias=False, kernel_initializer="he_normal")(x)

    # Transformer stack (each expects B,H,W,C)
    for _ in range(trans_layers):
        x = TransformerBottleneck(embed_dim=d_model, num_heads=num_heads, ff_dim=d_model*2)(x)

    # Global average pooling (2D, since we’re back in CNN format)
    features = layers.GlobalAveragePooling2D()(x)

    # Classifier head (no softmax, let loss handle it)
    outputs = layers.Dense(num_classes)(features)
    
    return Model(inputs, outputs)
model = build_hybrid_model(input_shape=(224,224,3), num_classes=4, trans_layers=2)
model.summary()
