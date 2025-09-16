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
    input_shape=(224, 224, 3),
    num_classes=4,
    d_model=128,
    num_heads=4,
    trans_layers=1,
    dropout_rate=0.2,
    pretrained=True
):
    """
    Hybrid model with pretrained EfficientNetB0 backbone + CBAM + Transformer bottleneck
    """
    # -------------------------
    # Pretrained Backbone
    # -------------------------
    backbone = tf.keras.applications.EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet' if pretrained else None
    )
    backbone.trainable = False  # freeze initially; fine-tune later

    inputs = layers.Input(shape=input_shape)
    x = backbone(inputs)

    # Optional: add a 1x1 conv to project channels to d_model
    x = layers.Conv2D(d_model, 1, padding='same', use_bias=False, kernel_initializer="he_normal")(x)

    # -------------------------
    # CBAM + Transformer stages
    # -------------------------
    # CBAM after backbone
    x = CBAM(d_model)(x)

    # Transformer bottleneck stack
    for _ in range(trans_layers):
        x = TransformerBottleneck(embed_dim=d_model, num_heads=num_heads, ff_dim=d_model*2, rate=dropout_rate)(x)

    # Global average pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Dropout for regularization
    x = layers.Dropout(dropout_rate)(x)

    # Classifier head
    outputs = layers.Dense(num_classes)(x)  # no softmax; let loss handle it

    model = Model(inputs, outputs)
    return model

# Removed import-time model creation to avoid side effects during imports.


def build_hybrid_attention_cnn(
    input_shape=(256, 256, 3),
    num_classes=4,
    base_filters=32,
    stages=(1, 2, 3),
    dropout_rate=0.2,
    d_model=128,
    num_heads=4,
    trans_layers=1,
    pre_transform_downsample=False,
    downsample_type='avg'
):
    """
    Compact Hybrid-Attention CNN with DWConv + CBAM + Transformer bottleneck.
    Encoder: depthwise separable conv blocks with CBAM per stage
    Bottleneck: Transformer blocks over the final feature map
    Head: GAP -> Dropout -> Dense(num_classes) logits
    """
    inputs = layers.Input(shape=input_shape)

    x = inputs
    filters = base_filters

    # Encoder stages
    for stage_idx, num_blocks in enumerate(stages):
        # First block of a stage can downsample
        stride = 2 if stage_idx > 0 else 1
        x = DepthwiseSeparableConv(
            filters=filters,
            kernel_size=3,
            strides=stride,
            padding='same',
            dropout_rate=dropout_rate
        )(x)
        for _ in range(1, num_blocks):
            x = DepthwiseSeparableConv(
                filters=filters,
                kernel_size=3,
                strides=1,
                padding='same',
                dropout_rate=dropout_rate
            )(x)
        # Attention refinement per stage
        x = CBAM(channels=filters)(x)
        # Increase channels for next stage
        filters *= 2

    # Optional downsampling before Transformer to reduce tokens
    if pre_transform_downsample:
        if downsample_type == 'avg':
            x = layers.AveragePooling2D(pool_size=2, strides=2, padding='same')(x)
        elif downsample_type == 'conv':
            x = layers.Conv2D(filters=int(x.shape[-1]), kernel_size=3, strides=2, padding='same', use_bias=False,
                              kernel_initializer="he_normal")(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
        else:
            raise ValueError('Unknown downsample_type')

    # Project to d_model if needed
    if x.shape[-1] != d_model:
        x = layers.Conv2D(d_model, kernel_size=1, padding='same', use_bias=False, kernel_initializer="he_normal")(x)

    # Transformer bottleneck layers
    for _ in range(trans_layers):
        x = TransformerBottleneck(embed_dim=d_model, num_heads=num_heads, ff_dim=d_model * 2, rate=dropout_rate)(x)

    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes)(x)  # logits

    model = Model(inputs, outputs, name="HybridAttentionCNN")
    return model
