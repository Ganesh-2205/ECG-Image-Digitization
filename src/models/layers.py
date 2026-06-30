"""
Custom layers and modular block definitions for semantic segmentation neural networks.
"""

import tensorflow as tf
from tensorflow.keras import layers


class ConvBlock(layers.Layer):
    """
    Standard convolutional block consisting of:
    [Conv2D -> BatchNormalization -> Activation] x 2
    """

    def __init__(
        self,
        filters: int,
        kernel_size: int = 3,
        dropout_rate: float = 0.1,
        activation: str = "relu",
        **kwargs
    ) -> None:
        """
        Initializes the convolutional block.

        Args:
            filters: Number of channels.
            kernel_size: Sizing of filter kernels.
            dropout_rate: Spatial dropout rate.
            activation: Activation function type.
        """
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.activation_name = activation

        self.conv1 = layers.Conv2D(
            filters, kernel_size, padding="same", kernel_initializer="he_normal"
        )
        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.Activation(activation)
        self.drop = layers.Dropout(dropout_rate) if dropout_rate > 0 else None

        self.conv2 = layers.Conv2D(
            filters, kernel_size, padding="same", kernel_initializer="he_normal"
        )
        self.bn2 = layers.BatchNormalization()
        self.act2 = layers.Activation(activation)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Executes block operations.
        """
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.act1(x)

        if self.drop:
            x = self.drop(x, training=training)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.act2(x)
        
        return x


class UpsampleBlock(layers.Layer):
    """
    Standard decoder block in UNet:
    Conv2DTranspose -> Concatenate with skip connection -> ConvBlock
    """

    def __init__(
        self,
        filters: int,
        dropout_rate: float = 0.1,
        activation: str = "relu",
        **kwargs
    ) -> None:
        """
        Initializes the upsampling decoder block.

        Args:
            filters: Number of target filters.
            dropout_rate: Spatial dropout rate for conv block.
            activation: Activation type.
        """
        super().__init__(**kwargs)
        self.filters = filters
        
        self.up = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding="same")
        self.concat = layers.Concatenate()
        self.conv = ConvBlock(filters, dropout_rate=dropout_rate, activation=activation)

    def call(self, inputs: tf.Tensor, skip_connect: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Executes block operations.
        """
        x = self.up(inputs)
        x = self.concat([x, skip_connect])
        x = self.conv(x, training=training)
        return x
