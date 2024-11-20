import tensorflow as tf
from tensorflow.keras.layers import Layer
import keras.backend as K

class MakeOnes(Layer):
    """
    Seems to turn top row to ones.

    line 244: x = Lambda(lambda x: x[:, :1] * 0 + 1)(inp_style[0])
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        return x[:, :1] * 0 + 1


class Center(Layer):
    """
    Use values centered around 0, but normalize to [0, 1].
    
    Provides better initialization.
    ```
    line 281: x = Lambda(lambda y: y/2 + 0.5)(x)
    ```
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        return x / 2 + 0.5


class CropToFit(Layer):
    """
    @keras.saving.register_keras_serializable()
    def crop_to_fit(x):
        height = x[1].shape[1]
        width = x[1].shape[2]

        return x[0][:, :height, :width, :]
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        height = x[1].shape[1]
        width = x[1].shape[2]
        return x[0][:, :height, :width, :]


class UpsampleToSize(Layer):
    """
    @keras.saving.register_keras_serializable()
    def upsample_to_size(x):
        y = im_size // x.shape[2]
        x = K.resize_images(x, y, y, "channels_last",interpolation='bilinear')
        return x
    """
    def __init__(self, im_size, **kwargs):
        super().__init__(**kwargs)
        self.im_size = im_size

    def call(self, x):
        y = self.im_size // x.shape[2]
        x = K.resize_images(x, y, y, "channels_last", interpolation="bilinear")
        return x



class Upsample(Layer):
    """
    @keras.saving.register_keras_serializable()
    def upsample(x):
        return K.resize_images(x,2,2,"channels_last",interpolation='bilinear')
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        x = K.resize_images(x, 2, 2, "channels_last", interpolation="bilinear")
        return x