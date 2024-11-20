"""
 |  The main reason to subclass `tf.keras.layers.Layer` instead of using a
 |  `Lambda` layer is saving and inspecting a Model. `Lambda` layers
 |  are saved by serializing the Python bytecode, which is fundamentally
 |  non-portable. They should only be loaded in the same environment where
 |  they were saved. Subclassed layers can be saved in a more portable way
 |  by overriding their `get_config()` method. Models that rely on
 |  subclassed Layers are also often easier to visualize and reason about.
 |
 |  ```python
 |  scale = tf.Variable(1.)
 |  scale_layer = tf.keras.layers.Lambda(lambda x: x * scale)
 |  ```
 |  
 |  Because `scale_layer` does not directly track the `scale` variable, it will
 |  not appear in `scale_layer.trainable_weights` and will therefore not be
 |  trained if `scale_layer` is used in a Model.
 |  
 |  A better pattern is to write a subclassed Layer:
 |  
 |  ```python
 |  class ScaleLayer(tf.keras.layers.Layer):
 |      def __init__(self, **kwargs):
 |          super().__init__(**kwargs)
 |          self.scale = tf.Variable(1.)
 |  
 |      def call(self, inputs):
 |          return inputs * self.scale
 |  ``` 
"""
import tensorflow as tf
from tensorflow.keras.layers import Layer
import keras.backend as K

class MakeOnesLambda(Layer):
    """
    Seems to turn top row to ones.

    line 244: x = Lambda(lambda x: x[:, :1] * 0 + 1)(inp_style[0])
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        return x[:, :1] * 0 + 1


class CenterLambda(Layer):
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

class CropToFitLambda(Layer):
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


class UpsampleToSizeLambda(Layer):
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



class UpsampleLambda(Layer):
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