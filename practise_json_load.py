# %%
import numpy as np
import tensorflow as tf

x = np.random.random((1000, 32))
y = np.random.random((1000, 1))

@tf.keras.utils.register_keras_serializable()
class MModel(tf.keras.Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = tf.keras.layers.Dense(10)
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

    def get_config(self):
        config = super().get_config()
        return config


model = MModel()

model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.fit(x, y, epochs=5)

model.save("save1.json")
model_ = tf.keras.models.load_model("save1.json")

print()
print(f"{model.evaluate(x, y,verbose=0)  = }")
print(f"{model_.evaluate(x, y,verbose=0) = }")

for m, m_ in zip(model.weights, model_.weights):
    np.testing.assert_allclose(m.numpy(), m_.numpy())
# %%