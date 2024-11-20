# %%
from PIL import Image
import numpy as np
import keras
import tensorflow as tf
from StyleGAN2 import StyleGAN
# TODO: import noiselist, nImage

im_size = 32 # 256
latent_size = 512
BATCH_SIZE = 16
POLICY = "color,translation,cutout"

# %% Workspace --loading pretrained
# from keras.src.saving import serialization_lib
# serialization_lib.enable_unsafe_deserialization()
#* >> ValueError: bad marshal data (unknown type code)

model = StyleGAN()
model.save()





# %%
model.loadModel("gen", 19) 
#* >> ValueError: Requested the deserialization of a Lambda layer with a Python `lambda` inside it. This carries a potential risk of arbitrary code execution and thus it is disallowed by default. If you trust the source of the saved model, you can pass `safe_mode=False
#! but safe_mode is not a valid arg 

# %%
for i, layer in enumerate(model.GAN.G.layers):
    if isinstance(layer, tf.keras.layers.Lambda):
        print(layer.get_config()['name'])
# %% Claude suggestions
custom_objects = {
    'lambda': lambda x: x[:, :1] * 0 + 1,
    'lambda_12': lambda y: y/2 + 0.5
    }

# %% Inspect J
import json

with open('Models/gen.json', 'r') as f:
    model_dict = json.load(f)

# Print the layer configurations to see what lambdas exist
for layer in model_dict['config']['layers']:
    if layer['class_name'] == 'Lambda':
        print(f"Found Lambda layer: {layer}")

# %% Inspect 2
import json

with open('Models/gen.json', 'r') as f:
    model_dict = json.load(f)
    
# Find all Lambda layers
lambda_layers = [layer for layer in model_dict['config']['layers'] 
                if layer['class_name'] == 'Lambda']

# Print their configurations
for layer in lambda_layers:
    print(f"Layer name: {layer['config']['name']}")
    print(f"Layer config: {layer['config']}")
# %%
MODE = "pretrained"

if __name__ == "__main__":
    train_size = 100 # 100-shot-learning
    batch_size = 16

    # get CIFAR10 data (add pandas another time?)
    datastr = "cifar10"
    (x_train, y_train), (x_test, y_test) = getattr(tf.keras.datasets, datastr).load_data()
    x_train = x_train[:train_size, ...]
    x_train = tf.image.resize(x_train, (im_size, im_size))

    # view one train image
    img = Image.fromarray(np.uint8(x_train[0, ...] * 255)).convert('RGB')
    display(img)


    # make dataset
    dataset = tf.data.Dataset.from_tensor_slices((x_train,)) \
        .repeat() \
        .batch(batch_size) \
        .prefetch(tf.data.AUTOTUNE)

    with tf.device('/GPU:0'):
        dataset = dataset.map(lambda x: tf.cast(x, tf.float32))


    model = StyleGAN(dataset, lr=0.0001, silent=False)
    model.evaluate(0)

    if MODE == "training":
        # while model.GAN.steps < 1000: # 1000001:
        while model.GAN.images_seen < 300_000:
            try:
                model.train()
            except Exception as e:
                print(e)

        print("Done! Plotting")
        n1 = noiseList(64)
        n2 = nImage(64)
        im = model.generateTruncated(n1, noi=n2, trunc=50/50, outImage=True, num=0)
        for i in range(64):
            img = Image.fromarray(np.uint8(im[i, ...] * 255)).convert('RGB')
            img.save(f"figures/{datastr}_{i}_300k.png")
        display(img)
    

    import json
    with open("/Users/alison/Documents/DPhil/github/stylegan2-silicon/Models/dis.json", "r") as f:
        model_json = json.dumps(json.load(f))
    model = keras.models.model_from_json(model_json)
    n1 = noiseList(64)
    n2 = nImage(64)
    im = model.generateTruncated(n1, noi=n2, trunc=50/50, outImage=True, num=0)
    for i in range(64):
        img = Image.fromarray(np.uint8(im[i, ...] * 255)).convert('RGB')
        img.save(f"figures/{datastr}_{i}_300k.png")
        display(img)

    # %%
    from diffaugment.data import create_from_images

    datadir = "data/100-shot-panda"
    create_from_images(datadir, 32, channels_first=False)

    # load tfrecord dataset
    """
    Commands:

        ```python
        model.load(31)

        n1 = noiseList(64)
        n2 = nImage(64)
        for i in range(50):
            print(i, end = '\r')
            model.generateTruncated(n1, noi = n2, trunc = i / 50, outImage = True, num = i)
        ```
    """

# %%
"""
Debugging:
    - loading smaller CIFAR10 data n=10: this may have fixed for now (only need 100 for DiffAugment anyway)
    - error handling: 
    - only running train() once, ie, not in while loop: 
    - examine model.train(), can you specify epochs?
    - psutil: won't print
    - try batching the dataset: didn't help

Other issues:
    - what does model.load() do?
    - model.train() seems to be a single step


Error handling:
    ```
    import tensorflow as tf

    # Wrap critical sections
    try:
        # Your training code here
        with tf.device('/GPU:0'):
            # Specific training operations
    except tf.errors.ResourceExhaustedError as e:
        print("GPU Memory Overflow:", e)
    except Exception as e:
        print("Unexpected Error:", e)
    ```
"""