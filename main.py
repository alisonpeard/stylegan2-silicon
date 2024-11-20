# %%
from PIL import Image
from math import log2
from IPython.display import display
import numpy as np
import tensorflow as tf
from StyleGAN2 import StyleGAN

im_size = 256
latent_size = 512
BATCH_SIZE = 16
POLICY = "color,translation,cutout"
n_layers = int(log2(im_size) - 1)

def noise(n):
    return np.random.normal(0.0, 1.0, size = [n, latent_size]).astype('float32')

def noiseList(n):
    return [noise(n)] * n_layers

def nImage(n):
    return np.random.uniform(0.0, 1.0, size = [n, im_size, im_size, 1]).astype('float32')


if __name__ == "__main__":
    train_size = 100 # 100-shot-learning

    # get CIFAR10 data (add pandas another time?)
    datastr = "cifar10"
    (x_train, y_train), (x_test, y_test) = getattr(tf.keras.datasets, datastr).load_data()
    x_train = x_train[:train_size, ...]
    x_train = tf.image.resize(x_train, (im_size, im_size))

    # view one train image
    img = Image.fromarray(np.uint8(x_train[0, ...])).convert('RGB')
    display(img)

    # make dataset
    dataset = tf.data.Dataset.from_tensor_slices((x_train,)) \
        .repeat() \
        .batch(BATCH_SIZE) \
        .prefetch(tf.data.AUTOTUNE)

    with tf.device('/GPU:0'):
        dataset = dataset.map(lambda x: tf.cast(x, tf.float32))


    model = StyleGAN(dataset, lr=0.0001, silent=False)
    model.evaluate(0)

    if False:
        while model.GAN.steps < 1000001: # 1000001:
        # while model.GAN.images_seen < 300_000:
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

            model.save(1)
    
    # %%

    """
    from diffaugment.data import create_from_images

    datadir = "data/100-shot-panda"
    create_from_images(datadir, 32, channels_first=False)

    # load tfrecord dataset

    Commands:

        ```python
        model.load(19)

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