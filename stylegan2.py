# %%
from PIL import Image
from math import floor, log2
import numpy as np
import time
from functools import partial
from random import random
import psutil
import os

import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model, clone_model, model_from_json
from  tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import VarianceScaling
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.losses import BinaryCrossentropy

from datagen import printProgressBar
from conv_mod import *

# Alison additions
from diffaugment.augment import DiffAugment
import lambdas

im_size = 256 # 256
latent_size = 512
BATCH_SIZE = 16
POLICY = "color,translation,cutout"
# directory = "Earth"

cha = 24

n_layers = int(log2(im_size) - 1)

mixed_prob = 0.9


def T(tensor:tf.Tensor):
    """Chill DiffAugment wrapper."""
    return DiffAugment(tensor, policy=POLICY, channels_first=False)

def noise(n):
    return np.random.normal(0.0, 1.0, size = [n, latent_size]).astype('float32')

def noiseList(n):
    return [noise(n)] * n_layers

def mixedList(n):
    # Sample two noise lists and concat
    tt = int(random() * n_layers)
    p1 = [noise(n)] * tt
    p2 = [noise(n)] * (n_layers - tt)
    return p1 + [] + p2

def nImage(n):
    return np.random.uniform(0.0, 1.0, size = [n, im_size, im_size, 1]).astype('float32')


def gradient_penalty(samples, output, weight):
    gradients = K.gradients(output, samples)[0]
    gradients_sqr = K.square(gradients)
    gradient_penalty = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))

    # (weight / 2) * ||grad||^2
    # Penalize the gradient norm
    return K.mean(gradient_penalty) * weight * 0.5


#Blocks
def g_block(inp, istyle, inoise, fil, u=True):
    if u:
        #Custom upsampling because of clone_model issue
        out = lambdas.Upsample()(inp)
    else:
        out = layers.Activation('linear')(inp)
        
    rgb_style = layers.Dense(fil, kernel_initializer=VarianceScaling(200 / out.shape[2]))(istyle)
    style = layers.Dense(inp.shape[-1], kernel_initializer = 'he_uniform')(istyle)
    delta = lambdas.CropToFit()([inoise, out])
    d = layers.Dense(fil, kernel_initializer = 'zeros')(delta)

    out = Conv2DMod(filters=fil, kernel_size=3, padding='same', kernel_initializer='he_uniform')([out, style])
    out = layers.add([out, d])
    out = layers.LeakyReLU(0.2)(out)

    style = layers.Dense(fil, kernel_initializer = 'he_uniform')(istyle)
    d = layers.Dense(fil, kernel_initializer = 'zeros')(delta)

    out = Conv2DMod(filters = fil, kernel_size = 3, padding='same', kernel_initializer = 'he_uniform')([out, style])
    out = layers.add([out, d])
    out = layers.LeakyReLU(0.2)(out)

    rgb = to_rgb(out, rgb_style)
    return out, rgb


def d_block(inp, fil, p = True):
    res = layers.Conv2D(fil, 1, kernel_initializer = 'he_uniform')(inp)

    out = layers.Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_uniform')(inp)
    out = layers.LeakyReLU(0.2)(out)
    out = layers.Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_uniform')(out)
    out = layers.LeakyReLU(0.2)(out)

    out = layers.add([res, out])

    if p:
        out = layers.AveragePooling2D(pool_size=(2, 2))(out)

    return out


def to_rgb(inp, style):
    size = inp.shape[2]
    x = Conv2DMod(3, 1, kernel_initializer = VarianceScaling(200/size), demod = False)([inp, style])
    out = lambdas.UpsampleToSize(im_size)(x)
    return out


def from_rgb(inp, conc = None):
    fil = int(im_size * 4 / inp.shape[2])
    z = layers.AveragePooling2D(pool_size=(2, 2))(inp)
    x = layers.Conv2D(fil, 1, kernel_initializer = 'he_uniform')(z)
    if conc is not None:
        x = layers.concatenate([x, conc])
    return x, z



# %%

class GAN(object):

    def __init__(self, steps = 1, lr = 0.0001, decay = 0.00001):
        #Models
        self.D = None
        self.S = None
        self.G = None

        self.GE = None
        self.SE = None

        self.DM = None
        self.AM = None

        #Config
        self.LR = lr
        self.steps = steps
        self.images_seen = 0
        self.beta = 0.99

        #Init Models
        self.discriminator()
        self.generator()

        self.GMO = Adam(learning_rate=self.LR, beta_1=0, beta_2=0.99)
        self.DMO = Adam(learning_rate=self.LR, beta_1=0, beta_2=0.99)

        self.GE = clone_model(self.G)
        self.GE.set_weights(self.G.get_weights())

        self.SE = clone_model(self.S)
        self.SE.set_weights(self.S.get_weights())

    def discriminator(self):
        if self.D:
            return self.D

        inp = layers.Input(shape = [im_size, im_size, 3])
        x = d_block(inp, 1 * cha)   #128

        # Alison addition to allow smaller images
        im_sizes = np.array([4, 8, 16, 32, 64, 128, 256])
        filters = np.array([64, 32, 16, 8, 6, 4 , 2])
        assert im_size in im_sizes, "Invalid image size for network."
        filters = filters[im_sizes <= im_size]
        im_sizes = im_sizes[im_sizes <= im_size] # I think will have to change to numpy

        for depth in filters[::-1]:
            x = d_block(x, depth * cha)
        x = d_block(x, 32 * cha, p=False)  #4

        # x = d_block(x, 2 * cha)   #64
        # x = d_block(x, 4 * cha)   #32
        # x = d_block(x, 6 * cha)  #16
        # x = d_block(x, 8 * cha)  #8
        # x = d_block(x, 16 * cha)  #4
        # x = d_block(x, 32 * cha, p=False)  #4

        x = layers.Flatten()(x) # Todo: change to patchGAN?
        x = layers.Dense(1, kernel_initializer = 'he_uniform')(x)

        self.D = Model(inputs = inp, outputs = x)
        return self.D

    def generator(self):
        if self.G:
            return self.G

        # === Style Mapping ===
        self.S = Sequential()
        self.S.add(layers.Dense(512, input_shape = [latent_size]))
        self.S.add(layers.LeakyReLU(0.2))
        self.S.add(layers.Dense(512))
        self.S.add(layers.LeakyReLU(0.2))
        self.S.add(layers.Dense(512))
        self.S.add(layers.LeakyReLU(0.2))
        self.S.add(layers.Dense(512))
        self.S.add(layers.LeakyReLU(0.2))

        # === Generator ===
        #Inputs
        inp_style = []
        for i in range(n_layers):
            inp_style.append(layers.Input([512]))
        inp_noise = layers.Input([im_size, im_size, 1])

        #!Latent - causes problem loading from JSON 'lambda'
        x = lambdas.MakeOnes()(inp_style[0])

        outs = []

        #Actual Model
        x = layers.Dense(4*4*4*cha, activation = 'relu', kernel_initializer = 'random_normal')(x)
        x = layers.Reshape([4, 4, 4*cha])(x)

        x, r = g_block(x, inp_style[0], inp_noise, 32 * cha, u=False)  #4
        outs.append(r)

        # Alison addition to allow smaller images
        out_size = 0
        i = 1
        im_sizes = [8, 16, 32, 64, 128, 256]
        filters = [16, 8, 6, 4, 2, 1]
        assert im_size in im_sizes, "Invalid image size for network."
        for size, depth in zip(im_sizes, filters):
            if size <= im_size:
                x, r = g_block(x, inp_style[i], inp_noise, depth * cha)
                outs.append(r)
                i += 1

        x = layers.add(outs)
        #!Latent - causes problem loading from JSON 'lambda_12'
        x = lambdas.Center()(x) #Use values centered around 0, but normalize to [0, 1], providing better initialization
        self.G = Model(inputs = inp_style + [inp_noise], outputs = x)
        return self.G

    def GenModel(self):
        #Generator Model for Evaluation
        inp_style = []
        style = []

        for i in range(n_layers):
            inp_style.append(layers.Input([latent_size]))
            style.append(self.S(inp_style[-1]))

        inp_noise = layers.Input([im_size, im_size, 1])
        gf = self.G(style + [inp_noise])

        self.GM = Model(inputs = inp_style + [inp_noise], outputs = gf)
        return self.GM

    def GenModelA(self):
        #Parameter Averaged Generator Model

        inp_style = []
        style = []

        for i in range(n_layers):
            inp_style.append(layers.Input([latent_size]))
            style.append(self.SE(inp_style[-1]))

        inp_noise = layers.Input([im_size, im_size, 1])
        gf = self.GE(style + [inp_noise])

        self.GMA = Model(inputs = inp_style + [inp_noise], outputs = gf)
        return self.GMA

    def EMA(self):
        #Parameter Averaging
        for i in range(len(self.G.layers)):
            up_weight = self.G.layers[i].get_weights()
            old_weight = self.GE.layers[i].get_weights()
            new_weight = []
            for j in range(len(up_weight)):
                new_weight.append(old_weight[j] * self.beta + (1-self.beta) * up_weight[j])
            self.GE.layers[i].set_weights(new_weight)

        for i in range(len(self.S.layers)):
            up_weight = self.S.layers[i].get_weights()
            old_weight = self.SE.layers[i].get_weights()
            new_weight = []
            for j in range(len(up_weight)):
                new_weight.append(old_weight[j] * self.beta + (1-self.beta) * up_weight[j])
            self.SE.layers[i].set_weights(new_weight)

    def MAinit(self):
        #Reset Parameter Averaging
        self.GE.set_weights(self.G.get_weights())
        self.SE.set_weights(self.S.get_weights())



class StyleGAN(object):

    def __init__(self, dataset=None, steps=1, lr=0.0001, decay=0.00001, silent=True):
        #Init GAN and Eval Models
        self.GAN = GAN(steps = steps, lr = lr, decay = decay)
        self.GAN.GenModel()
        self.GAN.GenModelA()
        self.GAN.G.summary()

        #Data generator
        if dataset is not None:
            self.im = iter(dataset) # manually resets the dataset
        else:
            self.im = None

        #Set up variables
        self.lastblip = time.process_time() # replaced --A
        self.silent = silent
        self.ones = np.ones((BATCH_SIZE, 1), dtype=np.float32)
        self.zeros = np.zeros((BATCH_SIZE, 1), dtype=np.float32)
        self.nones = -self.ones
        self.pl_mean = 0
        self.av = np.zeros([44])

        # Losses
        self.bce = BinaryCrossentropy(from_logits=True)

    def train(self):
        #Train Alternating
        if random() < mixed_prob:
            style = mixedList(BATCH_SIZE)
        else:
            style = noiseList(BATCH_SIZE)

        #Apply penalties every 16 steps
        apply_gradient_penalty = self.GAN.steps % 2 == 0 or self.GAN.steps < 10000
        apply_path_penalty = self.GAN.steps % 16 == 0

        batch = self.im.next()
        a, b, c, d = self.train_step(batch, style, nImage(BATCH_SIZE), apply_gradient_penalty, apply_path_penalty)

        #Adjust path length penalty mean
        #d = pl_mean when no penalty is applied
        if self.pl_mean == 0:
            self.pl_mean = np.mean(d)
        self.pl_mean = 0.99*self.pl_mean + 0.01*np.mean(d)

        if self.GAN.steps % 10 == 0 and self.GAN.steps > 20000:
            self.GAN.EMA()

        if self.GAN.steps <= 25000 and self.GAN.steps % 1000 == 2:
            self.GAN.MAinit()

        if np.isnan(a):
            print("NaN Value Error.")
            exit()

        #Print info
        if self.GAN.steps % 100 == 0 and not self.silent:
            print("\n\nRound " + str(self.GAN.steps) + ":")
            print("D:", np.array(a))
            print("G:", np.array(b))
            print("PL:", self.pl_mean)

            s = round((time.process_time() - self.lastblip), 4)
            self.lastblip = time.process_time()

            steps_per_second = 100 / s
            steps_per_minute = steps_per_second * 60
            steps_per_hour = steps_per_minute * 60
            print("Steps/Second: " + str(round(steps_per_second, 2)))
            print("Steps/Hour: " + str(round(steps_per_hour)))

            min1k = floor(1000/steps_per_minute)
            sec1k = floor(1000/steps_per_second) % 60
            print("1k Steps: " + str(min1k) + ":" + str(sec1k))
            steps_left = 200000 - self.GAN.steps + 1e-7
            hours_left = steps_left // steps_per_hour
            minutes_left = (steps_left // steps_per_minute) % 60

            print("Til Completion: " + str(int(hours_left)) + "h" + str(int(minutes_left)) + "m")
            print()

            #Save Model
            if self.GAN.steps % 500 == 0:
                self.save(floor(self.GAN.steps / 10000))
            if self.GAN.steps % 1000 == 0 or (self.GAN.steps % 100 == 0 and self.GAN.steps < 2500):
                self.evaluate(floor(self.GAN.steps / 1000))

        # printProgressBar(self.GAN.steps % 100, 99, decimals = 0)
        # print("self.GAN.steps =", self.GAN.steps+1)
        # self.GAN.steps = self.GAN.steps + 1
        print("self.GAN.images_seen =", self.GAN.images_seen)
        self.GAN.images_seen += batch.shape[0] # Alison addition

    @tf.function
    def train_step(self, images, style, noise, perform_gp=True, perform_pl=False):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            #Get style information
            w_space = []
            pl_lengths = self.pl_mean
            for i in range(len(style)):
                w_space.append(self.GAN.S(style[i]))

            #Generate images
            generated_images = self.GAN.G(w_space + [noise])

            #Discriminate
            real_output = self.GAN.D(T(images), training=True)           # add diffaugment
            fake_output = self.GAN.D(T(generated_images), training=True) # add diffaugment

            #Loss functions
            # Todo: Try Spectral Normalisation of Discriminator (as alternative to GP)
            gen_loss = self.bce(tf.ones_like(fake_output), fake_output) # Logistic NS
            real_disc_loss = self.bce(tf.ones_like(real_output), real_output)
            fake_disc_loss = self.bce(tf.zeros_like(fake_output), fake_output)
            divergence = real_disc_loss + fake_disc_loss # -log(1-sigmoid(fake_scores_out)) -log(sigmoid(real_scores_out))
            disc_loss = divergence

            if perform_gp:
                #R1 gradient penalty
                disc_loss += gradient_penalty(images, real_output, 10)

            if perform_pl:
                #Slightly adjust W space
                w_space_2 = []
                for i in range(len(style)):
                    std = 0.1 / (K.std(w_space[i], axis = 0, keepdims = True) + 1e-8)
                    w_space_2.append(w_space[i] + K.random_normal(tf.shape(w_space[i])) / (std + 1e-8))

                #Generate from slightly adjusted W space
                pl_images = self.GAN.G(w_space_2 + [noise])

                #Get distance after adjustment (path length)
                delta_g = K.mean(K.square(pl_images - generated_images), axis = [1, 2, 3])
                pl_lengths = delta_g

                if self.pl_mean > 0:
                    gen_loss += K.mean(K.square(pl_lengths - self.pl_mean))

        #Get gradients for respective areas
        gradients_of_generator = gen_tape.gradient(gen_loss, self.GAN.GM.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.GAN.D.trainable_variables)

        #Apply gradients
        self.GAN.GMO.apply_gradients(zip(gradients_of_generator, self.GAN.GM.trainable_variables))
        self.GAN.DMO.apply_gradients(zip(gradients_of_discriminator, self.GAN.D.trainable_variables))

        return disc_loss, gen_loss, divergence, pl_lengths

    def evaluate(self, num=0, trunc=1.0):
        n1 = noiseList(64)
        n2 = nImage(64)
        trunc = np.ones([64, 1]) * trunc
        generated_images = self.GAN.GM.predict(n1 + [n2], batch_size=BATCH_SIZE)

        r = []
        for i in range(0, 64, 8):
            r.append(np.concatenate(generated_images[i:i+8], axis = 1))

        c1 = np.concatenate(r, axis = 0)
        c1 = np.clip(c1, 0.0, 1.0)
        x = Image.fromarray(np.uint8(c1*255))
        x.save("Results/i"+str(num)+".png")

        # Moving Average
        # generated_images = self.GAN.GMA.predict(n1 + [n2, trunc], batch_size=BATCH_SIZE) # Alison removed
        generated_images = self.GAN.GMA.predict(n1 + [n2], batch_size=BATCH_SIZE)
        #generated_images = self.generateTruncated(n1, trunc = trunc)

        r = []
        for i in range(0, 64, 8):
            r.append(np.concatenate(generated_images[i:i+8], axis = 1))

        c1 = np.concatenate(r, axis = 0)
        c1 = np.clip(c1, 0.0, 1.0)
        x = Image.fromarray(np.uint8(c1*255))
        x.save("Results/i"+str(num)+"-ema.png")

        #Mixing Regularities
        nn = noise(8)
        n1 = np.tile(nn, (8, 1))
        n2 = np.repeat(nn, 8, axis = 0)
        tt = int(n_layers / 2)

        p1 = [n1] * tt
        p2 = [n2] * (n_layers - tt)

        latent = p1 + [] + p2
        # generated_images = self.GAN.GMA.predict(latent + [nImage(64), trunc], batch_size=BATCH_SIZE) # Alison removed
        generated_images = self.GAN.GMA.predict(latent + [nImage(64)], batch_size=BATCH_SIZE)
        #generated_images = self.generateTruncated(latent, trunc = trunc)

        r = []
        for i in range(0, 64, 8):
            r.append(np.concatenate(generated_images[i:i+8], axis = 0))

        c1 = np.concatenate(r, axis = 1)
        c1 = np.clip(c1, 0.0, 1.0)
        x = Image.fromarray(np.uint8(c1*255))
        x.save("Results/i"+str(num)+"-mr.png")


    def generateTruncated(self, style, noi = np.zeros([44]), trunc = 0.5, outImage = False, num = 0):
        #Get W's center of mass
        if self.av.shape[0] == 44: #44 is an arbitrary value
            print("Approximating W center of mass")
            self.av = np.mean(self.GAN.S.predict(noise(2000), batch_size = 64), axis = 0)
            self.av = np.expand_dims(self.av, axis = 0)

        if noi.shape[0] == 44:
            noi = nImage(64)

        w_space = []
        for i in range(len(style)):
            tempStyle = self.GAN.S.predict(style[i])
            tempStyle = trunc * (tempStyle - self.av) + self.av
            w_space.append(tempStyle)

        generated_images = self.GAN.GE.predict(w_space + [noi], batch_size = BATCH_SIZE)

        if outImage:
            r = []
            for i in range(0, 64, 8):
                r.append(np.concatenate(generated_images[i:i+8], axis = 0))

            c1 = np.concatenate(r, axis = 1)
            c1 = np.clip(c1, 0.0, 1.0)
            x = Image.fromarray(np.uint8(c1*255))
            x.save("Results/t"+str(num)+".png")
        return generated_images

    def saveModel(self, model, name, num):
        json = model.to_json()
        with open("Models/"+name+".json", "w") as json_file:
            json_file.write(json)
        model.save_weights("Models/"+name+"_"+str(num)+".h5")

    def loadModel(self, name, num):
        with open("Models/"+name+".json", 'r') as file:
            json = file.read()

        mod = model_from_json(json,
                              custom_objects={
                                  'Conv2DMod': Conv2DMod,
                                  'Upsample': lambdas.Upsample,
                                  'UpsampleToSize': lambdas.UpsampleToSize,
                                  'MakeOnes': lambdas.MakeOnes,
                                  'CropToFit': lambdas.CropToFit,
                                  'Center': lambdas.Center
                                  }
                              )
        mod.load_weights("Models/"+name+"_"+str(num)+".h5")
        return mod

    def save(self, num): #Save JSON and Weights into /Models/
        self.saveModel(self.GAN.S, "sty", num)
        self.saveModel(self.GAN.G, "gen", num)
        self.saveModel(self.GAN.D, "dis", num)
        self.saveModel(self.GAN.GE, "genMA", num)
        self.saveModel(self.GAN.SE, "styMA", num)


    def load(self, num): #Load JSON and Weights from /Models/
        #Load Models
        self.GAN.D = self.loadModel("dis", num)
        self.GAN.S = self.loadModel("sty", num)
        self.GAN.G = self.loadModel("gen", num)
        self.GAN.GE = self.loadModel("genMA", num)
        self.GAN.SE = self.loadModel("styMA", num)
        self.GAN.GenModel()
        self.GAN.GenModelA()


# %%
if __name__ == "__main__":
    train_size = 100 # 100-shot-learning
    batch_size = 16

    # get CIFAR10 data (add pandas another time?)
    datastr = "cifar10"
    (x_train, y_train), (x_test, y_test) = getattr(tf.keras.datasets, datastr).load_data()
    x_train = x_train[:train_size, ...]
    x_train = tf.image.resize(x_train, (im_size, im_size))

    # make dataset
    dataset = tf.data.Dataset.from_tensor_slices((x_train,)) \
        .repeat() \
        .batch(batch_size) \
        .prefetch(tf.data.AUTOTUNE)

    with tf.device('/GPU:0'):
        dataset = dataset.map(lambda x: tf.cast(x, tf.float32))

    model = StyleGAN(dataset, lr=0.0001, silent=False)
    model.evaluate(0)
    
    if False: # training
        while model.GAN.steps < 1000001:
            try:
                model.train()
            except Exception as e:
                print(e)

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