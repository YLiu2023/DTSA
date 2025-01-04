#set random seed
#GPU 0: Tesla V100-SXM2-16GB
import os
import random
import tensorflow as tf
import numpy as np
#seed:5,10,20,31,40,50,60,70,104,567,
      # 784,871,1002,1945,4000,3912,5678,40532,78,90
      # 43,56,563,25,34,64,66,18,27,19
#check version
print('numpy',np.__version__)
print('tensorflow',tf.__version__)
def set_seed(seed=31):

  os.environ['PYTHONHASHSEED']=str(0)
  random.seed(seed)
  tf.random.set_seed(seed)
  tf.keras.utils.set_random_seed(seed)
  tf.compat.v1.set_random_seed(seed)
  np.random.seed(seed)
  os.environ['TF_DETERMINISTIC_OPS'] = '1'

seed='66'
set_seed(int(seed))
print(int(seed))

# self define
#load data
def LoadData_pickle(path,name,type='rb'):
  with open(path+name+'.pkl', type) as f:
          data=pickle.load(f)

  return data
def LoadData_pickle_T(path,name,type='rb'):
  with open(path+name+'_test.pkl', type) as f:
          data,label=pickle.load(f)

  return data,label
def one_hot_MPT(y,depth):
# #one-hot
  y=tf.one_hot(y,depth=depth+1,dtype=tf.int32)
  # y=y[:,1:]# 前面一列删除

  return y
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
import pandas as pd
# visiualize
def scatter(x, colors, n_class):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", n_class))

    # We create a scatter plot.
    f = plt.figure(figsize=(10, 10))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40,
                    c=palette[colors.astype(np.int32)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    # plt.grid(c='r')
    ax.axis('on')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(n_class):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts
def t_sne(x, y, n_class, savename):
    # import warnings filter
    from warnings import simplefilter
    # ignore all future warnings
    simplefilter(action='ignore', category=FutureWarning)

    sns.set_style('whitegrid')
    sns.set_palette('muted')
    sns.set_context("notebook", font_scale=1.5,
                    rc={"lines.linewidth": 2.5})
    from sklearn.manifold import TSNE
    digits_proj = TSNE(random_state=128).fit_transform(x)

    scatter(digits_proj, y, n_class)

    plt.savefig(savename)
import os
import tensorflow as tf
os.makedirs("visual", exist_ok=True)
def save_gan(model, ep, **kwargs):
  #test set
  #6->0,9->fault
    name = model.__class__.__name__.lower()
    if name == "cyclegan":
        if "img6" not in kwargs or "img9" not in kwargs:

            raise ValueError
        img6, img9 = kwargs["img6"], kwargs["img9"]#5*2
        y6, y9 = kwargs["y6"], kwargs["y9"]#5*2
        img9_, img6_ = model.g12.call([img6,y9], training=False), model.g21.call([img9,y6], training=False)

        x_eval=np.vstack((img9_.numpy(),img9.numpy()))
        y_eval=np.array([0]*50+[1]*50+[2]*50+[3]*50+[4]*50+[5]*50+[6]*50+[7]*50+[8]*50+[9]*50)

        #MSE metric
        score6=tf.reduce_mean(tf.losses.mean_squared_error(img6,img6_))
        score9=tf.reduce_mean(tf.losses.mean_squared_error(img9,img9_))
        score=score6+score9


    else:
        raise ValueError(name)
    plt.clf()
    plt.close()
    return score.numpy(),x_eval,y_eval
# SWD TF2 version
def sw_loss(true_distribution,
            generated_distribution,
            num_projections, batch_size):
    s = true_distribution.get_shape().as_list()[-1]

    # num_projections=140
    # batch_size=140
    theta = tf.random.normal(shape=[s, num_projections])
    theta = tf.nn.l2_normalize(theta, axis=0)

    # project the samples (images). After being transposed, we have tensors
    # of the format: [projected_image1, projected_image2, ...].
    # Each row has the projections along one direction. This makes it
    # easier for the sorting that follows.
    projected_true = tf.transpose(
        tf.matmul(true_distribution, theta))

    projected_fake = tf.transpose(
        tf.matmul(generated_distribution, theta))

    sorted_true, true_indices = tf.nn.top_k(
        projected_true,
        batch_size)

    sorted_fake, fake_indices = tf.nn.top_k(
        projected_fake,
        batch_size)
    # print(sorted_fake.shape, fake_indices.shape)

    # For faster gradient computation, we do not use sorted_fake to compute
    # loss. Instead we re-order the sorted_true so that the samples from the
    # true distribution go to the correct sample from the fake distribution.
    # This is because Tensorflow did not have a GPU op for rearranging the
    # gradients at the time of writing this code.

    # It is less expensive (memory-wise) to rearrange arrays in TF.
    # Flatten the sorted_true from [batch_size, num_projections].
    flat_true = tf.reshape(sorted_true, [-1])

    # Modify the indices to reflect this transition to an array.
    # new index = row + index
    rows = np.asarray(
        [batch_size * np.floor(i * 1.0 / batch_size)
         for i in range(num_projections * batch_size)])
    rows = rows.astype(np.int32)
    flat_idx = tf.reshape(fake_indices, [-1, 1]) + np.reshape(rows, [-1, 1])

    # The scatter operation takes care of reshaping to the rearranged matrix
    shape = tf.constant([batch_size * num_projections])
    rearranged_true = tf.reshape(
        tf.scatter_nd(flat_idx, flat_true, shape),
        [num_projections, batch_size])

    return tf.reduce_mean(tf.square(projected_fake - rearranged_true))
from tensorflow import keras
class InstanceNormalization(keras.layers.Layer):
    def __init__(self, axis=(1, 2), epsilon=1e-6):
        super().__init__()
        # NHWC
        self.epsilon = epsilon
        self.axis = axis
        self.beta, self.gamma = None, None

    def build(self, input_shape):
        # NHWC
        shape = [1,  1, input_shape[-1]]
        self.gamma = self.add_weight(
            name='gamma',
            shape=shape,
            initializer='ones')

        self.beta = self.add_weight(
            name='beta',
            shape=shape,
            initializer='zeros')

    def call(self, x, *args, **kwargs):
        mean = tf.math.reduce_mean(x, axis=self.axis, keepdims=True)
        diff = x - mean
        variance = tf.reduce_mean(tf.math.square(diff), axis=self.axis, keepdims=True)
        x_norm = diff * tf.math.rsqrt(variance + self.epsilon)
        return x_norm * self.gamma + self.beta

#SEM
from tensorflow.keras.layers import GlobalAveragePooling1D, Reshape, Dense, Input,LeakyReLU
from keras import backend as K
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, add, Permute, Conv2D
def SqueezeAndExcitation(inputs, ratio=8):
    b,_, c = inputs.shape

    x = GlobalAveragePooling1D()(inputs)
    x = Dense(c//ratio, activation="relu",kernel_initializer='he_normal', use_bias=False)(x)
    x = Dense(c, activation="sigmoid", kernel_initializer='he_normal',use_bias=False)(x)

    x = multiply([inputs, x])
    return x


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, Activation, Flatten, Dense, Concatenate, Input, \
    Reshape, ReLU, Conv1D, Conv1DTranspose, Add, BatchNormalization, Embedding, GaussianNoise
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
import time
from sklearn.preprocessing import scale
_leaky_relu = lambda net: tf.nn.leaky_relu(net, alpha=0.3)
def do_norm(norm):
    if norm == "batch":
        _norm = BatchNormalization()
    elif norm == "instance":
        _norm = InstanceNormalization()
    else:
        _norm = []
    return _norm
def gen_block_down(filters, k_size, strides, padding, input, names=None, norm="instance"):
    init = RandomNormal(stddev=0.02)

    g = Conv1D(filters, k_size, strides=strides, kernel_initializer=init, padding=padding, name=names)(input)
    g = do_norm(norm)(g)
    g = ReLU()(g)
    return g
def gen_block_down_dis(filters, k_size, strides, padding, input, norm="instance"):
    init = RandomNormal(stddev=0.02)

    g = Conv1D(filters, k_size, strides=strides, kernel_initializer=init, padding=padding, )(input)
    g = do_norm(norm)(g)
    g = LeakyReLU()(g)
    return g
def gen_block_up(filters, k_size, strides, padding, input, names=None, norm="instance"):
    init = RandomNormal(stddev=0.02)
    g = Conv1DTranspose(filters, k_size, strides=strides, kernel_initializer=init, padding=padding, name=names)(input)
    g = do_norm(norm)(g)
    g = ReLU()(g)
    return g
# generate a resnet block
def resnet_block(n_filters, input_layer):
    # weight initialization
    init = RandomNormal(stddev=0.02)

    # first layer convolutional layer
    g = Conv1D(n_filters, 5, padding='same', kernel_initializer=init)(input_layer)
    g = InstanceNormalization(axis=-1)(g)
    g = ReLU()(g)

    # second convolutional layer
    g = Conv1D(n_filters, 5, padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)

    # concatenate merge channel-wise with input layer
    g = Add()([g, input_layer])
    # g = ReLU()(g)
    return g
def resnet_block_SENet(n_filters, input_layer):
    # set_seed(784)
    # weight initialization
    init = RandomNormal(stddev=0.02)

    # residual
    # first layer convolutional layer
    g = Conv1D(n_filters, 5, padding='same', kernel_initializer=init)(input_layer)
    g = InstanceNormalization(axis=-1)(g)
    g = ReLU()(g)

    # second convolutional layer
    g = Conv1D(n_filters, 5, padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)

    # sem
    x_se = SqueezeAndExcitation(g)
    x_se = InstanceNormalization(axis=-1)(x_se)

    # concatenate merge channel-wise with input layer
    g = Add()([x_se, input_layer])

    return g
from tensorflow.python.ops.gen_array_ops import gather_eager_fallback
# build CycleGAN
class CycleGAN(keras.Model):
    def __init__(self, lambda_, img_shape, use_identity=False) -> object:
        super().__init__()
        self.lambda_ = lambda_
        self.img_shape = img_shape
        self.use_identity = use_identity

        self.g12 = self._get_generator("g12")
        self.g21 = self._get_generator("g21")

        self.d12 = self._get_discriminator("d12")
        self.d21 = self._get_discriminator("d21")

        self.d12.summary()
        self.g12.summary()

        self.opt_G = keras.optimizers.Adam(1e-4, beta_1=0.5)
        self.opt_D = keras.optimizers.Adam(2e-5, beta_1=0.5)
        # self.loss_bool = keras.losses.BinaryCrossentropy(from_logits=True)
        self.loss_img = keras.losses.MeanAbsoluteError()  # a better result when using mse
        self.loss_identity = tf.losses.MeanAbsoluteError()

    def d_loss_wasserstein(self, real_logits, fake_logits):

        d_loss = tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)

        return d_loss

    def g_loss_wasserstein(self, fake_logits):

        g_loss = -tf.reduce_mean(fake_logits)

        return g_loss

    def wasserstein_gradient_penalty(self, x, x_fake, y, discriminator):

        # temp_shape = [x.shape[0]]+[1 for _ in  range(len(x.shape)-1)]

        epsilon = tf.random.uniform([], 0.0, 1.0)
        x_hat = epsilon * x + (1 - epsilon) * x_fake

        with tf.GradientTape() as t:
            t.watch(x_hat)
            d_hat = discriminator([x_hat, y], training=False)
        gradients = t.gradient(d_hat, x_hat)
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients)))
        gradient_penalty = 1 * tf.reduce_mean((slopes - 1.0) ** 2)

        return gradient_penalty

    def _get_generator(self, name):

        # data input
        in_image = Input(shape=self.img_shape)  # (1024)
        x = Reshape((1024, 1))(in_image)  # (1024,1)

        # label input
        label = Input(shape=(), dtype=tf.float32)  # <----input
        label_emb = Embedding(6, 64)(label)  # first position is for user defined,second position is used for bath size
        emb_img = Reshape((1024, 1))(Dense(1024 * 1, activation=keras.activations.relu)(label_emb))
        u = tf.concat((x, emb_img), axis=2)

        # Encoder
        enc = gen_block_down(64, 5, 2, 'same', u)  # (512,64)
        enc = gen_block_down(128, 5, 2, 'same', enc)  # (256,128)
        enc = gen_block_down(64, 5, 2, 'same', enc)  # (128,64)

        # resnet
        enc = resnet_block(64, enc)  # (none,128,64)
        enc = resnet_block(64, enc)  # (none,128,64)
        enc = resnet_block(64, enc)  # (none,128,64)
        enc = resnet_block(64, enc)  # (none,128,64)

        # Decoder
        dec = gen_block_up(32, 5, 2, 'same', enc)  # (256,128)
        dec = gen_block_up(16, 5, 2, 'same', dec)  # (512,64)
        # dec=gen_block_up(1,5,2,'same',dec)# (1024,32)
        dec = Conv1DTranspose(1, 5, 2, kernel_initializer=RandomNormal(), padding='same')(dec)  # (1024,1)
        dec = tf.keras.activations.tanh(dec)
        out_img = Flatten(name='pzq3')(dec)  # (1024)

        model = keras.models.Model(inputs=[in_image, label], outputs=out_img, name=name)
        # model.summary()

        return model

    def _get_discriminator(self, name):

        # input image
        in_image = Input(shape=self.img_shape)  # (1024)
        g = GaussianNoise(0.01)(in_image)
        # noise=tf.random.normal(shape=tf.shape(in_image),mean=0.0,stddev=1.0,dtype=tf.float32)
        # in_image+=noise
        x = keras.layers.Reshape((1024, 1))(g)  # (1024,1)

        # label
        label = Input(shape=(), dtype=tf.float32)  # <----input
        label_emb = Embedding(6, 64)(label)  # first position is for user defined,second position is used for bath size
        emb_img = Reshape((1024, 1))(Dense(1024 * 1, activation=keras.activations.relu)(label_emb))
        u = tf.concat((x, emb_img), axis=2)

        x = gen_block_down_dis(64, 5, 2, 'same', u)  # (512,32)
        x = gen_block_down_dis(128, 5, 2, 'same', x)  # (256,64)
        x = gen_block_down_dis(256, 5, 2, 'same', x)  # (128,128)
        x = gen_block_down_dis(512, 5, 2, 'same', x)  # (64,256)
        x = gen_block_down_dis(1024, 5, 2, 'same', x)  # (32,512)
        # x=gen_block_down_dis(512,5,2,'same',x) #(16,1024)
        # x=gen_block_down_dis(256,5,2,'same',x) #(8,512)

        x = gen_block_down_dis(1, 5, 2, 'same', x)  # (4,1)
        x = Flatten()(x)  # (1)
        logits = keras.layers.Dense(1, activation=None)(x)

        model = keras.models.Model(inputs=[in_image, label], outputs=logits, name=name)
        # model.summary()
        return model

    def cycle_loss(self, real_img1, real_y1, real_img2, real_y2):
        # cycle_loss

        fake2, fake1 = self.g12([real_img1, real_y2]), self.g21([real_img2, real_y1])

        loss1 = self.loss_img(real_img1, self.g21([fake2, real_y1]))
        loss2 = self.loss_img(real_img2, self.g12([fake1, real_y2]))
        cycle_loss = loss1 + loss2

        # identity_loss
        fake1_id, fake2_id = self.g21([real_img1, real_y1]), self.g12([real_img2, real_y2])
        loss1_id = self.loss_identity(real_img1, fake1_id)
        loss2_id = self.loss_identity(real_img2, fake2_id)
        identity_loss = loss1_id + loss2_id

        return cycle_loss, fake2, fake1, identity_loss

    def train_g(self, img1, y1, img2, y2, loss_type):
        with tf.GradientTape() as tape:
            cycle_loss, fake2, fake1, identity_loss = self.cycle_loss(img1, y1, img2, y2)

            # img1
            pred2 = self.d12([fake2, y2])

            # img2
            pred1 = self.d21([fake1, y1])

            if loss_type == 'wd':
                # wd
                d_loss12 = self.g_loss_wasserstein(pred2)

                # wd
                d_loss21 = self.g_loss_wasserstein(pred1)

            else:
                # swd
                num = 32
                r_y2 = self.d12([img2, y2])
                d_loss12 = sw_loss(r_y2, pred2, num_projections=num, batch_size=num)

                # swd
                r_y1 = self.d21([img1, y1])
                d_loss21 = sw_loss(r_y1, pred1, num_projections=num, batch_size=num)

            loss12 = d_loss12 + 5 * cycle_loss + 5 * identity_loss
            loss21 = d_loss21 + 5 * cycle_loss + 5 * identity_loss

            loss = loss12 + loss21
        var = self.g12.trainable_variables + self.g21.trainable_variables
        grads = tape.gradient(loss, var)
        self.opt_G.apply_gradients(zip(grads, var))

        return d_loss12 + d_loss21, cycle_loss

    def train_d(self, img1, y1_, img2, y2_):
        length = len(img1)  # length of img1=length of img2

        with tf.GradientTape() as d_tape:
            fake2, fake1 = self.g12([img1, y2_]), self.g21([img2, y1_])
            # y_real = tf.ones((length, 1), tf.float32)
            # y_fake = tf.zeros((length, 1), tf.float32)

            # adversarial_1

            y2 = self.d12([img2, y2_])
            pred2 = self.d12([fake2, y2_])
            # loss2_real = self.loss_bool(y_real, y2)
            # loss2_fake = self.loss_bool(y_fake, pred2)

            # loss_12 = loss2_real + loss2_fake

            loss_12 = self.d_loss_wasserstein(y2, pred2)
            loss_12 += self.wasserstein_gradient_penalty(x=img2, x_fake=fake2, y=y2_,
                                                         discriminator=self.d12)

            # adversarial_2
            y1 = self.d21([img1, y1_])
            pred1 = self.d21([fake1, y1_])

            # loss1_real = self.loss_bool(y_real, y1)
            # loss1_fake = self.loss_bool(y_fake, pred1)

            # loss_21 = loss1_real + loss1_fake
            loss_21 = self.d_loss_wasserstein(y1, pred1)
            loss_21 += self.wasserstein_gradient_penalty(x=img1, x_fake=fake1, y=y1_,
                                                         discriminator=self.d21)

            # total adversarial loss
            dis_loss = loss_12 + loss_21

        var = self.d12.trainable_variables + self.d21.trainable_variables
        dis_grads = d_tape.gradient(dis_loss, var)
        self.opt_D.apply_gradients(zip(dis_grads, var))
        return dis_loss

    def train_on_step(self, img1, y1, img2, y2, loss_type):

        # for _ in range(3):
        d_loss = self.train_d(img1, y1, img2, y2)
        # for _ in range(3):
        g_loss, cyc_loss = self.train_g(img1, y1, img2, y2, loss_type)
        return g_loss, d_loss, cyc_loss
def train(seed, loss_type, gan, x0, y0, ds_x, ds_y, test6, testy_6, test9, testy_9, step, batch_size):
    loss_G = []
    loss_D = []
    loss_CYC = []
    s_value = []
    r_value = []

    general = '../Results/robot_condi/' + loss_type + '/' + seed

    dir_ = general + '/visual/'
    dir_loss = general + '/loss/'
    dir_model = general + '/models/'
    dir_utils = general + '/others/'

    os.makedirs(dir_, exist_ok=True)
    os.makedirs(dir_loss, exist_ok=True)
    os.makedirs(dir_utils, exist_ok=True)
    t0 = time.time()
    rate = 0

    for t in range(step):
        idx6 = np.random.randint(0, len(x0), batch_size)
        img6 = tf.cast(tf.gather(x0, idx6), dtype=tf.float32)
        y6 = tf.cast(tf.gather(y0, idx6), dtype=tf.float32)

        idx9 = np.random.randint(0, len(ds_x), batch_size)
        img9 = tf.cast(tf.gather(ds_x, idx9), dtype=tf.float32)
        y9 = tf.cast(tf.gather(ds_y, idx9), dtype=tf.float32)

        g_loss, d_loss, cyc_loss = gan.train_on_step(img6, y6, img9, y9, loss_type)

        current_score, x_eval, y_eval = save_gan(gan, t,
                                                 img6=tf.convert_to_tensor(test6, tf.float32),
                                                 img9=tf.convert_to_tensor(test9, tf.float32),
                                                 y6=tf.convert_to_tensor(testy_6, tf.float32),
                                                 y9=tf.convert_to_tensor(testy_9, tf.float32))

        loss_G.append(g_loss.numpy())
        loss_D.append(d_loss.numpy())
        loss_CYC.append(cyc_loss.numpy())

        if t == 0:
            score0 = current_score
            print('initial:', score0)

            s_value.append([score0, t])
            ty = 0
            r_value.append([ty, t])

        else:

            score = current_score
            s_value.append([score, t])
            current_rate = 1 - score / score0

            if current_rate > rate:
                r_value.append([current_rate, t])
                print('Rate：', current_rate, 'at step', t, 'Cost: gen', g_loss.numpy(), 'dis', d_loss.numpy(), 'cyc',
                      cyc_loss.numpy())
                rate = current_rate
                best_step = t

                # visual
                path = dir_ + "{}.png".format(best_step)
                t_sne(x_eval, y_eval, n_class=10, savename=path)

                # models
                os.makedirs(dir_model + 'model_' + str(t), exist_ok=True)
                gan.save_weights(dir_model + 'model_' + str(t) + '/model.ckpt')

    # last iteration
    # print('running time:', t1 - t0)

    # visual
    path = dir_ + "/{}.png".format(t)
    t_sne(x_eval, y_eval, n_class=10, savename=path)

    # save weights
    # save_weights(gan,t)
    gan.save_weights(dir_model + 'model_' + str(t) + '/model.ckpt')

    loss_G.append(g_loss.numpy())
    loss_D.append(d_loss.numpy())
    loss_CYC.append(cyc_loss.numpy())

    t1 = time.time()
    print('running time:', t1 - t0)

    # loss
    np.savetxt(dir_loss + 'loss_g.txt', np.array(loss_G))
    np.savetxt(dir_loss + 'loss_d.txt', np.array(loss_D))
    np.savetxt(dir_loss + 'loss_cyc.txt', np.array(loss_CYC))

    # others
    np.savetxt(dir_utils + 'score.txt', np.array(s_value))
    np.savetxt(dir_utils + 'rate.txt', np.array(r_value))

import pickle
from sklearn import preprocessing
#Load source domain datasets
#------------------domainA------------------------
fault='C0S3L0_all'

x0=LoadData_pickle(path='../dataset/original/',
                    name=fault)[5]

x_A=x0[0:500,:]
y_A=np.array([0]*500)
x_A_T=x0[500:750,:]
y_A_T=np.array([0]*250)
print('Domain A train',x_A.shape,y_A.shape)
print('Domain A test',x_A_T.shape,y_A_T.shape)
#------------------domainB---------------------------------
print()
x1=LoadData_pickle(path='../dataset/original/',
                        name='C3S3L0_all')[5]
x_t1=x1[0:100,:]
x_T1=x1[100:150,:]

x2=LoadData_pickle(path='../dataset/original/',
                        name='C4S3L0_all')[5]
x_t2=x2[0:100,:]
x_T2=x2[100:150,:]

x3=LoadData_pickle(path='../dataset/original/',
                        name='C6S3L0_all')[5]
x_t3=x3[0:100,:]
x_T3=x3[100:150,:]

x7=LoadData_pickle(path='../dataset/original/',
                        name='C7S3L0_all')[5]
x_t7=x7[0:100,:]
x_T7=x7[100:150,:]

x8=LoadData_pickle(path='../dataset/original/',
                        name='C8S3L0_all')[5]
x_t8=x8[0:100,:]
x_T8=x8[100:150,:]
x_B=np.vstack((x_t1,x_t2,x_t3,x_t7,x_t8))
x_B_T=np.vstack((x_T1,x_T2,x_T3,x_T7,x_T8))

y_B=np.array([1]*100+[2]*100+[3]*100+[4]*100+[5]*100)
y_B_T=np.array([1]*50+[2]*50+[3]*50+[4]*50+[5]*50)
print('Domain B train',x_B.shape,y_B.shape)
print('Domain B test',x_B_T.shape,y_B_T.shape)

#5, run
IMG_SHAPE=(1024,)
LAMBDA=10
BATCH_SIZE=32
n_epochs=1200
bat_per_step=int(len(x_A)/BATCH_SIZE)
n_steps=bat_per_step*n_epochs
print('n_steps:',n_steps)
cyclegan: CycleGAN=CycleGAN(lambda_=LAMBDA, img_shape=IMG_SHAPE)

# train(seed,'swd',cyclegan,x_A,y_A,x_B,y_B,
#        x_A_T,y_A_T,x_B_T,y_B_T,
#       n_steps,BATCH_SIZE)


# 载入模型
dir='../Results/'
checkpoint_dir=dir+'model_17999'+'/model.ckpt'
print(checkpoint_dir)
cyclegan.load_weights(checkpoint_dir)
print('Load successfully!!!')


# 用x0生成数据C1、C2、C3、C4、C5
x0=x0
print('C0.shape:',x0.shape)
x_A=x0[0:1000,:]
y_B=np.array([1]*1000)
xA=tf.convert_to_tensor(x_A, tf.float32)
yB=tf.convert_to_tensor(y_B, tf.float32)
imgB_1 = cyclegan.g12.call([xA,yB], training=False)
C1=imgB_1.numpy()
print('C1.shape:',C1.shape)

with open('../dataset/generated/gen_data_C_1.pkl','wb') as f:
    pickle.dump(C1,f,protocol=4)


