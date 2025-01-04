import tensorflow as tf
from tensorflow.keras import layers, models
# import pickle5 as pickle
import pickle
import matplotlib.pyplot as plt
from pylab import xticks
from tensorflow.keras.initializers import RandomNormal
_leaky_relu = lambda net: tf.nn.leaky_relu(net, alpha=0.3)
import numpy as np
from scipy import linalg
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array, as_float_array
# from tensorflow_model_remediation.min_diff.losses import MMDLoss
from sklearn.preprocessing import normalize
#SWD TF2 version
import tensorflow as tf
def do_fft_norm(x):
    xx0_fft=np.abs(np.fft.fft(x))*2/len(x)
    xx0_fft=xx0_fft[:len(x)]
    return xx0_fft

def freq_Analysis(x):
    x-=np.mean(x)
    x=do_fft_norm(x)
    return x
def sw_loss(true_distribution, generated_distribution,num_projections,batch_size):

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

def mmd_loss(features_source, features_target, kernel=True):
    # 计算源域和目标域的统计数据
    source_kernel_sum = tf.reduce_sum(features_source, axis=0)
    target_kernel_sum = tf.reduce_sum(features_target, axis=0)
    source_kernel_square_sum = tf.reduce_sum(tf.square(features_source), axis=0)
    target_kernel_square_sum = tf.reduce_sum(tf.square(features_target), axis=0)
    source_kernel_inner_product = tf.reduce_sum(features_source * features_source, axis=0)
    target_kernel_inner_product = tf.reduce_sum(features_target * features_target, axis=0)

    if kernel:
        # 使用核函数进行MMD估计
        # 这里使用了RBF核函数（高斯核）
        kernel_matrix = tf.exp(-1 * (tf.square(source_kernel_sum) + tf.square(
            target_kernel_sum) - 2 * source_kernel_inner_product - 2 * target_kernel_inner_product))
        kernel_matrix /= tf.sqrt(tf.multiply(tf.cast(tf.shape(features_source)[0], tf.float32),
                                             tf.cast(tf.shape(features_target)[0], tf.float32)))
    else:
        # 不使用核函数，直接计算MMD
        kernel_matrix = (source_kernel_square_sum - source_kernel_sum ** 2) / tf.cast(tf.shape(features_source)[0],
                                                                                      tf.float32) - (
                                    target_kernel_square_sum - target_kernel_sum ** 2) / tf.cast(
            tf.shape(features_target)[0], tf.float32)

    # 计算MMD损失
    mmd_loss = tf.reduce_mean(kernel_matrix)
    return mmd_loss
class ZCA(BaseEstimator, TransformerMixin):
    def __init__(self, regularization=1e-6, copy=False):
        self.regularization = regularization
        self.copy = copy

    def fit(self, X, y=None):
        """Compute the mean, whitening and dewhitening matrices.
        Parameters
        ----------
        X : array-like with shape [n_samples, n_features]
            The data used to compute the mean, whitening and dewhitening
            matrices.
        """
        X = check_array(X, accept_sparse=None, copy=self.copy,
                        ensure_2d=True)
        X = as_float_array(X, copy=self.copy)
        self.mean_ = X.mean(axis=0)
        X_ = X - self.mean_
        cov = np.dot(X_.T, X_) / (X_.shape[0]-1)
        U, S, _ = linalg.svd(cov)
        s = np.sqrt(S.clip(self.regularization))
        s_inv = np.diag(1./s)
        s = np.diag(s)
        self.whiten_ = np.dot(np.dot(U, s_inv), U.T)
        self.dewhiten_ = np.dot(np.dot(U, s), U.T)
        return self

    def transform(self, X, y=None, copy=None):
        """Perform ZCA whiteningzcc
        Parameters
        ----------
        X : array-like with shape [n_samples, n_features]
            The data to whiten along the features axis.
        """
        check_is_fitted(self, 'mean_')
        X = as_float_array(X, copy=self.copy)
        return np.dot(X - self.mean_, self.whiten_.T)

    def inverse_transform(self, X, copy=None):
        """Undo the ZCA transform and rotate back to the original
        representation
        Parameters
        ----------
        X : array-like with shape [n_samples, n_features]
            The data to rotate back.
        """
        check_is_fitted(self, 'mean_')
        X = as_float_array(X, copy=self.copy)
        return np.dot(X, self.dewhiten_) + self.mean_

@tf.custom_gradient
def GradientReversalOperator(x):
    def grad(dy):
        return -1 * dy

    return x, grad
class GradientReversalLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(GradientReversalLayer, self).__init__()

    def call(self, inputs):
        return GradientReversalOperator(inputs)
def conv1d_block(filters, k_size, strides, padding, input):
    # init = RandomNormal(stddev=0.02)
    g = layers.Conv1D(filters, k_size, strides=strides, padding=padding)(input)
    g = layers.BatchNormalization()(g)
    g = layers.LeakyReLU(0.3)(g)
    return g
# resnet
def TFData_preprocessing(x,y,batch_size,conditional=True):
  if conditional:
      x=tf.data.Dataset.from_tensor_slices((x,y))
      x=x.shuffle(10000).batch(batch_size)
  else:
      x=tf.data.Dataset.from_tensor_slices(x)
      x=x.shuffle(10000).batch(batch_size)

  return x

class DANN(tf.keras.Model):
    def __init__(self, input_shape, num_classes):
        super(DANN, self).__init__()

        self.feature_extractor,features = self.build_FeatureExtractor(input_shape)
        self.label_classifier= self.build_LabelPredictor((features.shape[1],),num_classes)
        self.domain_classifier = self.build_domain_classifier((features.shape[1],))

        # self.optimizer_cls = tf.keras.optimizers.Adam(1e-4)
        # self.optimizer_domain = tf.keras.optimizers.Adam(1e-4)

        self.optimizer_cls=tf.keras.optimizers.SGD(learning_rate=1e-4, momentum=0.9, nesterov=True)
        self.optimizer_domain=tf.keras.optimizers.SGD(learning_rate=1e-4, momentum=0.9, nesterov=True)

        self.loss_cls = tf.keras.losses.CategoricalCrossentropy()  # for label predictor
        self.loss_domain = tf.keras.losses.BinaryCrossentropy()  # for domain classifier
        # self.loss_mmd=MMDLoss()
    def build_FeatureExtractor(self,input_shape):
        in_image = layers.Input(shape=input_shape)
        #first module
        x = layers.Reshape((1024, 1))(in_image)

        g=conv1d_block(16,5,8,'same',x)
        g = layers.MaxPooling1D(strides=2)(g)

        g = conv1d_block(32, 5, 1, 'same', g)
        g = layers.MaxPooling1D(strides=2)(g)

        features = layers.Flatten()(g)
        model=tf.keras.models.Model(inputs=in_image, outputs=features)
        return model,features
    def build_ADa(self,input_shape):
        input_features = layers.Input(shape=input_shape)
        # g=layers.Dropout(0.5)(input_features)
        g=layers.Dense(1024)(input_features)
        g = layers.BatchNormalization()(g)
        g=layers.Activation('relu')(g)
        # g = layers.Dropout(0.5)(g)
        g = layers.Dense(512)(g)
        g = layers.BatchNormalization()(g)
        latent = layers.Activation('relu')(g)
        model_label = tf.keras.models.Model(inputs=input_features, outputs=latent)
        return model_label,latent
    def build_LabelPredictor(self,input_shape,num_classes):
        input_features = layers.Input(shape=input_shape)
        # g=layers.Dropout(0.5)(input_features)
        g=layers.Dense(128)(input_features)
        g = layers.BatchNormalization()(g)
        g=layers.Activation('relu')(g)
        # g = layers.Dropout(0.5)(g)
        g = layers.Dense(64)(g)
        g = layers.BatchNormalization()(g)
        latent = layers.Activation('relu')(g)
        logits = layers.Dense(num_classes, activation='softmax')(latent)
        model_label = tf.keras.models.Model(inputs=input_features, outputs=logits)
        return model_label
    def build_domain_classifier(self,input_shape):  # domain classifier
        input_features = layers.Input(shape=input_shape)
        g=GradientReversalLayer()(input_features)
        g = layers.Reshape((1024, 1))(g)
        g=conv1d_block(16,5,8,'same',g)
        g = layers.MaxPooling1D(strides=2)(g)
        g=conv1d_block(8,5,4,'same',g)
        g = layers.MaxPooling1D(strides=2)(g)
        g = layers.Flatten()(g)
        logits = layers.Dense(1, activation='sigmoid')(g)
        model = tf.keras.models.Model(inputs=input_features, outputs=logits)
        return model
    def label_classifier_loss(self,y_true, y_pred):
        return self.loss_cls(y_true, y_pred)
    def domain_classifier_loss(self,y_true, y_pred):
        return self.loss_domain(y_true, y_pred)
    def train_loss(self, source_img, source_label, target_img):
        # train_classifier
        with tf.GradientTape(persistent=True) as tape:
            source_features = self.feature_extractor(source_img,training=True)
            target_features = self.feature_extractor(target_img,training=True)
            source_pred = self.label_classifier(source_features,training=True)
            loss_s_class = self.label_classifier_loss(source_label, source_pred)

            s_disc_outputs = self.domain_classifier(source_features, training=True)
            t_disc_outputs = self.domain_classifier(target_features, training=True)

            loss_disc=self.domain_classifier_loss(tf.zeros_like(s_disc_outputs), s_disc_outputs)\
                    + self.domain_classifier_loss(tf.ones_like(t_disc_outputs), t_disc_outputs)
            # loss_domian=loss_disc
            # loss_discrimiative=self.loss_mmd(source_features,target_features)
            loss_discrimiative = mmd_loss(source_features, target_features, kernel=True)
            loss_domian=loss_disc+0.5*loss_discrimiative

            #label swd
            target_pred = self.label_classifier(target_features, training=True)
            loss_label=sw_loss(source_pred,target_pred,source_pred.shape[0],source_pred.shape[0])
            loss_mian=loss_s_class+0.5*loss_label

        var_d = self.domain_classifier.trainable_variables
        var_g = self.feature_extractor.trainable_variables+\
                self.label_classifier.trainable_variables

        disc_gradients = tape.gradient(loss_domian, var_d)
        main_gradients = tape.gradient(loss_mian, var_g)
        # fe_grad = tape.gradient(loss_discrimiative, self.feature_extractor.trainable_variables)

        self.optimizer_domain.apply_gradients(zip(disc_gradients, var_d))
        self.optimizer_cls.apply_gradients(zip(main_gradients, var_g))
        # self.optimizer_domain.apply_gradients(zip(fe_grad, self.feature_extractor.trainable_variables))

        return  loss_disc,loss_s_class,loss_discrimiative
    def train_loss2(self, source_img, source_label):
        with tf.GradientTape() as tape:
            source_features = self.feature_extractor(source_img,training=True)
            source_pred = self.label_classifier(source_features,training=True)
            loss_s_class = self.label_classifier_loss(source_label, source_pred)
            loss_main = loss_s_class

        var_g = self.feature_extractor.trainable_variables+\
                self.label_classifier.trainable_variables
        main_gradients = tape.gradient(loss_main, var_g)
        self.optimizer_cls.apply_gradients(zip(main_gradients, var_g))
        return  loss_main

    @tf.function
    def train_on_step(self, source_img,source_label,target_img):
        loss = self.train_loss(source_img,source_label,target_img)
        return loss
    @tf.function
    def train_on_step2(self, source_img,source_label):
        class_loss = self.train_loss2(source_img,source_label)
        return class_loss


    def train(self,source_train_db,target_train_db,source_test_db,target_test_db,epochs):
        loss_cls=[]
        loss_domian=[]
        loss_discrimiative=[]
        accuracy_s_t=[]
        accuracy_t_t = []

        for epoch in range(epochs):
            for (source_img, source_label), (target_img, _) in zip(source_train_db, target_train_db):
                domain_loss,cls_loss,discriminative_loss=self.train_on_step(source_img,source_label,target_img)
            loss_cls.append(cls_loss),loss_domian.append(domain_loss),loss_discrimiative.append(discriminative_loss)
            #for testing
            # acc in source test
            correct, total = 0, 0
            for sx_T, sy_T in source_test_db:
                pred_sy_T = self.label_classifier(self.feature_extractor(sx_T))
                pred_y_T = tf.cast(tf.argmax(pred_sy_T, axis=-1),tf.int32)
                y_T = tf.cast(sy_T, tf.int32)
                correct += float(tf.reduce_sum(tf.cast(tf.equal(pred_y_T, y_T), tf.float32)))
                total += sx_T.shape[0]
                acc_source_test = correct / total
                accuracy_s_t.append(acc_source_test)

            #acc in target test
            correct2, total2 = 0, 0
            for tx_T, ty_T in target_test_db:
                pred_ty_T = self.label_classifier(self.feature_extractor(tx_T))
                pred_y_T2 = tf.cast(tf.argmax(pred_ty_T, axis=-1),tf.int32)
                y_T2 = tf.cast(ty_T, tf.int32)
                correct2 += float(tf.reduce_sum(tf.cast(tf.equal(pred_y_T2, y_T2), tf.float32)))
                total2 += tx_T.shape[0]
                acc_target_test = correct2 / total2
                accuracy_t_t.append(acc_target_test)

        return np.array(accuracy_s_t),np.array(accuracy_t_t),np.array(loss_cls),np.array(loss_domian),np.array(loss_discrimiative)

    def train_source_only(self, source_train_db, source_test_db, target_test_db, epochs):

        for epoch in range(epochs):
            for source_img, source_label in source_train_db:
                cls_loss = self.train_on_step2(source_img, source_label)

            #for testing
            # acc in source test
            correct, total = 0, 0
            for sx_T, sy_T in source_test_db:
                pred_sy_T = self.label_classifier(self.feature_extractor(sx_T))

                pred_y_T = tf.cast(tf.argmax(pred_sy_T, axis=-1),tf.int32)
                y_T = tf.cast(sy_T, tf.int32)
                correct += float(tf.reduce_sum(tf.cast(tf.equal(pred_y_T, y_T), tf.float32)))
                total += sx_T.shape[0]
                acc_source_test = correct / total

            #acc in target test
            correct2, total2 = 0, 0
            for tx_T, ty_T in target_test_db:
                pred_ty_T = self.label_classifier(self.feature_extractor(tx_T))
                pred_y_T2 = tf.cast(tf.argmax(pred_ty_T, axis=-1),tf.int32)
                y_T2 = tf.cast(ty_T, tf.int32)
                correct2 += float(tf.reduce_sum(tf.cast(tf.equal(pred_y_T2, y_T2), tf.float32)))
                total2 += tx_T.shape[0]
                acc_target_test = correct2 / total2


            print('epoch:', epoch, 'cls_loss:', cls_loss.numpy(),
                  'acc in source',acc_source_test,'acc in target',acc_target_test)

#load source data
def LoadData_pickle(path,name,type='rb'):
  with open(path+name+'.pkl', type) as f:
          data=pickle.load(f)

  return data

faults=['C3S3L0_all','C4S3L0_all','C6S3L0_all','C7S3L0_all','C8S3L0_all']
# real load
dataset_real_T=[]
dataset_real_t=[]
for fault in faults:
    x=LoadData_pickle(path='../dataset/original/',
                            name=fault)[5]

    xT,xt=x[0:700],x[700:]

    dataset_real_T.extend(xT)
    dataset_real_t.extend(xt)
dataset_real_train=np.array(dataset_real_T)
dataset_real_test=np.array(dataset_real_t)

#load target data
faults=['gen_data_C_1','gen_data_C_2','gen_data_C_3','gen_data_C_4','gen_data_C_5']
dataset_gen_T=[]
dataset_gen_t=[]
for fault in faults:
    x=LoadData_pickle(path='../dataset/generated/',
                            name=fault)
    xT, xt = x[0:700], x[700:]

    dataset_gen_T.extend(xT)
    dataset_gen_t.extend(xt)
dataset_gen_train=np.array(dataset_gen_T)
dataset_gen_test=np.array(dataset_gen_t)


num_T=700
train_label=tf.one_hot(np.array([0]*num_T+[1]*num_T+[2]*num_T+[3]*num_T+[4]*num_T),depth=5)
num_t=300
test_label=np.array([0]*num_t+[1]*num_t+[2]*num_t+[3]*num_t+[4]*num_t)
print(train_label.shape,test_label.shape)

source_train_db = TFData_preprocessing(dataset_real_train, train_label, batch_size=32)
source_test_db = TFData_preprocessing(dataset_real_test, test_label, batch_size=32)

target_train_db = TFData_preprocessing(dataset_gen_train, train_label, batch_size=32)
target_test_db = TFData_preprocessing(dataset_gen_test, test_label, batch_size=32)

dann=DANN(input_shape=(1024,),num_classes=5)
# #pre-train
epoch=1000
# # dann.train_source_only(source_train_db, source_test_db, target_test_db, epochs=100)
acc_source,acc_target,loss_cls,loss_domian,loss_dis=dann.train(source_train_db, target_train_db,source_test_db,target_test_db, epochs=epoch)
# print(acc_source,acc_target)