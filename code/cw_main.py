from __future__ import print_function
import os
import numpy as np
import pickle

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.ops import init_ops

from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score, precision_score, recall_score

from cw import cw
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.random.seed(12345)

###########################
# Input format
# data_train.pkl: n_train x timestamps x n_feature
# target_train.pkl: n_train x label
# data_test.pkl: n_test x timestamps x n_feature
# target_test.pkl: n_test x label
###########################

def run_fold(path, hidden_dim, fc_dim, timesteps, test_flag, lamb_list):

    print('\nLoading Data')

    path_string = path + '/data_train.pkl'
    fn = open(path_string, 'rb')
    data_train = pickle.load(fn)

    path_string = path + '/target_train.pkl'
    fn = open(path_string, 'rb')
    labels_train = pickle.load(fn)

    path_string = path + '/data_test.pkl'
    fn = open(path_string, 'rb')
    data_test = pickle.load(fn)

    path_string = path + '/target_test.pkl'
    fn = open(path_string, 'rb')
    labels_test = pickle.load(fn)

    input_dim = data_test.shape[2]
    output_dim = labels_test.shape[1]

    print('\nConstruction graph')


    def model(x, logits=False, training=False):
        x = tf.unstack(x, timesteps, 1)

        initializer = init_ops.random_normal_initializer(mean=0, stddev=0.1, seed=1)
        # regularizer = tf.contrib.layers.l1_regularizer(scale=0.05)
        with tf.variable_scope('rnn', initializer=initializer):
            # rnn_cell = rnn.BasicRNNCell(hidden_dim)
            rnn_cell = rnn.BasicLSTMCell(hidden_dim)
            outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

        with tf.variable_scope('fc', initializer=initializer):
            z = tf.layers.dense(outputs[-1], units=fc_dim, activation=tf.nn.relu)
            # z = tf.layers.dropout(z, rate=0.25, training=training, seed=1)

        with tf.variable_scope('sm', initializer=initializer):
            logits_ = tf.layers.dense(z, units=output_dim, name='logits')

        y = tf.nn.softmax(logits_, name='ybar')
        if logits:
            return y, logits_
        return y


    class Dummy:
        pass

    env = Dummy()


    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
        env.x = tf.placeholder(tf.float32, (None, timesteps, input_dim), name='x')
        env.y = tf.placeholder(tf.float32, (None, output_dim), name='y')
        env.training = tf.placeholder_with_default(False, (), name='mode')

        env.ybar, logits = model(env.x, logits=True, training=env.training)

        with tf.variable_scope('acc'):
            count = tf.equal(tf.argmax(env.y, axis=1), tf.argmax(env.ybar, axis=1))
            env.acc = tf.reduce_mean(tf.cast(count, tf.float32), name='acc')

        with tf.variable_scope('loss'):
            reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'model')
            xent = tf.nn.softmax_cross_entropy_with_logits(labels=env.y, logits=logits)
            env.loss = tf.reduce_mean(xent) + tf.reduce_sum(reg_loss)

        with tf.variable_scope('train_op'):
            optimizer = tf.train.AdamOptimizer()
            vs = tf.global_variables()
            env.train_op = optimizer.minimize(env.loss, var_list=vs)

        env.saver = tf.train.Saver()

        # Note here that the shape has to be fixed during the graph construction
        # since the internal variable depends upon the shape.
        env.x_fixed = tf.placeholder(tf.float32, (1, timesteps, input_dim), name='x_fixed')
        env.adv_lamb = tf.placeholder(tf.float32, (), name='adv_lamb')

        # Better to use SGD optimizer, Adam optimizer will have problems in providing gradients for ISTA
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.02)
        env.adv_train_op, env.xadv, env.noise, env.setter = cw(model, env.x_fixed, lamb=env.adv_lamb, optimizer=optimizer)

    print('\nInitializing graph')

    tf.set_random_seed(1)
    env.sess = tf.InteractiveSession()
    env.sess.run(tf.global_variables_initializer())
    env.sess.run(tf.local_variables_initializer())


    def evaluate(env, X_data, y_data):

        loss, acc = env.sess.run([env.loss, env.acc], feed_dict={env.x: X_data, env.y: y_data})
        print(' loss: {0:.4f} acc: {1:.4f}'.format(loss, acc))

        return loss, acc


    def train(env, X_data, y_data, X_valid=None, y_valid=None, epochs=1, load=False, batch_size=64, name='model'):

        if load:
            if not hasattr(env, 'saver'):
                return print('\nError: cannot find saver op')
            print('\nLoading saved model')
            dir = path + '/model_l1/{}'
            return env.saver.restore(env.sess, dir.format(name))

        print('\nTrain model')
        n_sample = X_data.shape[0]
        n_batch = int((n_sample+batch_size-1) / batch_size)
        for epoch in range(epochs):

            for batch in range(n_batch):
                start = batch * batch_size
                end = min(n_sample, start + batch_size)
                env.sess.run(env.train_op, feed_dict={env.x: X_data[start:end], env.y: y_data[start:end], env.training: True})
            if X_valid is not None:
                evaluate_2(env, X_valid, y_valid)

        if hasattr(env, 'saver'):
            print('\n Saving model')
            dir = path + '/model'
            if not os.path.exists(dir):   # compatible for python 2.
                os.makedirs(dir)
            # os.makedirs(dir, exist_ok=True) # compatible for python 3.
            fn = dir + '/{}'
            env.saver.save(env.sess, fn.format(name))


    def predict(env, X_data):
        return env.sess.run(env.ybar, feed_dict={env.x: X_data})


    def evaluate_2(env, X_data, y_data):

        y_pred = predict(env, X_data)
        Y_true = np.argmax(y_data, axis=1)
        Y_pred = np.argmax(y_pred, axis=1)
        auc_score = roc_auc_score(y_data[:, 1], y_pred[:, 1], average='micro')
        f1 = f1_score(Y_true, Y_pred)

        print(' AUC: {0:.4f} f1: {1:.4f}'.format(auc_score, f1))
        return auc_score, f1


    def make_cw(env, x_data, epochs=10000, lamb=0.0):

        print('\nMaking adversarials:')

        x_adv = np.empty_like(x_data)
        feed_dict = {env.x_fixed: x_data, env.adv_lamb: lamb}

        env.sess.run(env.noise.initializer)
        flag = 0    # break if 1
        maxp = 0    # maximum perturbation
        nonz = 0    # number of location changed
        avgp = 0    # average perturbation
        for epoch in range(epochs):
            # print('Epoch:', epoch)
            env.sess.run(env.adv_train_op, feed_dict=feed_dict)
            env.sess.run(env.setter, feed_dict=feed_dict)
            xadv = env.sess.run(env.xadv, feed_dict=feed_dict)

            ypred = predict(env, x_data)
            yadv = predict(env, xadv)

            label_ypred = np.argmax(ypred, axis=1)
            label_yadv = np.argmax(yadv, axis=1)

            if label_yadv != label_ypred:
                print('Classification changed at Epoch {}!'.format(epoch))
                flag += 1

                diff = xadv - x_data
                maxp = np.max(np.max(abs(diff), axis=0))
                nonz = np.count_nonzero(diff)
                avgp = np.sum(np.sum(abs(diff), axis=0))/nonz
                print('Maximum perturbation: {:.4f}, Number of cells changed: {:d}, Average pertubation: {:.4f}'.format(maxp, nonz, avgp))
                print()
                break
        return x_adv, maxp, nonz, avgp, flag


    print('\nTraining')
    train(env, data_train, labels_train, data_test, labels_test, load=test_flag, epochs=25, name='model')

    print('\nEvaluating on clean data')
    evaluate(env, data_test, labels_test)

    y_pred= predict(env, data_train)
    Y_true = np.argmax(labels_train, axis=1)
    Y_pred = np.argmax(y_pred, axis=1)
    auc_score = roc_auc_score(labels_train[:, 1], y_pred[:, 1], average='micro')
    f1 = f1_score(Y_true, Y_pred)
    ps = precision_score(Y_true, Y_pred)
    rc = recall_score(Y_true, Y_pred)
    cm = confusion_matrix(Y_true, Y_pred)

    print("Train AUC_score = {:.4f}".format(auc_score))
    print("Train f1 = {:.4f}".format(f1))
    print("Train PS_score = {:.4f}".format(ps))
    print("Train RC_score = {:.4f}".format(rc))
    print(cm)

    y_pred= predict(env, data_test)
    Y_true = np.argmax(labels_test, axis=1)
    Y_pred = np.argmax(y_pred, axis=1)
    auc_score = roc_auc_score(labels_test[:, 1], y_pred[:, 1], average='micro')
    f1 = f1_score(Y_true, Y_pred)
    ps = precision_score(Y_true, Y_pred)
    rc = recall_score(Y_true, Y_pred)
    cm = confusion_matrix(Y_true, Y_pred)

    print("Test AUC_score = {:.4f}".format(auc_score))
    print("Test f1 = {:.4f}".format(f1))
    print("Test PS_score = {:.4f}".format(ps))
    print("Test RC_score = {:.4f}".format(rc))
    print(cm)


    print('\nGenerating adversarial data')
    # Identify correct labeled samples
    idx = np.equal(Y_true, Y_pred)
    data_clean = data_test[idx]

    n_obs = data_clean.shape[0]
    n_lamb = len(lamb_list)
    print(n_obs, n_lamb)
    MP = np.zeros((n_obs, n_lamb))
    AP = np.zeros((n_obs, n_lamb))
    NZ = np.zeros((n_obs, n_lamb))
    FL = np.zeros((n_obs, n_lamb))

    for i in range(n_obs):

        x = data_clean[i:i + 1]
        for j in range(len(lamb_list)):
            print(j)

            lamb = lamb_list[j]
            x_adv, maxp, nonz, avgp, flag = make_cw(env, x, epochs=100, lamb=lamb)
            MP[i, j] = maxp
            AP[i, j] = avgp
            NZ[i, j] = nonz
            FL[i, j] = flag

    dir = path + '/output'
    if not os.path.exists(dir):   # compatible for python 2.
        os.makedirs(dir)
    # os.makedirs(dir, exist_ok=True)  # compatible for python 3.

    fn = dir + '/Adv_metric.pkl'
    f = open(fn, 'wb')
    pickle.dump([MP, AP, NZ, FL], f, protocol=2)


if __name__ == '__main__':

    hidden_dim = 128
    fc_dim = 32
    timesteps = 48
    nFold = 5
    test_flag=False

    seq = np.linspace(np.log(0.0005), np.log(0.015), num=25)
    lamb_list = np.exp(seq)
    for i in range(nFold):
        i=0
        path = '../fold_' + str(i)
        print("\n=========   Fold: {}  =========".format(i))
        tf.reset_default_graph()
        run_fold(path, hidden_dim, fc_dim, timesteps, test_flag, lamb_list)
        break

