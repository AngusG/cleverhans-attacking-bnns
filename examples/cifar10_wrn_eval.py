from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import numpy as np
from six.moves import xrange

import keras
from keras import backend
from keras.utils import np_utils
from cifar10_attack import data_cifar10_std
# import cifar10_input as data_input

import argparse
import logging
import tensorflow as tf
from tensorflow.python.platform import flags

from cleverhans.utils_mnist import data_mnist
from cleverhans.utils import to_categorical
from cleverhans.utils import set_log_level
from cleverhans.utils import build_model_save_path, parse_model_settings
from cleverhans.utils_tf import model_train, model_eval, batch_eval, tf_model_load
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks_tf import jacobian_graph, jacobian_augmentation

# from cleverhans_tutorials.tutorial_models import make_basic_cnn,
# make_basic_binary_cnn, MLP
from cleverhans_tutorials.tutorial_models import MLP
from cleverhans_tutorials.tutorial_models import Flatten, Linear, SimpleLinear, ReLU, SoftmaxT1

# for the oracle, be sure to:
# export PYTHONPATH=$PYTHONPATH:~/Documents/low-bitwidth/wrn-tensorflow

import resnet

FLAGS = flags.FLAGS
MAX_BATCH_SIZE = 100
TEST_SET_SIZE = 1000


def setup_tutorial():
    """
    Helper function to check correct configuration of tf for tutorial
    :return: True if setup checks completed
    """

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    return True


def prep_bbox(sess, x, y, X_train, Y_train, X_test, Y_test,
              img_rows, img_cols, channels, nb_epochs, batch_size, learning_rate,
              rng, phase=None, binary=False, scale=False, nb_filters=64,
              model_path=None, adv=False, delay=0, eps=0.3):
    """
    Define and train a model that simulates the "remote"
    black-box oracle described in the original paper.
    :param sess: the TF session
    :param x: the input placeholder for MNIST
    :param y: the ouput placeholder for MNIST
    :param X_train: the training data for the oracle
    :param Y_train: the training labels for the oracle
    :param X_test: the testing data for the oracle
    :param Y_test: the testing labels for the oracle
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param learning_rate: learning rate for training
    :param rng: numpy.random.RandomState
    :return:
    """

    # Define TF model graph (for the black-box model)
    save = False
    train_from_scratch = False

    if model_path is not None:
        if os.path.exists(model_path):
            # check for existing model in immediate subfolder
            if any(f.endswith('.meta') for f in os.listdir(model_path)):
                binary, scale, nb_filters, batch_size, learning_rate, nb_epochs, adv = parse_model_settings(
                    model_path)
                train_from_scratch = False
            else:
                model_path = build_model_save_path(
                    model_path, binary, batch_size, nb_filters, learning_rate, nb_epochs, adv, delay, scale)
                print(model_path)
                save = True
                train_from_scratch = True
    else:
        train_from_scratch = True  # train from scratch, but don't save since no path given

    if binary:
        if scale:
            from cleverhans_tutorials.tutorial_models import make_scaled_binary_cnn
            model = make_scaled_binary_cnn(phase, 'bb_binsc_', input_shape=(
                None, img_rows, img_cols, channels), nb_filters=nb_filters)
        else:
            from cleverhans_tutorials.tutorial_models import make_basic_binary_cnn
            model = make_basic_binary_cnn(phase, 'bb_bin_', input_shape=(
                None, img_rows, img_cols, channels), nb_filters=nb_filters)
    else:
        from cleverhans_tutorials.tutorial_models import make_basic_cnn
        model = make_basic_cnn(phase, 'bb_fp_', input_shape=(
            None, img_rows, img_cols, channels), nb_filters=nb_filters)

    preds = model(x, reuse=False)
    print("Defined TensorFlow model graph.")

    def evaluate():
        # Print out the accuracy on legitimate data
        eval_params = {'batch_size': batch_size}
        acc = model_eval(
            sess, x, y, preds, X_test, Y_test, phase=phase, args=eval_params)
        print('Test accuracy of black-box on legitimate test '
              'examples: %.4f' % acc)

    # Train an MNIST model
    train_params = {
        'binary': binary,
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'loss_name': 'bb train loss',
        'filename': 'bb_model',
        'train_scope': 'bb_model',
        'reuse_global_step': False,
        'is_training': True
    }

    if adv:
        from cleverhans.attacks import FastGradientMethod
        fgsm = FastGradientMethod(model, back='tf', sess=sess)
        fgsm_params = {'eps': eps, 'clip_min': 0., 'clip_max': 1.}
        adv_x_train = fgsm.generate(x, phase, **fgsm_params)
        preds_adv = model.get_probs(adv_x_train)

    if train_from_scratch:
        if save:
            train_params.update({'log_dir': model_path})
            if adv and delay > 0:
                train_params.update({'nb_epochs': delay})

        # do clean training for 'nb_epochs' or 'delay' epochs
        model_train(sess, x, y, preds, X_train, Y_train, phase=phase,
                    evaluate=evaluate, args=train_params, save=save, rng=rng)

        # optionally do additional adversarial training
        if adv:
            print("Adversarial training for %d epochs" % (nb_epochs - delay))
            train_params.update({'nb_epochs': nb_epochs - delay})
            train_params.update({'reuse_global_step': True})
            model_train(sess, x, y, preds, X_train, Y_train, phase=phase,
                        predictions_adv=preds_adv, evaluate=evaluate, args=train_params,
                        save=save, rng=rng)
    else:
        tf_model_load(sess, model_path)
        print('Restored model from %s' % model_path)

    accuracy = evaluate()

    return model, preds, accuracy, model_path


def substitute_model(img_rows=32, img_cols=32, nb_classes=10):
    """
    Defines the model architecture to be used by the substitute. Use
    the example model interface.
    :param img_rows: number of rows in input
    :param img_cols: number of columns in input
    :param nb_classes: number of classes in output
    :return: tensorflow model
    """
    input_shape = (None, img_rows, img_cols, 3)

    # Define a fully connected model (it's different than the black-box)
    layers = [Flatten(),
              SimpleLinear(200),
              ReLU(),
              SimpleLinear(200),
              ReLU(),
              SimpleLinear(nb_classes),
              SoftmaxT1()]

    return MLP(layers, input_shape)


def train_sub(sess, x, y, bbox_preds, X_sub, Y_sub, nb_classes,
              nb_epochs_s, batch_size, learning_rate, data_aug, lmbda,
              rng, binary=False, phase=None, model_path=None):
    """
    This function creates the substitute by alternatively
    augmenting the training data and training the substitute.
    :param sess: TF session
    :param x: input TF placeholder
    :param y: output TF placeholder
    :param bbox_preds: output of black-box model predictions
    :param X_sub: initial substitute training data
    :param Y_sub: initial substitute training labels
    :param nb_classes: number of output classes
    :param nb_epochs_s: number of epochs to train substitute model
    :param batch_size: size of training batches
    :param learning_rate: learning rate for training
    :param data_aug: number of times substitute training data is augmented
    :param lmbda: lambda from arxiv.org/abs/1602.02697
    :param rng: numpy.random.RandomState instance
    :param phase: placeholder for batch_norm phase (training or testing)
    :param phase_val: True if training, False if testing
    :return:
    """
    # Define TF model graph (for the black-box model)
    model_sub = substitute_model()
    preds_sub = model_sub(x)
    print("Defined TensorFlow model graph for the substitute.")

    # Define the Jacobian symbolically using TensorFlow
    grads = jacobian_graph(preds_sub, x, nb_classes)

    train_params = {
        'binary': False,
        'nb_epochs': nb_epochs_s,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'filename': 'sub_model',
        'train_scope': 'sub_model',
        'reuse_global_step': False,
        'is_training': True
    }

    # Train the substitute and augment dataset alternatively
    for rho in xrange(data_aug):
        print("Substitute training epoch #" + str(rho))

        if rho > 0:
            train_params.update({'reuse_global_step': True})
        if model_path is not None:
            train_params.update({'log_dir': model_path})
            model_train(sess, x, y, preds_sub, X_sub, to_categorical(Y_sub),
                        phase=phase, save=True, init_all=False, args=train_params,
                        rng=rng)
        else:
            model_train(sess, x, y, preds_sub, X_sub, to_categorical(Y_sub),
                        phase=phase, init_all=False, args=train_params,
                        rng=rng)

        # If we are not at last substitute training iteration, augment dataset
        if rho < data_aug - 1:
            print("Augmenting substitute training data.")
            # Perform the Jacobian augmentation
            X_sub = jacobian_augmentation(sess, x, X_sub, Y_sub, grads, lmbda)

            print("Labeling substitute training data.")
            # Label the newly generated synthetic points using the black-box
            Y_sub = np.hstack([Y_sub, Y_sub])
            X_sub_prev = X_sub[int(len(X_sub) / 2):]
            eval_params = {'batch_size': batch_size}
            bbox_val = batch_eval(sess, [x], [bbox_preds], [X_sub_prev],
                                  feed={phase: False}, args=eval_params)[0]
            # Note here that we take the argmax because the adversary
            # only has access to the label (not the probabilities) output
            # by the black-box model
            Y_sub[int(len(X_sub) / 2):] = np.argmax(bbox_val, axis=1)

    return model_sub, preds_sub


def cifar10_blackbox(nb_classes=10, batch_size=128, nb_samples=10, l2_weight=0.0001,
                     momentum=0.9, initial_lr=0.1, lr_step_epoch=100.0, lr_decay=0.1,
                     num_residual_units=2, num_train_instance=50000, num_test_instance=10000,
                     k=1, eps=0.3, learning_rate=0.001, nb_epochs=10,
                     holdout=150, data_aug=6, nb_epochs_s=10, lmbda=0.1, binary=False,
                     scale=False, model_path=None, targeted=False, data_dir=None,
                     adv=False, delay=0):
    """
    MNIST tutorial for the black-box attack from arxiv.org/abs/1602.02697
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :return: a dictionary with:
             * black-box model accuracy on test set
             * substitute model accuracy on test set
             * black-box model accuracy on adversarial examples transferred
               from the substitute model
    """

    # Set logging level to see debug information
    set_log_level(logging.DEBUG)

    # Dictionary used to keep track and return key accuracies
    accuracies = {}

    # Perform tutorial setup
    assert setup_tutorial()

    if not hasattr(backend, "tf"):
        raise RuntimeError("This tutorial requires keras to be configured"
                           " to use the TensorFlow backend.")

    # Image dimensions ordering should follow the Theano convention
    if keras.backend.image_dim_ordering() != 'tf':
        keras.backend.set_image_dim_ordering('tf')
        print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to "
              "'th', temporarily setting to 'tf'")

    # Create TF session and set as Keras backend session
    sess = tf.Session()
    keras.backend.set_session(sess)

    # Get CIFAR10 test data
    X_train, Y_train, X_test, Y_test = data_cifar10_std()

    # Y_train_onehot = np_utils.to_categorical(Y_train, nb_classes)
    Y_test_onehot = np_utils.to_categorical(Y_test, nb_classes)

    # Y_test is for evaluating oracle
    Y_test_bbox = np.argmax(Y_test, axis=1)
    Y_test_bbox = Y_test_bbox.reshape(Y_test_bbox.shape[0],)
    Y_test_bbox = Y_test_bbox.astype('int32')


    #Y_test = Y_test.reshape(Y_test.shape[0],)
    #Y_test = Y_test.astype('int32')
    #Y_train = Y_train.astype('int32')

    # Initialize substitute training set reserved for adversary
    X_sub = X_test[:holdout]
    Y_sub = np.argmax(Y_test_onehot[:holdout], axis=1)

    # Redefine test set as remaining samples unavailable to adversaries
    X_test = X_test[holdout:]
    Y_test = Y_test[holdout:]

    # CIFAR10-specific dimensions
    img_rows = 32
    img_cols = 32
    channels = 3

    rng = np.random.RandomState([2017, 8, 30])

    # with tf.Graph().as_default():

    # Define input and output TF placeholders
    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, channels))
    y = tf.placeholder(tf.int32, shape=(None))

    phase = tf.placeholder(tf.bool, name='phase')
    y_s = tf.placeholder(tf.float32, shape=(None, nb_classes))

    # Seed random number generator so tutorial is reproducible

    # Simulate the black-box model locally
    # You could replace this by a remote labeling API for instance
    print("Preparing the WideResNet black-box model.")
    '''
    prep_bbox_out = prep_bbox(sess, x, y, X_train, Y_train, X_test, Y_test,
                              img_rows, img_cols, channels, nb_epochs, batch_size, learning_rate,
                              rng=rng, phase=phase, binary=binary, scale=scale,
                              nb_filters=nb_filters, model_path=model_path,
                              adv=adv, delay=delay, eps=eps)

    model, bbox_preds, accuracies['bbox'], model_path = prep_bbox_out
    '''
    decay_step = lr_step_epoch * num_train_instance / batch_size
    hp = resnet.HParams(batch_size=batch_size,
                        num_classes=nb_classes,
                        num_residual_units=num_residual_units,
                        k=k,
                        weight_decay=l2_weight,
                        initial_lr=initial_lr,
                        decay_step=decay_step,
                        lr_decay=lr_decay,
                        momentum=momentum)

    print(binary)
    binary = True if binary else False
    print(binary)
    network = resnet.ResNet(binary, hp, x, y, None)
    network.build_model()

    # bbox_preds = network.preds
    bbox_preds = network.probs

    init = tf.global_variables_initializer()
    sess.run(init)

    # Create a saver.
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=10000)

    if 'model' in model_path.split('/')[-1]:
        saver.restore(sess, model_path)
        print('restored %s' % model_path)
    else:
        saver.restore(
            sess, tf.train.latest_checkpoint(model_path))
        print('restored %s' % model_path)

    '''
    if os.path.isdir(model_path):
        ckpt = tf.train.get_checkpoint_state(model_path)
        # Restores from checkpoint
        if ckpt and ckpt.model_checkpoint_path:
            print('\tRestore from %s' % ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint file found in the dir [%s]' % model_path)
            sys.exit(1)
    elif os.path.isfile(model_path):
        print('\tRestore from %s' % model_path)
        saver.restore(sess, model_path)
    else:
        print('No checkpoint file found in the path [%s]' % model_path)
        sys.exit(1)
    '''

    eval_params = {'batch_size': batch_size}
    acc = model_eval(
        sess, x, y, bbox_preds, X_test, Y_test, phase=phase, args=eval_params)
    print('Test accuracy of black-box on legitimate test examples: %.4f' % acc)

    


def main(argv=None):
    cifar10_blackbox(nb_classes=FLAGS.nb_classes,
                     batch_size=FLAGS.batch_size,
                     nb_samples=FLAGS.nb_samples,
                     l2_weight=FLAGS.l2_weight,
                     momentum=FLAGS.momentum,
                     initial_lr=FLAGS.initial_lr,
                     lr_step_epoch=FLAGS.lr_step_epoch,
                     lr_decay=FLAGS.lr_decay,
                     num_residual_units=FLAGS.num_residual_units,
                     num_train_instance=FLAGS.num_train_instance,
                     num_test_instance=FLAGS.num_test_instance,
                     k=FLAGS.k,
                     eps=FLAGS.eps,
                     learning_rate=FLAGS.learning_rate,
                     nb_epochs=FLAGS.nb_epochs,
                     holdout=FLAGS.holdout,
                     data_aug=FLAGS.data_aug,
                     nb_epochs_s=FLAGS.nb_epochs_s,
                     lmbda=FLAGS.lmbda,
                     binary=FLAGS.binary,
                     scale=FLAGS.scale,
                     model_path=FLAGS.model_path,
                     targeted=FLAGS.targeted,
                     data_dir=FLAGS.data_dir,
                     adv=FLAGS.adv,
                     delay=FLAGS.delay)


if __name__ == '__main__':
    par = argparse.ArgumentParser()

    # General flags
    par.add_argument('--gpu', help='id of GPU to use')
    par.add_argument('--model_path', help='Path to save or load model')
    par.add_argument(
        '--data_dir', help='Path to training data', default='/scratch/gallowaa/mnist')
    par.add_argument('--nb_classes', type=int, default=10)
    par.add_argument('--batch_size', type=int, default=128)
    par.add_argument('--learning_rate', type=float, default=0.001)
    par.add_argument('--num_train_instance', type=int, default=50000,
                     help='Number of training images.')
    par.add_argument('--num_test_instance', type=int, default=10000,
                     help='Number of test images.')

    # For oracle
    par.add_argument('--nb_epochs', type=int, default=20)
    par.add_argument('--binary', help='Use a binary model?',
                     action="store_true")
    par.add_argument('--scale', help='Scale activations of the binary model?',
                     action="store_true")
    par.add_argument('--l2_weight', type=float, default=0.0001,
                     help='L2 loss weight applied all the weights')
    par.add_argument('--momentum', type=float, default=0.9,
                     help='The momentum of MomentumOptimizer')
    par.add_argument('--initial_lr', type=float, default=0.1,
                     help='Initial learning rate')
    par.add_argument('--lr_step_epoch', type=float, default=100.0,
                     help='Epochs after which learing rate decays')
    par.add_argument('--lr_decay', type=float, default=0.1,
                     help='Learning rate decay factor')
    '''
    par.add_argument('--nb_filters', type=int, default=64,
                        help='Number of filters in first layer')
    '''
    par.add_argument('--num_residual_units', type=int, default=4,
                     help='Number of residual block per group. Total number of conv layers will be 6n+4')
    par.add_argument('--k', type=int, default=2,
                     help='Network width multiplier')

    # For substitute
    par.add_argument('--holdout', type=int, default=150)
    par.add_argument('--data_aug', type=int, default=6)
    par.add_argument('--nb_epochs_s', type=int, default=10)
    par.add_argument('--lmbda', type=float, default=0.1)

    # Attack
    par.add_argument('--eps', type=float, default=0.1)
    par.add_argument(
        '--targeted', help='Run a targeted attack?', action="store_true")
    par.add_argument('--nb_samples', type=int,
                     default=10, help='Nb of inputs to attack')

    # Adversarial training flags
    par.add_argument(
        '--adv', help='Do FGSM adversarial training?', action="store_true")
    par.add_argument('--delay', type=int,
                     default=10, help='Nb of epochs to delay adv training by')

    FLAGS, _ = par.parse_known_args()

    if FLAGS.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    tf.app.run()
