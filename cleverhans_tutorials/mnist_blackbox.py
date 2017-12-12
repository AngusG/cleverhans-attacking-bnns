from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import numpy as np
from six.moves import xrange

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

from cleverhans_tutorials.tutorial_models import make_basic_cnn, make_basic_binary_cnn, MLP
from cleverhans_tutorials.tutorial_models import Flatten, Linear, SimpleLinear, ReLU, SoftmaxT1

FLAGS = flags.FLAGS
MAX_BATCH_SIZE = 100
TEST_SET_SIZE = 1000

INIT_T = 1.0
# enum adversarial training types
ADVERSARIAL_TRAINING_MADRYETAL = 1
ADVERSARIAL_TRAINING_FGSM = 2
MAX_EPS = 0.3


def setup_tutorial():
    """
    Helper function to check correct configuration of tf for tutorial
    :return: True if setup checks completed
    """

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    return True


def prep_bbox(sess, logits_scalar, x, y, X_train, Y_train, X_test, Y_test,
              img_rows, img_cols, channels, nb_epochs, batch_size, learning_rate,
              rng, phase=None, binary=False, scale=False, nb_filters=64,
              model_path=None, adv=0, delay=0, eps=0.3):
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
            #from cleverhans_tutorials.tutorial_models import make_scaled_binary_cnn
            # model = make_scaled_binary_cnn(phase, 'bb_binsc_', input_shape=(
            from cleverhans_tutorials.tutorial_models import make_scaled_binary_rand_cnn
            model = make_scaled_binary_rand_cnn(phase, logits_scalar, 'bb_binsc_', input_shape=(
                None, img_rows, img_cols, channels), nb_filters=nb_filters)
        else:
            from cleverhans_tutorials.tutorial_models import make_basic_binary_cnn
            model = make_basic_binary_cnn(phase, logits_scalar, 'bb_bin_', input_shape=(
                None, img_rows, img_cols, channels), nb_filters=nb_filters)
    else:
        from cleverhans_tutorials.tutorial_models import make_basic_cnn
        model = make_basic_cnn(phase, logits_scalar, 'bb_fp_', input_shape=(
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

    if adv != 0:
        if adv == ADVERSARIAL_TRAINING_MADRYETAL:
            from cleverhans.attacks import MadryEtAl
            nb_iter = 20
            train_attack_params = {'eps': MAX_EPS, 'eps_iter': 0.01,
                                   'nb_iter': nb_iter}
            train_attacker = MadryEtAl(model, sess=sess)

        if adv == ADVERSARIAL_TRAINING_FGSM:
            from cleverhans.attacks import FastGradientMethod
            train_attacker = FastGradientMethod(model, back='tf', sess=sess)

        # create the adversarial trainer
        train_attack_params.update({'clip_min': 0., 'clip_max': 1.})
        adv_x_train = train_attacker.generate(x, phase, **train_attack_params)
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


def substitute_model(img_rows=28, img_cols=28, nb_classes=10):
    """
    Defines the model architecture to be used by the substitute. Use
    the example model interface.
    :param img_rows: number of rows in input
    :param img_cols: number of columns in input
    :param nb_classes: number of classes in output
    :return: tensorflow model
    """
    input_shape = (None, img_rows, img_cols, 1)

    # Define a fully connected model (it's different than the black-box)
    layers = [Flatten(),
              SimpleLinear(200),
              ReLU(),
              SimpleLinear(200),
              ReLU(),
              SimpleLinear(nb_classes),
              SoftmaxT1()]

    return MLP(layers, input_shape)


def train_sub(sess, logits_scalar, x, y, bbox_preds, X_sub, Y_sub, nb_classes,
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


def mnist_blackbox(train_start=0, train_end=60000, test_start=0,
                   test_end=10000, nb_classes=10, batch_size=128, nb_samples=10, nb_filters=64,
                   eps=0.3, learning_rate=0.001, nb_epochs=10, holdout=150, data_aug=6,
                   nb_epochs_s=10, lmbda=0.1, binary=False, scale=False, model_path=None,
                   targeted=False, data_dir=None, adv=False, delay=0):
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

    # Create TF session
    sess = tf.Session()

    # Get MNIST data
    X_train, Y_train, X_test, Y_test = data_mnist(datadir=data_dir,
                                                  train_start=train_start,
                                                  train_end=train_end,
                                                  test_start=test_start,
                                                  test_end=test_end)

    # Initialize substitute training set reserved for adversary
    X_sub = X_test[:holdout]
    Y_sub = np.argmax(Y_test[:holdout], axis=1)
    print("Y_sub shape=")
    print(Y_sub.shape)


    # Redefine test set as remaining samples unavailable to adversaries
    X_test = X_test[holdout:]
    Y_test = Y_test[holdout:]

    # MNIST-specific dimensions
    img_rows = 28
    img_cols = 28
    channels = 1
    nb_classes = 10

    # Define input and output TF placeholders
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))
    phase = tf.placeholder(tf.bool, name='phase')

    logits_scalar = tf.placeholder_with_default(
        INIT_T, shape=(), name="logits_temperature")

    # Seed random number generator so tutorial is reproducible
    rng = np.random.RandomState([2017, 8, 30])

    # Simulate the black-box model locally
    # You could replace this by a remote labeling API for instance
    print("Preparing the black-box model.")
    prep_bbox_out = prep_bbox(sess, logits_scalar, x, y, X_train, Y_train, X_test, Y_test,
                              img_rows, img_cols, channels, nb_epochs, batch_size, learning_rate,
                              rng=rng, phase=phase, binary=binary, scale=scale,
                              nb_filters=nb_filters, model_path=model_path,
                              adv=adv, delay=delay, eps=eps)

    model, bbox_preds, accuracies['bbox'], model_path = prep_bbox_out

    # Train substitute using method from https://arxiv.org/abs/1602.02697
    print("Training the substitute model.")
    train_sub_out = train_sub(sess, logits_scalar, x, y, bbox_preds, X_sub, Y_sub,
                              nb_classes, nb_epochs_s, batch_size,
                              learning_rate, data_aug, lmbda, rng=rng,
                              phase=phase, model_path=model_path)
    model_sub, preds_sub = train_sub_out

    # Evaluate the substitute model on clean test examples
    eval_params = {'batch_size': batch_size}
    acc = model_eval(sess, x, y, preds_sub, X_test,
                     Y_test, phase=phase, args=eval_params)
    accuracies['sub'] = acc

    # Initialize the Fast Gradient Sign Method (FGSM) attack object.
    fgsm_par = {'eps': eps, 'ord': np.inf, 'clip_min': 0., 'clip_max': 1.}
    fgsm = FastGradientMethod(model_sub, sess=sess)

    if targeted:
        from cleverhans.utils import build_targeted_dataset
        adv_inputs, true_labels, adv_ys = build_targeted_dataset(
            X_test, Y_test, np.arange(nb_samples), nb_classes, img_rows, img_cols, channels)
        att_batch_size = np.clip(
            nb_samples * (nb_classes - 1), a_max=MAX_BATCH_SIZE, a_min=1)
        nb_adv_per_sample = nb_classes - 1
        yname = "y_target"

    else:
        att_batch_size = np.minimum(nb_samples, MAX_BATCH_SIZE)
        nb_adv_per_sample = 1
        adv_ys = None
        yname = "y"

    # Craft adversarial examples using the substitute
    eval_params = {'batch_size': att_batch_size}

    if targeted:
        fgsm_par.update({yname: adv_ys})
        x_adv_sub = fgsm.generate_np(adv_inputs, phase, **fgsm_par)
        accuracy = model_eval(sess, x, y, model(x, reuse=True), x_adv_sub, true_labels,
                              phase=phase, args=eval_params)
    else:
        x_adv_sub = fgsm.generate(x, phase, **fgsm_par)

        # Evaluate the accuracy of the "black-box" model on adversarial
        # examples
        accuracy = model_eval(sess, x, y, model(x_adv_sub), X_test, Y_test,
                              phase=phase, args=eval_params)
    print('Test accuracy of oracle on adversarial examples generated '
          'using the substitute: ' + str(accuracy))
    accuracies['bbox_on_sub_adv_ex'] = accuracy

    return accuracies


def main(argv=None):
    mnist_blackbox(nb_classes=FLAGS.nb_classes, batch_size=FLAGS.batch_size,
                   nb_samples=FLAGS.nb_samples, nb_filters=FLAGS.nb_filters,
                   eps=FLAGS.eps, learning_rate=FLAGS.learning_rate,
                   nb_epochs=FLAGS.nb_epochs, holdout=FLAGS.holdout,
                   data_aug=FLAGS.data_aug, nb_epochs_s=FLAGS.nb_epochs_s,
                   lmbda=FLAGS.lmbda, binary=FLAGS.binary, scale=FLAGS.scale, model_path=FLAGS.model_path,
                   targeted=FLAGS.targeted, data_dir=FLAGS.data_dir,
                   adv=FLAGS.adv, delay=FLAGS.delay)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # General flags
    parser.add_argument('--gpu', help='id of GPU to use')
    parser.add_argument("--model_path", help='Path to save or load model')
    parser.add_argument(
        '--data_dir', help='Path to training data', default='/scratch/gallowaa/mnist')
    parser.add_argument("--nb_classes", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.001)

    # For oracle
    parser.add_argument("--nb_epochs", type=int, default=15)
    parser.add_argument(
        '--binary', help='Use a binary model?', action="store_true")
    parser.add_argument('--scale', help='Scale activations of the binary model?',
                        action="store_true")
    parser.add_argument('--nb_filters', type=int, default=64,
                        help='Number of filters in first layer')

    # For substitute
    parser.add_argument("--holdout", type=int, default=150)
    parser.add_argument("--data_aug", type=int, default=6)
    parser.add_argument("--nb_epochs_s", type=int, default=10)
    parser.add_argument("--lmbda", type=float, default=0.1)

    # Attack
    parser.add_argument("--eps", type=float, default=0.3)
    parser.add_argument(
        '--targeted', help='Run a targeted attack?', action="store_true")
    parser.add_argument('--nb_samples', type=int,
                        default=10, help='Nb of inputs to attack')

    # Adversarial training flags
    parser.add_argument(
        '--adv', help='Adversarial training type?', type=int, default=0)
    parser.add_argument('--delay', type=int,
                        default=10, help='Nb of epochs to delay adv training by')

    FLAGS, _ = parser.parse_known_args()

    if FLAGS.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    tf.app.run()
