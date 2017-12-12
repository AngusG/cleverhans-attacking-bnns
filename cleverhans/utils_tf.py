from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from distutils.version import LooseVersion
import math
import numpy as np
import os
from six.moves import xrange
import tensorflow as tf
import time
import warnings
import logging

from .utils import batch_indices, _ArgsWrapper, create_logger, set_log_level

FLAGS = tf.app.flags.FLAGS

_logger = create_logger("cleverhans.utils.tf")


class _FlagsWrapper(_ArgsWrapper):

    """
    Wrapper that tries to find missing parameters in TensorFlow FLAGS
    for backwards compatibility.

    Plain _ArgsWrapper should be used instead if the support for FLAGS
    is removed.
    """

    def __getattr__(self, name):
        val = self.args.get(name)
        if val is None:
            warnings.warn('Setting parameters ({}) from TensorFlow FLAGS is '
                          'deprecated.'.format(name))
            val = FLAGS.__getattr__(name)
        return val


def model_loss(y, model, mean=True):
    """
    Define loss of TF graph
    :param y: correct labels
    :param model: output of the model
    :param mean: boolean indicating whether should return mean of loss
                 or vector of losses for each input of the batch
    :return: return mean of loss if True, otherwise return vector with per
             sample loss
    """

    op = model.op
    if "softmax" in str(op).lower():
        logits, = op.inputs
    else:
        logits = model

    out = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)

    if mean:
        out = tf.reduce_mean(out)
    return out


def initialize_uninitialized_global_variables(sess):
    """
    Only initializes the variables of a TensorFlow session that were not
    already initialized.
    :param sess: the TensorFlow session
    :return:
    """
    # List all global variables
    global_vars = tf.global_variables()

    # Find initialized status for all variables
    is_var_init = [tf.is_variable_initialized(var) for var in global_vars]
    is_initialized = sess.run(is_var_init)

    # List all variables that were not initialized previously
    not_initialized_vars = [var for (var, init) in
                            zip(global_vars, is_initialized) if not init]

    # Initialize all uninitialized variables found, if any
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))


def sign_changes_count_op(prv_k, k):

    print(k)
    print(prv_k)
    return tf.cast(np.product(k.shape[:]), tf.int32) - tf.reduce_sum(
        tf.cast(tf.equal(prv_k, k), tf.int32))


def normalized_sign_changes_op(prv_k, k):

    return 1.0 - tf.reduce_sum(tf.cast(tf.equal(
        prv_k, k), tf.float32)) / tf.cast(np.product(k.shape[:]), tf.float32)


def create_kernel_placeholder(model, i):

    return tf.placeholder(
        tf.float32, [model.layers[i].kernels.shape[0], model.layers[i].kernels.shape[1],
                     model.layers[i].kernels.shape[2], model.layers[i].kernels.shape[3]])


def model_train(sess, x, y, predictions, X_train, Y_train, model=None, phase=None,
                writer=None, save=False, predictions_adv=None, init_all=False,
                evaluate=None, verbose=True, feed=None, args=None, rng=None):
    """
    Train a TF graph
    :param sess: TF session to use when training the graph
    :param x: input placeholder
    :param y: output placeholder (for labels)
    :param predictions: model output predictions
    :param X_train: numpy array with training inputs
    :param Y_train: numpy array with training outputs
    :param save: boolean controlling the save operation
    :param predictions_adv: if set with the adversarial example tensor,
                            will run adversarial training
    :param init_all: (boolean) If set to true, all TF variables in the session
                     are (re)initialized, otherwise only previously
                     uninitialized variables are initialized before training.
    :param evaluate: function that is run after each training iteration
                     (typically to display the test/validation accuracy).
    :param verbose: (boolean) all print statements disabled when set to False.
    :param feed: An optional dictionary that is appended to the feeding
                 dictionary before the session runs. Can be used to feed
                 the learning phase of a Keras model for instance.
    :param args: dict or argparse `Namespace` object.
                 Should contain `nb_epochs`, `learning_rate`,
                 `batch_size`
                 If save is True, should also contain 'log_dir'
                 and 'filename'
    :param rng: Instance of numpy.random.RandomState
    :return: True if model trained
    """
    args = _FlagsWrapper(args or {})

    # Check that necessary arguments were given (see doc above)
    # assert args.binary, "Precision was not given in args dict"
    assert args.nb_epochs, "Number of epochs was not given in args dict"
    assert args.learning_rate, "Learning rate was not given in args dict"
    assert args.batch_size, "Batch size was not given in args dict"

    if save:
        assert args.log_dir, "Directory for save was not given in args dict"
        assert args.filename, "Filename for save was not given in args dict"

    if not verbose:
        set_log_level(logging.WARNING)
        warnings.warn("verbose argument is deprecated and will be removed"
                      " on 2018-02-11. Instead, use utils.set_log_level()."
                      " For backward compatibility, log_level was set to"
                      " logging.WARNING (30).")

    if rng is None:
        rng = np.random.RandomState()

    # Define loss
    loss = model_loss(y, predictions)
    if predictions_adv is not None:
        loss = (loss + model_loss(y, predictions_adv)) / 2

    with tf.variable_scope(args.train_scope, reuse=args.reuse_global_step):
        global_step = tf.get_variable(
            "global_step", dtype=tf.int32, initializer=tf.constant(0), trainable=False)

    if args.binary:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = tf.contrib.layers.optimize_loss(
                loss, global_step, learning_rate=args.learning_rate, optimizer='Adam',
                summaries=["gradients"])
    else:
        train_step = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
        train_step = train_step.minimize(loss)

    if writer is not None:
        if args.binary:
            k_ph = []
            k_prv_ph = []
            layer_indices = []
            for i, layer_name in enumerate(model.layer_names):
                if 'Conv2D' in layer_name and layer_name != 'Conv2D0':
                    k_ph.append(create_kernel_placeholder(model, i))
                    k_prv_ph.append(create_kernel_placeholder(model, i))
                    layer_indices.append(i)
            conv2_sign_changes = sign_changes_count_op(k_prv_ph[0], k_ph[0])
            conv3_sign_changes = sign_changes_count_op(k_prv_ph[1], k_ph[1])
            conv2_sign_changes_summary = tf.summary.scalar(
                "sign_changes/conv2", conv2_sign_changes)
            conv3_sign_changes_summary = tf.summary.scalar(
                "sign_changes/conv3", conv3_sign_changes)

        assert args.loss_name, "Name of scalar summary loss"
        training_summary = tf.summary.scalar(args.loss_name, loss)
        merge_op = tf.summary.merge_all()

    with sess.as_default():
        if hasattr(tf, "global_variables_initializer"):
            if init_all:
                tf.global_variables_initializer().run()
            else:
                initialize_uninitialized_global_variables(sess)
        else:
            warnings.warn("Update your copy of tensorflow; future versions of "
                          "CleverHans may drop support for this version.")
            sess.run(tf.initialize_all_variables())

        init_step = sess.run(global_step)

        for epoch in xrange(args.nb_epochs):
            # Compute number of batches
            nb_batches = int(math.ceil(float(len(X_train)) / args.batch_size))
            assert nb_batches * args.batch_size >= len(X_train)

            # Indices to shuffle training set
            index_shuf = list(range(len(X_train)))
            rng.shuffle(index_shuf)

            prev = time.time()
            for batch in range(nb_batches):

                step = init_step + (epoch * nb_batches + batch)

                # Compute batch start and end indices
                start, end = batch_indices(
                    batch, len(X_train), args.batch_size)

                if batch % 100 == 0:
                    if writer is not None:
                        if args.binary:
                            k_prv_np = []
                            for i in layer_indices:
                                k_prv_np.append(
                                    sess.run(model.layers[i].kernels))

                # Perform one training step
                feed_dict = {x: X_train[index_shuf[start:end]],
                             y: Y_train[index_shuf[start:end]],
                             phase: args.is_training}
                if feed is not None:
                    feed_dict.update(feed)
                sess.run(train_step, feed_dict=feed_dict)

                if batch % 100 == 0:
                    if writer is not None:
                        if args.binary:
                            k_np = []
                            for i in layer_indices:
                                k_np.append(sess.run(model.layers[i].kernels))
                            for i in range(len(layer_indices)):
                                feed_dict.update(
                                    {k_prv_ph[i]: k_prv_np[i], k_ph[i]: k_np[i]})
                        loss_val, merged_summ = sess.run(
                            [loss, merge_op], feed_dict=feed_dict)
                        writer.add_summary(merged_summ, step)
                        writer.flush()
                    else:
                        loss_val = sess.run(loss, feed_dict=feed_dict)

                    #print('epoch %d, batch %d, step %d, loss %.4f' %
                    #      (epoch, batch, step, loss_val))

            assert end >= len(X_train)  # Check that all examples were used
            cur = time.time()
            if verbose:
                _logger.info("Epoch " + str(epoch) + " took " +
                             str(cur - prev) + " seconds")
            if evaluate is not None:
                evaluate()
            #global_step = step
        if save:
            save_path = os.path.join(args.log_dir, args.filename)
            #save_path = args.log_dir
            saver = tf.train.Saver()
            if not os.path.exists(args.log_dir):
                os.makedirs(args.log_dir)
            saver.save(sess, save_path, global_step=step)
            _logger.info("Completed model training and saved at: " +
                         str(save_path))
        else:
            _logger.info("Completed model training.")

    return True


def model_eval(sess, x, y, predictions=None, X_test=None, Y_test=None, phase=None, writer=None,
               feed=None, args=None, model=None):
    """
    Compute the accuracy of a TF model on some data
    :param sess: TF session to use when training the graph
    :param x: input placeholder
    :param y: output placeholder (for labels)
    :param predictions: model output predictions
    :param X_test: numpy array with training inputs
    :param Y_test: numpy array with training outputs
    :param feed: An optional dictionary that is appended to the feeding
             dictionary before the session runs. Can be used to feed
             the learning phase of a Keras model for instance.
    :param args: dict or argparse `Namespace` object.
                 Should contain `batch_size`
    :param model: (deprecated) if not None, holds model output predictions
    :return: a float with the accuracy value
    """
    args = _FlagsWrapper(args or {})

    assert args.batch_size, "Batch size was not given in args dict"
    if X_test is None or Y_test is None:
        raise ValueError("X_test argument and Y_test argument "
                         "must be supplied.")
    if model is None and predictions is None:
        raise ValueError("One of model argument "
                         "or predictions argument must be supplied.")
    if model is not None:
        warnings.warn("model argument is deprecated. "
                      "Switch to predictions argument. "
                      "model argument will be removed after 2018-01-05.")
        if predictions is None:
            predictions = model
        else:
            raise ValueError("Exactly one of model argument"
                             " and predictions argument should be specified.")

    # Define accuracy symbolically
    if LooseVersion(tf.__version__) >= LooseVersion('1.0.0'):
        correct_preds = tf.equal(tf.argmax(y, axis=-1),
                                 tf.argmax(predictions, axis=-1))
    else:
        correct_preds = tf.equal(tf.argmax(y, axis=tf.rank(y) - 1),
                                 tf.argmax(predictions,
                                           axis=tf.rank(predictions) - 1))

    acc_value = tf.reduce_mean(tf.to_float(correct_preds))

    # Init result var
    accuracy = 0.0

    if writer is not None:
        eval_summary = tf.summary.scalar('acc', acc_value)

    with sess.as_default():

        # Compute number of batches
        nb_batches = int(math.ceil(float(len(X_test)) / args.batch_size))
        assert nb_batches * args.batch_size >= len(X_test)

        for batch in range(nb_batches):
            if batch % 100 == 0 and batch > 0:
                _logger.debug("Batch " + str(batch))

            # Must not use the `batch_indices` function here, because it
            # repeats some examples.
            # It's acceptable to repeat during training, but not eval.
            start = batch * args.batch_size
            end = min(len(X_test), start + args.batch_size)
            cur_batch_size = end - start

            # The last batch may be smaller than all others, so we need to
            # account for variable batch size here
            feed_dict = {x: X_test[start:end],
                         y: Y_test[start:end],
                         phase: False}
            if feed is not None:
                feed_dict.update(feed)

            if writer is not None:
                cur_acc, eval_summ = sess.run(
                    [acc_value, eval_summary], feed_dict=feed_dict)
                writer.add_summary(eval_summ, batch)
                writer.flush()
            else:
                cur_acc = acc_value.eval(feed_dict=feed_dict)
            accuracy += (cur_batch_size * cur_acc)

        assert end >= len(X_test)

        # Divide by number of examples to get final value
        accuracy /= len(X_test)

    return accuracy


def tf_model_load(sess, file_path=None):
    """

    :param sess: the session object to restore
    :param file_path: path to the restored session, if None is
                      taken from FLAGS.log_dir and FLAGS.filename
    :return:
    """
    with sess.as_default():
        saver = tf.train.Saver()
        if file_path is None:
            file_path = os.path.join(FLAGS.log_dir, FLAGS.filename)
        saver.restore(sess, tf.train.latest_checkpoint(file_path))

    return True


def batch_eval(sess, tf_inputs, tf_outputs, numpy_inputs, feed=None,
               args=None):
    """
    A helper function that computes a tensor on numpy inputs by batches.

    :param sess:
    :param tf_inputs:
    :param tf_outputs:
    :param numpy_inputs:
    :param feed: An optional dictionary that is appended to the feeding
             dictionary before the session runs. Can be used to feed
             the learning phase of a Keras model for instance.
    :param args: dict or argparse `Namespace` object.
                 Should contain `batch_size`
    """
    args = _FlagsWrapper(args or {})

    assert args.batch_size, "Batch size was not given in args dict"

    n = len(numpy_inputs)
    assert n > 0
    assert n == len(tf_inputs)
    m = numpy_inputs[0].shape[0]
    for i in xrange(1, n):
        assert numpy_inputs[i].shape[0] == m
    out = []
    for _ in tf_outputs:
        out.append([])
    with sess.as_default():
        for start in xrange(0, m, args.batch_size):
            batch = start // args.batch_size
            if batch % 100 == 0 and batch > 0:
                _logger.debug("Batch " + str(batch))

            # Compute batch start and end indices
            start = batch * args.batch_size
            end = start + args.batch_size
            numpy_input_batches = [numpy_input[start:end]
                                   for numpy_input in numpy_inputs]
            cur_batch_size = numpy_input_batches[0].shape[0]
            assert cur_batch_size <= args.batch_size
            for e in numpy_input_batches:
                assert e.shape[0] == cur_batch_size

            feed_dict = dict(zip(tf_inputs, numpy_input_batches))
            if feed is not None:
                feed_dict.update(feed)
            numpy_output_batches = sess.run(tf_outputs, feed_dict=feed_dict)
            for e in numpy_output_batches:
                assert e.shape[0] == cur_batch_size, e.shape
            for out_elem, numpy_output_batch in zip(out, numpy_output_batches):
                out_elem.append(numpy_output_batch)

    out = [np.concatenate(x, axis=0) for x in out]
    for e in out:
        assert e.shape[0] == m, e.shape
    return out


def model_argmax(sess, x, predictions, samples, feed=None):
    """
    Helper function that computes the current class prediction
    :param sess: TF session
    :param x: the input placeholder
    :param predictions: the model's symbolic output
    :param samples: numpy array with input samples (dims must match x)
    :param feed: An optional dictionary that is appended to the feeding
             dictionary before the session runs. Can be used to feed
             the learning phase of a Keras model for instance.
    :return: the argmax output of predictions, i.e. the current predicted class
    """
    feed_dict = {x: samples}
    if feed is not None:
        feed_dict.update(feed)
    probabilities = sess.run(predictions, feed_dict)

    if samples.shape[0] == 1:
        return np.argmax(probabilities)
    else:
        return np.argmax(probabilities, axis=1)


def l2_batch_normalize(x, epsilon=1e-12, scope=None):
    """
    Helper function to normalize a batch of vectors.
    :param x: the input placeholder
    :param epsilon: stabilizes division
    :return: the batch of l2 normalized vector
    """
    with tf.name_scope(scope, "l2_batch_normalize") as scope:
        x_shape = tf.shape(x)
        x = tf.contrib.layers.flatten(x)
        x /= (epsilon + tf.reduce_max(tf.abs(x), 1, keep_dims=True))
        square_sum = tf.reduce_sum(tf.square(x), 1, keep_dims=True)
        x_inv_norm = tf.rsqrt(np.sqrt(epsilon) + square_sum)
        x_norm = tf.multiply(x, x_inv_norm)
        return tf.reshape(x_norm, x_shape, scope)


def kl_with_logits(p_logits, q_logits, scope=None,
                   loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES):
    """Helper function to compute kl-divergence KL(p || q)
    """
    with tf.name_scope(scope, "kl_divergence") as name:
        p = tf.nn.softmax(p_logits)
        p_log = tf.nn.log_softmax(p_logits)
        q_log = tf.nn.log_softmax(q_logits)
        loss = tf.reduce_mean(tf.reduce_sum(p * (p_log - q_log), axis=1),
                              name=name)
        tf.losses.add_loss(loss, loss_collection)
        return loss

def clip_eta(eta, ord, eps):
    """
    Helper function to clip the perturbation to epsilon norm ball.
    :param eta: A tensor with the current perturbation.
    :param ord: Order of the norm (mimics Numpy).
                Possible values: np.inf, 1 or 2.
    :param eps: Epilson, bound of the perturbation.
    """

    # Clipping perturbation eta to self.ord norm ball
    if ord not in [np.inf, 1, 2]:
        raise ValueError('ord must be np.inf, 1, or 2.')
    if ord == np.inf:
        eta = tf.clip_by_value(eta, -eps, eps)
    elif ord in [1, 2]:
        reduc_ind = list(xrange(1, len(eta.get_shape())))
        if ord == 1:
            norm = tf.reduce_sum(tf.abs(eta),
                                 reduction_indices=reduc_ind,
                                 keep_dims=True)
        elif ord == 2:
            norm = tf.sqrt(tf.reduce_sum(tf.square(eta),
                                         reduction_indices=reduc_ind,
                                         keep_dims=True))
        eta = eta * eps / norm
    return eta
