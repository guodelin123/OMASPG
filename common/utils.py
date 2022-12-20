import tensorflow as tf
import numpy as np
import random
import datetime as dt
import collections
import csv

def str2bool(str):
    if str.lower() in ['none', 'false', 'null']:
        return False
    return  True

def int_or_None(str):
    if str.lower() == 'none':
        return None
    else:
        return int(str)

def float_or_None(str):
    if str.lower() == 'none':
        return None
    else:
        return float(str)


def huber_loss(e, d):
    a = tf.cast(tf.abs(e) <= d, tf.float32)
    b = tf.cast(tf.abs(e) > d, tf.float32)
    return a * e * e / 2 + b * d * (tf.abs(e) - d / 2)

def mse_loss(e):
    return e * e / 2

# following functions are copied from openai baselines

def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        #lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4: # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init


def fc(x, scope, nh, *, init_scale=1.0, init_bias=0.0):
    with tf.variable_scope(scope):
        nin = x.get_shape()[1].value
        w = tf.get_variable("w", [nin, nh], initializer=ortho_init(init_scale))
        b = tf.get_variable("b", [nh], initializer=tf.constant_initializer(init_bias))
        return tf.matmul(x, w)+b


def set_global_seed(seed):
    np.random.seed(seed)
    tf.set_random_seed(seed)
    random.seed(seed)
    return [seed]

def get_token(unique=False):
    # get the time as the unique token
    if not unique:
        return ''
    t = dt.datetime.now()
    unique_token = 'id_{}_{}_{}_{}_{}_{}'.format(t.year, t.month, t.day, t.hour, t.minute, t.second)
    return unique_token

def function(inputs, outputs, updates=None, givens=None):
    """Just like Theano function. Take a bunch of tensorflow placeholders and expressions
    computed based on those placeholders and produces f(inputs) -> outputs. Function f takes
    values to be fed to the input's placeholders and produces the values of the expressions
    in outputs.

    Input values can be passed in the same order as inputs or can be provided as kwargs based
    on placeholder name (passed to constructor or accessible via placeholder.op.name).

    Example:
        x = tf.placeholder(tf.int32, (), name="x")
        y = tf.placeholder(tf.int32, (), name="y")
        z = 3 * x + 2 * y
        lin = function([x, y], z, givens={y: 0})

        with single_threaded_session():
            initialize()

            assert lin(2) == 6
            assert lin(x=3) == 9
            assert lin(2, 2) == 10
            assert lin(x=2, y=3) == 12

    Parameters
    ----------
    inputs: [tf.placeholder, tf.constant, or object with make_feed_dict method]
        list of input arguments
    outputs: [tf.Variable] or tf.Variable
        list of outputs or a single output to be returned from function. Returned
        value will also have the same shape.
    """
    if isinstance(outputs, list):
        return _Function(inputs, outputs, updates, givens=givens)
    elif isinstance(outputs, (dict, collections.OrderedDict)):
        f = _Function(inputs, outputs.values(), updates, givens=givens)
        return lambda *args, **kwargs: type(outputs)(zip(outputs.keys(), f(*args, **kwargs)))
    else:
        f = _Function(inputs, [outputs], updates, givens=givens)
        return lambda *args, **kwargs: f(*args, **kwargs)[0]


class _Function(object):
    def __init__(self, inputs, outputs, updates, givens):
        for inpt in inputs:
            if not hasattr(inpt, 'make_feed_dict') and not (type(inpt) is tf.Tensor and len(inpt.op.inputs) == 0):
                assert False, "inputs should all be placeholders, constants, or have a make_feed_dict method"
        self.inputs = inputs
        updates = updates or []
        self.update_group = tf.group(*updates)
        self.outputs_update = list(outputs) + [self.update_group]
        self.givens = {} if givens is None else givens

    def _feed_input(self, feed_dict, inpt, value):
        if hasattr(inpt, 'make_feed_dict'):
            feed_dict.update(inpt.make_feed_dict(value))
        else:
            feed_dict[inpt] = value

    def __call__(self, *args):
        assert len(args) <= len(self.inputs), "Too many arguments provided"
        feed_dict = {}
        # Update the args
        for inpt, value in zip(self.inputs, args):
            self._feed_input(feed_dict, inpt, value)
        # Update feed dict with givens.
        for inpt in self.givens:
            feed_dict[inpt] = feed_dict.get(inpt, self.givens[inpt])
        results = tf.get_default_session().run(self.outputs_update, feed_dict=feed_dict)[:-1]
        return results
