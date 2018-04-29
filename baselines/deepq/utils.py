import os

import tensorflow as tf

# ================================================================
# Saving variables
# ================================================================

def load_state(fname):
    saver = tf.train.Saver()
    saver.restore(tf.get_default_session(), fname)

def save_state(fname):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    saver = tf.train.Saver()
    saver.save(tf.get_default_session(), fname)

# ================================================================
# Placeholders
# ================================================================

class TfInput(object):
    def __init__(self, name="(unnamed)"):
        """Generalized Tensorflow placeholder. The main differences are:
            - possibly uses multiple placeholders internally and returns multiple values
            - can apply light postprocessing to the value feed to placeholder.
        """
        self.name = name

    def get(self):
        """Return the tf variable(s) representing the possibly postprocessed value
        of placeholder(s).
        """
        raise NotImplemented()

    def make_feed_dict(data):
        """Given data input it to the placeholder(s)."""
        raise NotImplemented()


class PlaceholderTfInput(TfInput):
    def __init__(self, placeholder):
        """Wrapper for regular tensorflow placeholder."""
        super().__init__(placeholder.name)
        self._placeholder = placeholder

    def get(self):
        return self._placeholder

    def make_feed_dict(self, data):
        return {self._placeholder: data}

class BatchInput(PlaceholderTfInput):
    def __init__(self, shape, dtype=tf.float32, name=None):
        """Creates a placeholder for a batch of tensors of a given shape and dtype

        Parameters
        ----------
        shape: [int]
            shape of a single elemenet of the batch
        dtype: tf.dtype
            number representation used for tensor contents
        name: str
            name of the underlying placeholder
        """
        super().__init__(tf.placeholder(dtype, [None] + list(shape), name=name))

class Uint8Input(PlaceholderTfInput):
    def __init__(self, shape, name=None):
        """Takes input in uint8 format which is cast to float32 and divided by 255
        before passing it to the model.

        On GPU this ensures lower data transfer times.

        Parameters
        ----------
        shape: [int]
            shape of the tensor.
        name: str
            name of the underlying placeholder
        """

        super().__init__(tf.placeholder(tf.uint8, [None] + list(shape), name=name))
        self._shape = shape
        self._output = tf.cast(super().get(), tf.float32) / 255.0

    def get(self):
        return self._output

def create_dirs(dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return:
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
    except Exception as err:
        print("Creating directories error: {0}".format(err))

def create_list_dirs(input_dir, prefix_name, count):
    dirs_path = []
    for i in range(count):
        dirs_path.append(input_dir + prefix_name + '-' + str(i))
        create_dirs([input_dir + prefix_name + '-' + str(i)])
    return dirs_path
