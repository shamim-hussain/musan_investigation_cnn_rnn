
from keras.layers import *
from keras import backend as K
import tensorflow as tf


class CausalConv1D(Conv1D):
    def __init__(self, *args,**kwargs):
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        ks = self.kernel_size[0]
        kernel_shape = ((ks+1)//2, input_dim, self.filters)
        zeros_shape = (ks//2, input_dim, self.filters)

        kernel_var = self.add_weight(shape=kernel_shape,
                                     initializer=self.kernel_initializer,
                                     name='kernel',
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint)

        kernel_cons = K.zeros(zeros_shape, dtype=K.floatx())
        self.kernel = K.concatenate([kernel_var,kernel_cons], axis=0)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True
        

class SigTanActivation(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape=input_shape)
        self.built = True

    def call(self, x: tf.Tensor):
        if x.shape.as_list()[-1]%2:
            raise ValueError('Number of features must be even!given='+
                             str(x.shape.as_list()[-1]))
        x1, x2 = tf.split(x, 2, axis=-1)
        d = tf.nn.tanh(x1)
        m = tf.nn.sigmoid(x2)
        return d*m

    def compute_output_shape(self, input_shape):
        if input_shape[-1]%2:
            raise ValueError('Number of features must be even! given='+
                             str(x.shape.as_list()[-1]))
        return (*input_shape[:-1], int(input_shape[-1] / 2))


class SigTanActivation2(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape=input_shape)
        self.built = True

    def call(self, x: tf.Tensor):
        if x.shape.as_list()[-1]%2:
            raise ValueError('Number of features must be even!given='+
                             str(x.shape.as_list()[-1]))
        x1, x2 = tf.split(x, 2, axis=-1)
        d = tf.nn.tanh(x1)
        m = tf.nn.sigmoid(x2)
        return tf.concat((d*m, d*(1-m)), axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape