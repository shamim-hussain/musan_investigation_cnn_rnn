

import numpy as np
from keras.layers import *
from keras.models import *
from keras.optimizers import *

from keras import backend as K
from keras.utils import to_categorical
import tensorflow as tf
from custom_layers import *
from keras.callbacks import *

from keras.applications.mobilenet import MobileNet, relu6, DepthwiseConv2D
from keras.utils.generic_utils import CustomObjectScope

def get_gru(in_shape, name='', num_classes=3):
    in_t = Input(shape=in_shape)
    x = in_t
    
#    x = Conv1D(32, 3, padding='same', activation='relu')(x)
#    x = Dropout(.1)(x)
    
    p = SigTanActivation()(Conv1D(32, 3, padding='causal')(x))
    q = SigTanActivation()(SeparableConv1D(32, 6)(ZeroPadding1D((5,0))(x)))
    x = Concatenate()([p,q])
    x = Dropout(.1)(x)
    
    x = CuDNNGRU(32)(x)

    #x = GlobalAveragePooling1D()(x)
    #x = Dense(16, activation='relu')(x)
    x = Dropout(.1)(x)
    #x = Dense(16, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)
    
    out_t = x
    
    return Model(in_t, out_t, name=name or 'MobileNet')


def get_mobile_net(in_shape, name='', num_classes=3):
    in_t = Input(shape=in_shape)
    x = in_t
    base_model = MobileNet(include_top=False, input_shape=(160, 160, 3),
                           alpha=.25, depth_multiplier=1, dropout=0.25,
                           pooling='avg', weights='imagenet')#'imagenet'
    
    def run_thru(x, name=None):
        with CustomObjectScope({'relu6': relu6,
                                'DepthwiseConv2D': DepthwiseConv2D}):
            tdict = {base_model.input: x}
            lrs = base_model.layers[1:]
            for k, lr in zip(range(len(lrs), 0, -1), lrs):
                l_in = lr.input
                l_out = lr.output
                
                in_t = [tdict[t] for t in l_in]\
                    if isinstance(l_in, list)\
                    else tdict[l_in]
                        
                cg = lr.get_config()
                if name:
                    cg['name'] = name+'_'+cg['name']
                else:
                    del cg['name']
                mlr = lr.__class__.from_config(cg)
                
                out_t = mlr(in_t)
                tdict[l_out] = out_t
                
                mlr.set_weights(lr.get_weights())
                mlr.trainable = True #False if k>4 else 
                print('Built {}/{} layers'.format(k, len(lrs)))
        
        return tdict[base_model.output]
    
    
    x = Lambda(lambda d: d[:, :, :, None])(x)
    x = Lambda(lambda img: tf.image.grayscale_to_rgb(img))(x)
    #x = UpSampling2D((1, 2))(x)
    x = run_thru(x)
    x = Dropout(.1)(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)
    out_t = x
    
    return Model(in_t, out_t, name=name or 'MobileNet')


def get_wave_net(in_shape, name='', num_classes=3):

    in_t = Input(shape=in_shape)
    x = in_t
    
    p = SigTanActivation()(Conv1D(16, 3, padding='causal')(x))
    q = SigTanActivation()(SeparableConv1D(16, 6)(ZeroPadding1D((5,0))(x)))
    x = Concatenate()([p,q])
    x = Dropout(.1)(x)
    
    p = SigTanActivation()(Conv1D(8, 3, padding='causal')(x))
    q = SigTanActivation()(SeparableConv1D(8, 6)(ZeroPadding1D((5,0))(x)))
    x = Concatenate()([p,q])
    x = Dropout(.1)(x)
    
    y = x
    p = SigTanActivation()(Conv1D(8, 3, padding='causal')(x))
    q = SigTanActivation()(SeparableConv1D(8, 6, dilation_rate=1)
                            (ZeroPadding1D((5,0))(x)))
    x = Concatenate()([p,q])
    x = Add()([x,y])
    x = Dropout(.1)(x)
    
    y = x
    x = Conv1D(16, 3, dilation_rate=3, padding='causal')(x)
    x = SigTanActivation()(x)
    x = Add()([x,y])
    x = Dropout(.1)(x)
    
    y = x
    x = Conv1D(16, 3, dilation_rate=6, padding='causal')(x)
    x = SigTanActivation()(x)
    s3 = x
    x = Add()([x,y])
    x = Dropout(.1)(x)
    
    y = x
    x = Conv1D(16, 3, dilation_rate=12, padding='causal')(x)
    x = SigTanActivation()(x)
    s2 = x
    x = Add()([x,y])
    x = Dropout(.1)(x)
    
    y = x
    x = Conv1D(16, 3, dilation_rate=24, padding='causal')(x)
    x = SigTanActivation()(x)
    s1 = x
    #y = Add()([x,y])
    x = Dropout(.1)(x)
    
    x = Conv1D(32, 3, dilation_rate=48, padding='causal')(x)
    x = SigTanActivation()(x)
    
    class Choice_layer(Layer):
        def __init__(self, fr, **kwargs):
            self.fr = fr
            super().__init__(**kwargs)
        def build(self, input_shape):
            self.pos = K.random_uniform(shape=(), minval=self.fr, maxval=1.)
            self.built = True
        def call(self, x, training=None):
            lenx = tf.shape(x)[1]
            cp = tf.cast(tf.round((tf.cast(lenx, tf.float32)-.5)*self.pos), tf.int32)
            cx = x[:, cp, :]
            return K.in_train_phase(cx, x[:, -1, :], training=training)
        def compute_output_shape(self, input_shape):
            return (input_shape[0], input_shape[2])
    
    #x = Lambda(lambda x: x[:,-1, :])(x)
    #s1 = Lambda(lambda x: x[:, -1, :])(s1)
    #s2 = Lambda(lambda x: x[:, -1, :])(s2)
    #s3 = Lambda(lambda x: x[:, -1, :])(s3)
    
    x = Concatenate()([x, s1, s2, s3])
    x = Choice_layer(.5)(x)
    #x = Lambda(lambda x: x[:,-1, :])(x)
    
    x = Dense(16, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)
    out_t = x

    return Model(in_t, out_t)






