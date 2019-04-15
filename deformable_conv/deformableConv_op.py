
# coding: utf-8

# In[1]:


import tensorflow as tf
import keras.backend as K
import numpy as np


# In[2]:


def reshape(x, coord):
    """
      (b, h, w, 2*c)  -> (b*c, w, h, 2)
      (b, h, w, c) -> (b*c, h, w)
    
      x: (b, h, w, c) 
      coord: (b, h, w, 2*c) calculate by 1x1 convolution with 2 tuple of kernel
    """
    size = tf.shape(x)
    b = size[0]
    h = size[1]
    w = size[2]
    c = size[3]
    
    coord = tf.reshape(coord, (b, -1, c, 2))
    coord = tf.transpose(coord, [0,2,1,3])
    #coord = tf.reshape(coord, (b*c, -1, 2))
    
    #coord = tf.stack([coord_x, coord_y], axis=-1)
    coord = tf.reshape(coord, (b*c, h, w, 2))

    x = tf.transpose(x, [0,3,1,2])
    x = tf.reshape(x, (b*c, h, w))
    return x, coord
   
    
def warp_map(x, table):
    """
      x: (b, h, w)
      table: (b, w*h, 2)
    """
    size = tf.shape(x)
    b = size[0]
    h = size[1]
    w = size[2]
    
    x = tf.reshape(x, (b, w*h))
    pos = table[...,0] + table[...,1]*w
    pos = tf.reshape(pos, [-1])

    idx = tf.expand_dims(tf.range(b), axis=-1)
    idx = tf.reshape(tf.tile(idx, (1,w*h)),[-1])
    
    
    indices = tf.stack([idx, pos], axis=-1)
    output = tf.gather_nd(x, indices)
    output = tf.reshape(output, (b,-1))
    """
     output: (b, w*h)
    """
    return output
    
def bilinearInter(x, offset):
    """Bilinear Interplotation"""
    """
      x: (b, h, w, c) 
      offset: (b, h, w, 2*c) calculate by 1x1 convolution with 2 tuple of kernel
    """
    size = tf.shape(x)
    b = size[0]
    h = size[1]
    w = size[2]
    c = size[3]
    
    # offset:(b*c, h, w, 2)
    # x: (b*c, h, w)
    
    x,offset = reshape(x, offset)
   
    #(1, w)
    ex1 = tf.expand_dims(tf.range(w),axis=0)
    #(h, 1)
    ex2 = tf.expand_dims(tf.range(h),axis=-1)
    #(h, w)
    t1 = tf.tile(ex1,[h,1])
    t2 = tf.tile(ex2,[1,w])

    #(h, w, 2)
    g = tf.stack([t1,t2], axis=-1)
    #(1, h, w, 2)
    g = tf.expand_dims(g, axis=0)
    #(b*c, h, w, 2)
    grid = tf.tile(g, [b*c,1,1,1])
    # add the offset with grid coordinate
    coord = tf.cast(grid, 'float32') + tf.cast(offset, 'float32')
    #coord = tf.reshape(coord, [b*c, h*w, 2])
    
    coord_x = tf.clip_by_value(coord[...,0], 0, tf.cast(w, 'float32') - 1)
    coord_y = tf.clip_by_value(coord[...,1], 0, tf.cast(h, 'float32') - 1)
    coord = tf.stack([coord_x, coord_y], axis=-1)

    
    coord = tf.reshape(coord, [b*c, h*w, 2])
    assert(coord.shape[2] == 2)
    # calculate 4 vertex points
    lt_pt = tf.cast(tf.floor(coord),'int32')
    rb_pt = tf.cast(tf.ceil(coord), 'int32')
    rt_pt = tf.stack([lt_pt[...,0]+1, lt_pt[...,1]], axis=-1) # x+1,y
    lb_pt = tf.stack([lt_pt[...,0], lt_pt[...,1]+1], axis=-1) # x,y+1
    
    lt_value = warp_map(x, lt_pt)
    rb_value = warp_map(x, rb_pt)
    rt_value = warp_map(x, rt_pt)
    lb_value = warp_map(x, lb_pt)
    
    weight = coord - tf.cast(lt_pt, 'float32')
    value_up = (rt_value - lt_value)*weight[...,0] + lt_value 
    value_dn = (rb_value - lb_value)*weight[...,0] + lb_value
    value    = (value_dn - value_up)*weight[...,1] + value_up
    
    return value  


# In[ ]:


from keras.layers import Conv2D

class DeformableConv(Conv2D):
    """Deformable Convolution"""
    """
      x: must be channel_last type data
    """
    def __init__(self, filters, **kwargs):
        super(DeformableConv, self).__init__(filters=filters*2, kernel_size=(3,3), strides=1, padding='same', 
                                             use_bias=False, kernel_initializer='zeros', **kwargs)
        #super(DeformableConv, self).__init__(filters=filters*2, kernel_size=(1,1), strides=1, \
        #                                    padding='valid', use_biase=False, kernel_initializer='zeros', **kwargs)
    def build(self, input_shape):
        super(DeformableConv, self).build(input_shape)
    
    def call(self, x):
        # get offset (b,h,w,2*c)
        offset = super(DeformableConv, self).call(x)
        assert(offset.shape[3] == 2*x.shape[3])
        feature_maps = bilinearInter(x, offset)
        inshape = tf.shape(x)
        feature_maps = tf.reshape(feature_maps, inshape)
        return feature_maps
        #return offset
    
    def compute_output_shape(self, input_shape):
        return input_shape


# In[ ]:


#from __future__ import absolute_import, division
from keras.layers import Input, Conv2D, Activation, GlobalAvgPool2D, Dense, BatchNormalization
#from deform_conv.layers import ConvOffset2D


def get_cnn():
    inputs = l = Input((28, 28, 1), name='input')

    # conv11
    l = Conv2D(32, (3, 3), padding='same', name='conv11')(l)
    l = Activation('relu', name='conv11_relu')(l)
    l = BatchNormalization(name='conv11_bn')(l)

    # conv12
    l = Conv2D(64, (3, 3), padding='same', strides=(2, 2), name='conv12')(l)
    l = Activation('relu', name='conv12_relu')(l)
    l = BatchNormalization(name='conv12_bn')(l)

    # conv21
    l = Conv2D(128, (3, 3), padding='same', name='conv21')(l)
    l = Activation('relu', name='conv21_relu')(l)
    l = BatchNormalization(name='conv21_bn')(l)

    # conv22
    l = Conv2D(128, (3, 3), padding='same', strides=(2, 2), name='conv22')(l)
    l = Activation('relu', name='conv22_relu')(l)
    l = BatchNormalization(name='conv22_bn')(l)

    # out
    l = GlobalAvgPool2D(name='avg_pool')(l)
    l = Dense(10, name='fc1')(l)
    outputs = l = Activation('softmax', name='out')(l)

    return inputs, outputs


def get_deform_cnn(trainable):
    inputs = l = Input((28, 28, 1), name='input')

    # conv11
    l = Conv2D(32, (3, 3), padding='same', name='conv11', trainable=trainable)(l)
    l = Activation('relu', name='conv11_relu')(l)
    l = BatchNormalization(name='conv11_bn')(l)

    # conv12
    l_offset = DeformableConv(32, name='conv12_offset')(l)
    l = Conv2D(64, (3, 3), padding='same', strides=(2, 2), name='conv12', trainable=trainable)(l_offset)
    l = Activation('relu', name='conv12_relu')(l)
    l = BatchNormalization(name='conv12_bn')(l)

    # conv21
    l_offset = DeformableConv(64, name='conv21_offset')(l)
    l = Conv2D(128, (3, 3), padding='same', name='conv21', trainable=trainable)(l_offset)
    l = Activation('relu', name='conv21_relu')(l)
    l = BatchNormalization(name='conv21_bn')(l)

    # conv22
    l_offset = DeformableConv(128, name='conv22_offset')(l)
    l = Conv2D(128, (3, 3), padding='same', strides=(2, 2), name='conv22', trainable=trainable)(l_offset)
    l = Activation('relu', name='conv22_relu')(l)
    l = BatchNormalization(name='conv22_bn')(l)

    # out
    l = GlobalAvgPool2D(name='avg_pool')(l)
    l = Dense(10, name='fc1', trainable=trainable)(l)
    outputs = l = Activation('softmax', name='out')(l)

    return inputs, outputs

