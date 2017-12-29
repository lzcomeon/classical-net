import collections
import tensorflow as tf
slim = tf.contrib.slim

class Block(collections.namedtuple('Block',['scope','unit_fn','args'])):
    '''
    用namedtuple创建Block的类，但只包含数据结构，不包含具体方法
    需要定义一个典型的Block,需要输入三个参数，分别是scope,unit_fn和args。
    以Block('block',bottleneck,[(256,64,1)]x2+[(256,64,2)])这一行代码为例，
    它可以定义一个典型的Block.其中block1就是这个Block的名称；bottleneck是ResNet V2
    中的残差学习单元，而最后一个参数[(256,64,1)]x2+[(256,64,2)]则是这个Block的args
    args是一个列表，其中每一个元素都对应一个bottleneck残差学习单元,前面两个元素都是
    (256,64,1),最后一个是(256,64,2)。每一个元素都是一个三元tuple，即(depth,depth_bottleneck,stride)

    '''
    'A named tuple describing a ResNet block.'

def subsample(inputs,factor,scope = None):
    if factor == 1:
        return inputs
    else:
        return slim.max_pool2d(inputs,[1,1],stride = factor,scope = scope)

#定义conv2d_same函数创建卷积层
def conv2d_same(inputs,num_outputs,kernel_size,stride,scope = None):
    if stride == 1:
        return slim.conv2d(inputs,num_outputs,kernel_size,stride = 1,padding = 'SAME',scope = scope)
    else:
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(inputs,[[0,0],[pad_beg,pad_end],[pad_beg,pad_end],[0,0]])
        return slim.conv2d(inputs,num_outputs,kernel_size,stride = stride,
                           padding = 'VALID',scope = scope)

@slim.add_arg_scope
def stack_blocks_dense(net,blocks,outputs_collections = None):
    for block in blocks:
        with tf.variable_scope(block.scope,'block',[net])
