import os

import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2

def ConvBNLayer(net, from_layer, out_layer, use_bn, use_relu, num_output,
    kernel_size, pad, stride, dilation=1, use_scale=True, lr_mult=1, MobileNet=False,
    conv_prefix='', conv_postfix='', bn_prefix='', bn_postfix='_bn', scale_prefix='', scale_postfix='_scale', bias_prefix='', bias_postfix='_bias',
    **bn_params):
  if MobileNet:
     w = 'msra'
     y = 'msra'
  else:
     w = 'xavier'
     y = 'guassian' 
  if use_bn:
    # parameters for convolution layer with batchnorm.
    kwargs = {
        'param': [dict(lr_mult=lr_mult, decay_mult=1)],
        'weight_filler': dict(type=y, std=0.01),
        'bias_term': False,
        }
    # parameters for scale bias layer after batchnorm.
    if use_scale:
      sb_kwargs = {
          'bias_term': True,
          }
    else:
      bias_kwargs = {
          'param': [dict(lr_mult=lr_mult, decay_mult=0)],
          'filler': dict(type='constant', value=0.0),
          }
  else:
    kwargs = {
        'param': [
            dict(lr_mult=lr_mult, decay_mult=1),
            dict(lr_mult=2 * lr_mult, decay_mult=0)],
        'weight_filler': dict(type=y),
        'bias_filler': dict(type='constant', value=0)
        }

  conv_name = '{}{}{}'.format(conv_prefix, out_layer, conv_postfix)
  [kernel_h, kernel_w] = UnpackVariable(kernel_size, 2)
  [pad_h, pad_w] = UnpackVariable(pad, 2)
  [stride_h, stride_w] = UnpackVariable(stride, 2)
  if kernel_h == kernel_w:
    net[conv_name] = L.Convolution(net[from_layer], num_output=num_output,
        kernel_size=kernel_h, pad=pad_h, stride=stride_h, **kwargs)
  else:
    net[conv_name] = L.Convolution(net[from_layer], num_output=num_output,
        kernel_h=kernel_h, kernel_w=kernel_w, pad_h=pad_h, pad_w=pad_w,
        stride_h=stride_h, stride_w=stride_w, **kwargs)
  if dilation > 1:
    net.update(conv_name, {'dilation': dilation})
  if use_bn:
    bn_name = '{}{}{}'.format(bn_prefix, out_layer, bn_postfix)
    net[bn_name] = L.BatchNorm(net[conv_name], in_place=True)
    if use_scale:
      sb_name = '{}{}{}'.format(scale_prefix, out_layer, scale_postfix)
      net[sb_name] = L.Scale(net[bn_name], in_place=True, **sb_kwargs)
    else:
      bias_name = '{}{}{}'.format(bias_prefix, out_layer, bias_postfix)
      net[bias_name] = L.Bias(net[bn_name], in_place=True, **bias_kwargs)
  if use_relu:
    relu_name = '{}_relu'.format(conv_name)
    net[relu_name] = L.ReLU(net[conv_name], in_place=True)

def MobileBody(net, from_layer, num_output, pad=1, stride=1, kernel_size=3, group=1,
                 conv_name='', bn_name='', scale_name='', relu_name='', freeze_layers=[]):
    kwargs = {
      'bias_term': False,
      'param': [dict(lr_mult=1, decay_mult=1)],
      'weight_filler': dict(type='msra'),
    }
    bn_kwargs = {
      'param': [
        dict(lr_mult=0, decay_mult=0),
        dict(lr_mult=0, decay_mult=0),
        dict(lr_mult=0, decay_mult=0)],
    }
    sb_kwargs = {
      'bias_term': True,
      'filler': dict(value=1.0),
      'bias_filler': dict(value=0.0),
    }
    assert from_layer in net.keys()
    if group != 1:
      net[conv_name] = L.Convolution(net[from_layer], num_output=num_output, group=group, pad=pad, stride=stride, kernel_size=kernel_size, **kwargs)
    else:
      net[conv_name] = L.Convolution(net[from_layer], num_output=num_output,
                                     pad=pad, stride=stride, kernel_size=kernel_size, **kwargs)
    net[bn_name] = L.BatchNorm(net[conv_name], in_place=True, **bn_kwargs)
    net[scale_name] = L.Scale(net[bn_name], in_place=True, **sb_kwargs)
    net[relu_name] = L.ReLU(net[scale_name], in_place=True)

def MobileNetBody(net, from_layer):
    conv_name = 'conv1'
    bn_name = '{}/bn'.format(conv_name)
    scale_name = '{}/scale'.format(conv_name)
    relu_name = 'relu1'
    assert from_layer in net.keys()
    MobileBody(net, from_layer, num_output=32, pad=1, stride=2, kernel_size=3,
               conv_name=conv_name, bn_name=bn_name, scale_name=scale_name, relu_name=relu_name)
    from_layer = relu_name
    num_output = 32
    for i in xrange(2, 5):
      for j in xrange(1, 3):
        conv_name = 'conv{}_{}/dw'.format(i, j)
        bn_name = '{}/bn'.format(conv_name)
        scale_name = '{}/scale'.format(conv_name)
        relu_name = 'relu{}_{}/dw'.format(i, j)
        if j == 2:
          stride = 2
        else:
          stride = 1
        MobileBody(net, from_layer, num_output=num_output, pad=1, stride=stride, kernel_size=3, group=num_output,
                   conv_name=conv_name, bn_name=bn_name, scale_name=scale_name, relu_name=relu_name)
        from_layer = relu_name
        conv_name = 'conv{}_{}/sep'.format(i, j)
        bn_name = '{}/bn'.format(conv_name)
        scale_name = '{}/scale'.format(conv_name)
        relu_name = 'relu{}_{}/sep'.format(i, j)
        if num_output == 32 or j == 2:
          num_output = num_output * 2
        MobileBody(net, from_layer, num_output=num_output, pad=0, stride=1, kernel_size=1,
                   conv_name=conv_name, bn_name=bn_name, scale_name=scale_name, relu_name=relu_name)
        from_layer = relu_name
    # conv5
    for j in xrange(1, 7):
      conv_name = 'conv{}_{}/dw'.format(5, j)
      bn_name = '{}/bn'.format(conv_name)
      scale_name = '{}/scale'.format(conv_name)
      relu_name = 'relu{}_{}/dw'.format(5, j)
      if j == 6:
        stride = 2
      else:
        stride = 1
      MobileBody(net, from_layer, num_output=num_output, pad=1, stride=stride, kernel_size=3, group=num_output,
                 conv_name=conv_name, bn_name=bn_name, scale_name=scale_name, relu_name=relu_name)
      from_layer = relu_name
      conv_name = 'conv{}_{}/sep'.format(5, j)
      bn_name = '{}/bn'.format(conv_name)
      scale_name = '{}/scale'.format(conv_name)
      relu_name = 'relu{}_{}/sep'.format(5, j)
      if j == 6:
        num_output = num_output * 2
      MobileBody(net, from_layer, num_output=num_output, pad=0, stride=1, kernel_size=1,
                 conv_name=conv_name, bn_name=bn_name, scale_name=scale_name, relu_name=relu_name)
      from_layer = relu_name
    # conv6
    conv_name = 'conv6/dw'
    bn_name = '{}/bn'.format(conv_name)
    scale_name = '{}/scale'.format(conv_name)
    relu_name = 'relu6/dw'

    MobileBody(net, from_layer, num_output=num_output, pad=1, stride=1, kernel_size=3, group=num_output,
               conv_name=conv_name, bn_name=bn_name, scale_name=scale_name, relu_name=relu_name)
    from_layer = relu_name
    conv_name = 'conv6/sep'
    bn_name = '{}/bn'.format(conv_name)
    scale_name = '{}/scale'.format(conv_name)
    relu_name = 'relu6/sep'
    MobileBody(net, from_layer, num_output=num_output, pad=0, stride=1, kernel_size=1,
               conv_name=conv_name, bn_name=bn_name, scale_name=scale_name, relu_name=relu_name)

    return net
if __name__ == '__main__':
  net = caffe.NetSpec()
  net.data = L.Input(shape=[dict(dim=[1, 3, 224, 224])])
  MobileNetBody(net, from_layer='data')
  print(net.to_proto())


