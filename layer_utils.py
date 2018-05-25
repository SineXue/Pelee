import os, math

import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
from google.protobuf import text_format


def sfd_block(net):

  kwargs_1 = {
          'param': [
              dict(lr_mult=1, decay_mult=1)],
'convolution_param': dict(kernel_size=3,
                          stride=1,
                          group=288,
                          num_output=288,
                          pad=1,
                          bias_filler=dict(type='constant'),
                          weight_filler=dict(type='xavier'))
          }
  kwargs_2 = {
          'param': [
              dict(lr_mult=1, decay_mult=1)],
'convolution_param': dict(kernel_size=3,
                          stride=1,
                          group=192,
                          num_output=192,
                          pad=1,
                          bias_filler=dict(type='constant'),
                          weight_filler=dict(type='xavier'))
          }
  bn_kwargs = {
          'param': [
              dict(lr_mult=0, decay_mult=0),
              dict(lr_mult=0, decay_mult=0),
              dict(lr_mult=0, decay_mult=0)],
          'eps': 0.001,
          'moving_average_fraction': 0.999
          }
  sb_kwargs = {
          'bias_term': True,
          'filler': dict(value=1.0),
          'bias_filler': dict(value=0.0),
          }
  net.upP3 = L.Deconvolution(net['stage3_1/concat'], convolution_param=dict(kernel_size=4, stride=2, group=288, num_output=288, pad=1, bias_term=False, weight_filler=dict(type='bilinear')), param=[dict(lr_mult=0, decay_mult=0)])
  # add dw
  net.upP3_dw = L.Convolution(net.upP5, **kwargs_1)
  net.unP3_dw_bn = L.BatchNorm(net.upP3_dw, in_place=True, **bn_kwargs)
  net.unP3_dw_scale = L.Scale(net.unP3_dw_bn, in_place=True, **sb_kwargs)
  net.upP3_dw_relu = L.ReLU(net.unP3_dw_scale, in_place=True)
  # add slice
  net.upP3_left, net.upP3_right = L.Slice(net.upP3_dw, name='upP3_slice', ntop=2, slice_param=dict(axis=1, slice_point=144))
  # add max
  net.upP3_maxout = L.Eltwise(net.upP3_left, net.upP3_right, eltwise_param=dict(operation=2))
  net.p2 = L.Eltwise(net['stage2_2/concat'], net.upP3_maxout)
  net.upP2 = L.Deconvolution(net.p2, convolution_param=dict(kernel_size=4, stride=2, group=192, num_output=192, pad=1, bias_term=False, weight_filler=dict(type='bilinear')), param=[dict(lr_mult=0, decay_mult=0)])
  # add dw
  net.upP2_dw = L.Convolution(net.upP2, **kwargs_2)
  net.unP2_dw_bn = L.BatchNorm(net.upP2_dw, in_place=True, **bn_kwargs)
  net.unP2_dw_scale = L.Scale(net.unP2_dw_bn, in_place=True, **sb_kwargs)
  net.upP2_dw_relu = L.ReLU(net.unP2_dw_scale, in_place=True)
  # add slice
  net.upP2_left, net.upP2_right = L.Slice(net.upP2_dw, name='upP2_slice', ntop=2, slice_param=dict(axis=1, slice_point=96))
  # add max
  net.upP2_maxout = L.Eltwise(net.upP2_left, net.upP2_right, eltwise_param=dict(operation=2))
  net.p1 = L.Eltwise(net['stage1_tb'], net.upP2_maxout)

  return net



def fpn_block(net):
  use_relu=False
  use_bn=False
  ConvBNLayer(net, 'stage2_tb', 'newC1', use_bn, use_relu, 128, 1, 0, 1)
  ConvBNLayer(net, 'stage3_tb', 'newC2', use_bn, use_relu, 128, 1, 0, 1)
  ConvBNLayer(net, 'stage4_tb', 'newC3', use_bn, use_relu, 128, 1, 0, 1)
  ConvBNLayer(net, 'ext1/fe1_2', 'p4', use_bn, use_relu, 128, 1, 0, 1)
  net.upP4 = L.Deconvolution(net.p4, convolution_param=dict(kernel_size=4, stride=2, group=128, num_output=128, pad=1, bias_term=False, weight_filler=dict(type='bilinear')), param=[dict(lr_mult=0, decay_mult=0)])
  net.p3 = L.Eltwise(net['newC3'], net.upP4)
  ConvBNLayer(net, 'p3', 'p3_lateral', use_bn, use_relu, 128, 1, 0, 1)
  net.upP3 = L.Deconvolution(net.p3_lateral, convolution_param=dict(kernel_size=4, stride=2, group=128, num_output=128, pad=1, bias_term=False, weight_filler=dict(type='bilinear')),param=[dict(lr_mult=0, decay_mult=0)])
  net.p2 = L.Eltwise(net['newC2'], net.upP3)
  ConvBNLayer(net, 'p2', 'p2_lateral', use_bn, use_relu, 128, 1, 0, 1)
  net.upP2 = L.Deconvolution(net.p2_lateral, convolution_param=dict(kernel_size=4, stride=2, group=128, num_output=128, pad=1, bias_term=False, weight_filler=dict(type='bilinear')),param=[dict(lr_mult=0, decay_mult=0)])
  net.p1 = L.Eltwise(net['newC1'], net.upP2)
  return net


def res_block(net, from_layer, num_filter, block_id, bottleneck_fact=0.5, stride=2, pad=1, use_bn=True):

  branch1 = '{}'.format(block_id)
  ConvBNLayer(net, from_layer, branch1, use_bn=use_bn, use_relu=False, num_output=num_filter, kernel_size=1, pad=0, stride=stride)


  branch2a = '{}/b2a'.format(block_id)
  ConvBNLayer(net, from_layer, branch2a, use_bn=use_bn, use_relu=True, num_output=int(num_filter*bottleneck_fact), kernel_size=1, pad=0, stride=1)

  branch2b = '{}/b2b'.format(block_id)
  ConvBNLayer(net, branch2a, branch2b, use_bn=use_bn, use_relu=True, num_output=int(num_filter*bottleneck_fact), kernel_size=3, pad=pad, stride=stride)

  branch2c = '{}/b2c'.format(block_id)
  ConvBNLayer(net, branch2b, branch2c, use_bn=use_bn, use_relu=False, num_output=num_filter, kernel_size=1, pad=0, stride=1)

  res_name = '{}/res'.format(block_id)
  net[res_name] = L.Eltwise(net[branch1], net[branch2c])
  relu_name = '{}/relu'.format(res_name)
  net[relu_name] = L.ReLU(net[res_name], in_place=True)

  return relu_name

def ConvBNLayer(net, from_layer, out_layer, use_bn, use_relu, num_output,
    kernel_size, pad, stride, dilation=1, use_scale=True, lr_mult=1,
    conv_prefix='', conv_postfix='', bn_prefix='', bn_postfix='/bn',
    scale_prefix='', scale_postfix='/scale', bias_prefix='', bias_postfix='/bias',
    **bn_params):
  if use_bn:
    # parameters for convolution layer with batchnorm. weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')
    kwargs = {
        'param': [dict(lr_mult=lr_mult, decay_mult=1)],
        'weight_filler': dict(type='xavier'),
        'bias_term': False,
        }
    eps = bn_params.get('eps', 0.001)
    moving_average_fraction = bn_params.get('moving_average_fraction', 0.999)
    #moving_average_fraction = bn_params.get('moving_average_fraction', 0.1)
    use_global_stats = bn_params.get('use_global_stats', False)
    # parameters for batchnorm layer.
    bn_kwargs = {
        'param': [
            dict(lr_mult=0, decay_mult=0),
            dict(lr_mult=0, decay_mult=0),
            dict(lr_mult=0, decay_mult=0)],
        'eps': eps,
        'moving_average_fraction': moving_average_fraction,
        }
    bn_lr_mult = lr_mult
    if use_global_stats:
      # only specify if use_global_stats is explicitly provided;
      # otherwise, use_global_stats_ = this->phase_ == TEST;
      bn_kwargs = {
          'param': [
              dict(lr_mult=0, decay_mult=0),
              dict(lr_mult=0, decay_mult=0),
              dict(lr_mult=0, decay_mult=0)],
          'eps': eps,
          'use_global_stats': use_global_stats,
          }
      # not updating scale/bias parameters
      bn_lr_mult = 0
    # parameters for scale bias layer after batchnorm.
    if use_scale:
      sb_kwargs = {
          'bias_term': True,
          'param': [
              dict(lr_mult=bn_lr_mult, decay_mult=0),
              dict(lr_mult=bn_lr_mult, decay_mult=0)],
          'filler': dict(type='constant', value=1.0),
          'bias_filler': dict(type='constant', value=0.0),
          }
    else:
      bias_kwargs = {
          'param': [dict(lr_mult=bn_lr_mult, decay_mult=0)],
          'filler': dict(type='constant', value=0.0),
          }
  else:
    kwargs = {
        'param': [
            dict(lr_mult=lr_mult, decay_mult=1),
            dict(lr_mult=2 * lr_mult, decay_mult=0)],
        'weight_filler': dict(type='xavier'),
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
    net[bn_name] = L.BatchNorm(net[conv_name], in_place=True, **bn_kwargs)
    if use_scale:
      sb_name = '{}{}{}'.format(scale_prefix, out_layer, scale_postfix)
      net[sb_name] = L.Scale(net[bn_name], in_place=True, **sb_kwargs)
    else:
      bias_name = '{}{}{}'.format(bias_prefix, out_layer, bias_postfix)
      net[bias_name] = L.Bias(net[bn_name], in_place=True, **bias_kwargs)
  if use_relu:
    relu_name = '{}/relu'.format(conv_name)
    net[relu_name] = L.ReLU(net[conv_name], in_place=True)



def UnpackVariable(var, num):
  assert len > 0
  if type(var) is list and len(var) == num:
    return var
  else:
    ret = []
    if type(var) is list:
      assert len(var) == 1
      for i in xrange(0, num):
        ret.append(var[0])
    else:
      for i in xrange(0, num):
        ret.append(var)
    return ret
