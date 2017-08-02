import tensorflow as tf
from utils.ops import *


#------------------------network setting---------------------

def cpm_stage1(image, name):
  with tf.variable_scope(name) as scope:
      conv1_1 = conv2d(image, 64, 3, 1, relu=True, name='conv1_1')
      conv1_2 = conv2d(conv1_1, 64, 3, 1, relu=True, name='conv1_2')
      pool1 = max_pool(conv1_2, 2, 2, 'pool1')

      conv2_1 = conv2d(pool1, 128, 3, 1, relu=True, name='conv2_1')
      conv2_2 = conv2d(conv2_1, 128, 3, 1, relu=True, name='conv2_2')
      pool2 = max_pool(conv2_2, 2, 2, 'pool2')

      conv3_1 = conv2d(pool2, 256, 3, 1, relu=True, name='conv3_1')
      conv3_2 = conv2d(conv3_1, 256, 3, 1, relu=True, name='conv3_2')
      conv3_3 = conv2d(conv3_2, 256, 3, 1, relu=True, name='conv3_3')
      conv3_4 = conv2d(conv3_3, 256, 3, 1, relu=True, name='conv3_4')
      pool3 = max_pool(conv3_4, 2, 2, 'pool3')

      conv4_1 = conv2d(pool3, 512, 3, 1, relu=True, name='conv4_1')
      conv4_2 = conv2d(conv4_1, 512, 3, 1, relu=True, name='conv4_2')
      conv4_3 = conv2d(conv4_2, 256, 3, 1, relu=True, name='conv4_3')
      conv4_4 = conv2d(conv4_3, 256, 3, 1, relu=True, name='conv4_4')
      conv4_5 = conv2d(conv4_4, 256, 3, 1, relu=True, name='conv4_5')
      conv4_6 = conv2d(conv4_5, 256, 3, 1, relu=True, name='conv4_6')
      conv4_7 = conv2d(conv4_6, 128, 3, 1, relu=True, name='conv4_7')

      conv5_1 = conv2d(conv4_7, 512, 1, 1, relu=True, name='conv5_1')
      conv5_2 = conv2d(conv5_1, 16, 1, 1, relu=False, name='conv5_2')

      return conv5_2, conv4_7

def cpm_stage_x(image, name):
  with tf.variable_scope(name) as scope:
      conv1 = conv2d(image, 128, 7, 1, relu=True, name='conv1')
      conv2 = conv2d(conv1, 128, 7, 1, relu=True, name='conv2')
      conv3 = conv2d(conv2, 128, 7, 1, relu=True, name='conv3')
      conv4 = conv2d(conv3, 128, 7, 1, relu=True, name='conv4')
      conv5 = conv2d(conv4, 128, 7, 1, relu=True, name='conv5')
      conv6 = conv2d(conv5, 128, 1, 1, relu=True, name='conv6')
      conv7 = conv2d(conv6, 16, 1, 1, relu=False, name='conv7')

      return conv7

## --------------------------------------------------------------------------------

##  version 1

# def pose_net(image, name):
#   with tf.variable_scope(name) as scope:
#       pose_conv1 = conv2d(image, 512, 3, 1, relu=True, name='pose_conv1')
#       pose_conv2 = conv2d(pose_conv1, 512, 3, 1, relu=True, name='pose_conv2')
#       pose_conv3 = conv2d(pose_conv2, 256, 3, 1, relu=True, name='pose_conv3')
#       pose_conv4 = conv2d(pose_conv3, 256, 3, 1, relu=True, name='pose_conv4')
#       pose_conv5 = conv2d(pose_conv4, 128, 3, 1, relu=True, name='pose_conv5')
#       pose_conv6 = conv2d(pose_conv5, 128, 3, 1, relu=True, name='pose_conv6')
      
#       pose_conv7 = conv2d(pose_conv6, 512, 1, 1, relu=True, name='pose_conv7')
#       pose_conv8 = conv2d(pose_conv7, 16, 1, 1, relu=False, name='pose_conv8')
#       return pose_conv8, pose_conv6

# def parsing_refine_x(image, name):
#   with tf.variable_scope(name) as scope:
#       parsing_conv1 = conv2d(image, 128, 7, 1, relu=True, name='parsing_conv1')
#       parsing_conv2 = conv2d(parsing_conv1, 256, 7, 1, relu=True, name='parsing_conv2')
#       parsing_conv3 = conv2d(parsing_conv2, 512, 7, 1, relu=True, name='parsing_conv3')
#       parsing_conv4 = conv2d(parsing_conv3, 512, 7, 1, relu=True, name='parsing_conv4')
#       parsing_conv5 = conv2d(parsing_conv4, 1024, 7, 1, relu=True, name='parsing_conv5')
#       parsing_human1 = atrous_conv2d(parsing_conv5, 20, 3, rate=6, relu=False, name='parsing_human1')
#       parsing_human2 = atrous_conv2d(parsing_conv5, 20, 3, rate=12, relu=False, name='parsing_human2')
#       parsing_human3 = atrous_conv2d(parsing_conv5, 20, 3, rate=18, relu=False, name='parsing_human3')
#       parsing_human4 = atrous_conv2d(parsing_conv5, 20, 3, rate=24, relu=False, name='parsing_human4')
#       parsing_human = tf.add_n([parsing_human1, parsing_human2, parsing_human3, parsing_human4], name='parsing_human')

#       return parsing_human

###########################################
##
##  version 2
## modify to use combined feature.  05.27

# def pose_net2(image, name):
#   with tf.variable_scope(name) as scope:
#       pose_conv1 = conv2d(image, 512, 3, 1, relu=True, name='pose_conv1')
#       pose_conv2 = conv2d(pose_conv1, 512, 3, 1, relu=True, name='pose_conv2')
#       pose_conv3 = conv2d(pose_conv2, 256, 3, 1, relu=True, name='pose_conv3')
#       pose_conv4 = conv2d(pose_conv3, 256, 3, 1, relu=True, name='pose_conv4')
#       pose_conv5 = conv2d(pose_conv4, 128, 3, 1, relu=True, name='pose_conv5')
#       pose_conv6 = conv2d(pose_conv5, 128, 3, 1, relu=True, name='pose_conv6')
      
#       pose_conv7 = conv2d(pose_conv6, 512, 1, 1, relu=True, name='pose_conv7')
#       pose_conv8 = conv2d(pose_conv7, 16, 1, 1, relu=False, name='pose_conv8')
#       return pose_conv8, pose_conv6

# def pose_refine_2(image, name):
#   with tf.variable_scope(name) as scope:
#       conv1 = conv2d(image, 256, 7, 1, relu=True, name='conv1')
#       conv2 = conv2d(conv1, 256, 7, 1, relu=True, name='conv2')
#       conv3 = conv2d(conv2, 256, 7, 1, relu=True, name='conv3')
#       conv4 = conv2d(conv3, 256, 7, 1, relu=True, name='conv4')
#       conv5 = conv2d(conv4, 256, 1, 1, relu=True, name='conv5')
#       conv6 = conv2d(conv5, 16, 1, 1, relu=False, name='conv6')

#       return conv6

# def parsing_refine_2(image, name):
#   with tf.variable_scope(name) as scope:
#       parsing_conv1 = conv2d(image, 256, 7, 1, relu=True, name='parsing_conv1')
#       parsing_conv2 = conv2d(parsing_conv1, 256, 7, 1, relu=True, name='parsing_conv2')
#       parsing_conv3 = conv2d(parsing_conv2, 512, 7, 1, relu=True, name='parsing_conv3')
#       parsing_conv4 = conv2d(parsing_conv3, 512, 7, 1, relu=True, name='parsing_conv4')
#       parsing_conv5 = conv2d(parsing_conv4, 1024, 1, 1, relu=True, name='parsing_conv5')
#       parsing_human1 = atrous_conv2d(parsing_conv5, 20, 3, rate=6, relu=False, name='parsing_human1')
#       parsing_human2 = atrous_conv2d(parsing_conv5, 20, 3, rate=12, relu=False, name='parsing_human2')
#       parsing_human3 = atrous_conv2d(parsing_conv5, 20, 3, rate=18, relu=False, name='parsing_human3')
#       parsing_human4 = atrous_conv2d(parsing_conv5, 20, 3, rate=24, relu=False, name='parsing_human4')
#       parsing_human = tf.add_n([parsing_human1, parsing_human2, parsing_human3, parsing_human4], name='parsing_human')

#       return parsing_human

## end modifying
#################################################

##  refine net version 3.   07.04

# def pose_net(image, name):
#   with tf.variable_scope(name) as scope:
#       is_BN = False
#       pose_conv1 = conv2d(image, 512, 3, 1, relu=True, bn=is_BN, name='pose_conv1')
#       pose_conv2 = conv2d(pose_conv1, 512, 3, 1, relu=True, bn=is_BN, name='pose_conv2')
#       pose_conv3 = conv2d(pose_conv2, 256, 3, 1, relu=True, bn=is_BN, name='pose_conv3')
#       pose_conv4 = conv2d(pose_conv3, 256, 3, 1, relu=True, bn=is_BN, name='pose_conv4')
#       pose_conv5 = conv2d(pose_conv4, 128, 3, 1, relu=True, bn=is_BN, name='pose_conv5')
#       pose_conv6 = conv2d(pose_conv5, 128, 3, 1, relu=True, bn=is_BN, name='pose_conv6')

#       pose_conv7 = conv2d(pose_conv6, 512, 1, 1, relu=True, bn=is_BN, name='pose_conv7')
#       pose_conv8 = conv2d(pose_conv7, 16, 1, 1, relu=False, bn=is_BN, name='pose_conv8')

#       return pose_conv8, pose_conv6

# def pose_refine(pose, pose_fea, parsing_fea, name):
#   with tf.variable_scope(name) as scope:
#       is_BN = False
#       # 1*1 convolution remaps the heatmaps to match the number of channels of the intermediate features.
#       pose = conv2d(pose, 256, 1, 1, relu=True, bn=is_BN, name='remap')
#       # concat instead of sum
#       pos_par = tf.concat([pose, pose_fea, parsing_fea], 3)
#       # conv1 = conv2d(pos_par, 512, 7, 1, relu=True, bn=True, name='conv1')
#       # gcn_res2 = gcn_residual_module(conv1, 512, 15, name='gcn_res2')
#       # gcn_res3 = gcn_residual_module(gcn_res2, 256, 15, name='gcn_res3')
#       # gcn_res4 = gcn_residual_module(gcn_res3, 256, 15, name='gcn_res4')
#       # conv5 = residual_module(gcn_res4, 256, name='conv5')
#       # conv6 = conv2d(conv5, 16, 1, 1, relu=False, bn=False, name='conv6')
#       conv1 = conv2d(pos_par, 512, 7, 1, relu=True, bn=is_BN, name='conv1')
#       gcn_res2 = gcn_residual_module(conv1, 512, 15, is_BN=is_BN, name='gcn_res2')
#       conv3 = conv2d(gcn_res2, 512, 1, 1, relu=True, bn=is_BN, name='conv3')
#       conv4 = conv2d(conv3, 16, 1, 1, relu=False, bn=False, name='conv4')
#       return conv4

# ##  idea of large kernel matters

# def parsing_refine(parsing, parsing_fea, pose_fea, name):
#   with tf.variable_scope(name) as scope:
#       is_BN = False
#       parsing = conv2d(parsing, 256, 1, 1, relu=True, bn=is_BN, name='remap')
#       par_pos = tf.concat([parsing, parsing_fea, pose_fea], 3)
#       parsing_conv1 = conv2d(par_pos, 512, 3, 1, relu=True, bn=is_BN, name='parsing_conv1')
#       parsing_conv2 = conv2d(parsing_conv1, 512, 3, 1, relu=True, bn=is_BN, name='parsing_conv2')
#       #  gcn kernel size = 15
#       parsing_gcn = gcn(parsing_conv2, 20, 15, 1, relu=False, bn=False, name='parsing_gcn')
#       parsing_br = br(parsing_gcn, 20, 3, 1, name='parsing_br')
      
#       return parsing_br


#################################################


#################################################

##  refine net version 4.   07.17

# def pose_net(image, name):
#   with tf.variable_scope(name) as scope:
#       is_BN = False
#       pose_conv1 = conv2d(image, 512, 3, 1, relu=True, bn=is_BN, name='pose_conv1')
#       pose_conv2 = conv2d(pose_conv1, 512, 3, 1, relu=True, bn=is_BN, name='pose_conv2')
#       pose_conv3 = conv2d(pose_conv2, 256, 3, 1, relu=True, bn=is_BN, name='pose_conv3')
#       pose_conv4 = conv2d(pose_conv3, 256, 3, 1, relu=True, bn=is_BN, name='pose_conv4')
#       pose_conv5 = conv2d(pose_conv4, 256, 3, 1, relu=True, bn=is_BN, name='pose_conv5')
#       pose_conv6 = conv2d(pose_conv5, 256, 3, 1, relu=True, bn=is_BN, name='pose_conv6')

#       pose_conv7 = conv2d(pose_conv6, 512, 1, 1, relu=True, bn=is_BN, name='pose_conv7')
#       pose_conv8 = conv2d(pose_conv7, 16, 1, 1, relu=False, bn=is_BN, name='pose_conv8')

#       return pose_conv8, pose_conv6


# def pose_refine(pose, parsing, pose_fea, name):
#   with tf.variable_scope(name) as scope:
#       is_BN = False
#       # 1*1 convolution remaps the heatmaps to match the number of channels of the intermediate features.
#       pose = conv2d(pose, 128, 1, 1, relu=True, bn=is_BN, name='pose_remap')
#       parsing = conv2d(parsing, 128, 1, 1, relu=True, bn=is_BN, name='parsing_remap')
#       # concat 
#       pos_par = tf.concat([pose, parsing, pose_fea], 3)
#       conv1 = conv2d(pos_par, 512, 3, 1, relu=True, bn=is_BN, name='conv1')
#       conv2 = conv2d(conv1, 256, 5, 1, relu=True, bn=is_BN, name='conv2')
#       conv3 = conv2d(conv2, 256, 7, 1, relu=True, bn=is_BN, name='conv3')
#       conv4 = conv2d(conv3, 256, 9, 1, relu=True, bn=is_BN, name='conv4')
#       conv5 = conv2d(conv4, 256, 1, 1, relu=True, bn=is_BN, name='conv5')
#       conv6 = conv2d(conv5, 16, 1, 1, relu=False, bn=is_BN, name='conv6')
#       return conv6


# def parsing_refine(parsing, pose, parsing_fea, name):
#   with tf.variable_scope(name) as scope:
#       is_BN = False
#       pose = conv2d(pose, 128, 1, 1, relu=True, bn=is_BN, name='pose_remap')
#       parsing = conv2d(parsing, 128, 1, 1, relu=True, bn=is_BN, name='parsing_remap')

#       par_pos = tf.concat([parsing, pose, parsing_fea], 3)
#       parsing_conv1 = conv2d(par_pos, 512, 3, 1, relu=True, bn=is_BN, name='parsing_conv1')
#       parsing_conv2 = conv2d(parsing_conv1, 256, 5, 1, relu=True, bn=is_BN, name='parsing_conv2')
#       parsing_conv3 = conv2d(parsing_conv2, 256, 7, 1, relu=True, bn=is_BN, name='parsing_conv3')
#       parsing_conv4 = conv2d(parsing_conv3, 256, 9, 1, relu=True, bn=is_BN, name='parsing_conv4')
#       parsing_conv5 = conv2d(parsing_conv4, 256, 1, 1, relu=True, bn=is_BN, name='parsing_conv5')
#       parsing_human1 = atrous_conv2d(parsing_conv5, 20, 3, rate=6, relu=False, name='parsing_human1')
#       parsing_human2 = atrous_conv2d(parsing_conv5, 20, 3, rate=12, relu=False, name='parsing_human2')
#       parsing_human3 = atrous_conv2d(parsing_conv5, 20, 3, rate=18, relu=False, name='parsing_human3')
#       parsing_human4 = atrous_conv2d(parsing_conv5, 20, 3, rate=24, relu=False, name='parsing_human4')
#       parsing_human = tf.add_n([parsing_human1, parsing_human2, parsing_human3, parsing_human4], name='parsing_human')
#       return parsing_human
#################################################

def pose_net(image, name):
  with tf.variable_scope(name) as scope:
      is_BN = False

      res0 = residual_module(image, 256, is_BN, name='res0')
      pool1 = max_pool(image, 2, 2, name='pool1')
      res1 = residual_module(pool1, 256, is_BN, name='res1')
      
      res2_0 = residual_module(res1, 256, is_BN, name='res2_0')
      pool2 = max_pool(res1, 2, 2, name='pool2')
      res2_1 = residual_module(pool2, 256, is_BN, name='res2_1')
      res2_2 = residual_module(res2_1, 256, is_BN, name='res2_2')
      res2_3 = residual_module(res2_2, 256, is_BN, name='res2_3')
      upsample1 = tf.image.resize_nearest_neighbor(res2_3, tf.shape(res2_3)[1:3]*2, name = 'upsampling1')
      hg1 = tf.nn.relu(tf.add_n([upsample1, res2_0]), name='hg1')

      res3 = residual_module(hg1, 256, is_BN, name='res3')
      upsample2 = tf.image.resize_nearest_neighbor(res3, tf.shape(res3)[1:3]*2, name = 'upsampling2')
      hg2 = tf.nn.relu(tf.add_n([upsample2, res0]), name='hg2')

      linear = conv2d(hg2, 256, 1, 1, relu=True, bn=is_BN, name='linear')
      output = conv2d(linear, 16, 1, 1, relu=False, bn=is_BN, name='output')

      return output, linear

def pose_refine(pose, parsing, pose_fea, name):
  with tf.variable_scope(name) as scope:
      is_BN = False
      # 1*1 convolution remaps the heatmaps to match the number of channels of the intermediate features.
      pose = conv2d(pose, 128, 1, 1, relu=True, bn=is_BN, name='pose_remap')
      parsing = conv2d(parsing, 128, 1, 1, relu=True, bn=is_BN, name='parsing_remap')
      # concat 
      pos_par = tf.concat([pose, parsing, pose_fea], 3)
      conv1 = conv2d(pos_par, 512, 3, 1, relu=True, bn=is_BN, name='conv1')
      conv2 = conv2d(conv1, 256, 5, 1, relu=True, bn=is_BN, name='conv2')
      conv3 = conv2d(conv2, 256, 7, 1, relu=True, bn=is_BN, name='conv3')
      conv4 = conv2d(conv3, 256, 9, 1, relu=True, bn=is_BN, name='conv4')
      conv5 = conv2d(conv4, 256, 1, 1, relu=True, bn=is_BN, name='conv5')
      conv6 = conv2d(conv5, 16, 1, 1, relu=False, bn=is_BN, name='conv6')
      return conv6

def parsing_refine(parsing, pose, parsing_fea, name):
  with tf.variable_scope(name) as scope:
      is_BN = False
      pose = conv2d(pose, 128, 1, 1, relu=True, bn=is_BN, name='pose_remap')
      parsing = conv2d(parsing, 128, 1, 1, relu=True, bn=is_BN, name='parsing_remap')

      par_pos = tf.concat([parsing, pose, parsing_fea], 3)
      conv1 = conv2d(par_pos, 512, 3, 1, relu=True, bn=is_BN, name='conv1')
      conv2 = conv2d(conv1, 256, 5, 1, relu=True, bn=is_BN, name='conv2')
      conv3 = conv2d(conv2, 256, 7, 1, relu=True, bn=is_BN, name='conv3')
      conv4 = conv2d(conv3, 256, 9, 1, relu=True, bn=is_BN, name='conv4')
      conv5 = conv2d(conv4, 256, 11, 1, relu=True, bn=is_BN, name='conv5')

      global_pool = pool1 = avg_pool(conv5, 48, 48, name='global_pool')
      conv6 = conv2d(global_pool, 256, 1, 1, relu=True, bn=is_BN, name='conv6')
      upsample = tf.image.resize_images(conv6, tf.shape(conv5)[1:3])

      human1 = atrous_conv2d(conv5, 20, 3, rate=6, relu=True, name='human1')
      human2 = atrous_conv2d(conv5, 20, 3, rate=12, relu=True, name='human2')
      human3 = atrous_conv2d(conv5, 20, 3, rate=18, relu=True, name='human3')
      human4 = conv2d(conv5, 20, 1, 1, relu=True, bn=is_BN, name='human4')

      aspp = tf.concat([upsample, human1, human2, human3, human4], 3)
      conv7 = conv2d(aspp, 256, 1, 1, relu=True, bn=is_BN, name='conv7')
      conv8 = conv2d(conv7, 20, 1, 1, relu=False, bn=is_BN, name='conv8')

      return conv8