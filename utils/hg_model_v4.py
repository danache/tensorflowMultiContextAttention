# Converted to TensorFlow .caffemodel
# with the DeepLab-ResNet configuration.
# The batch normalisation layer is provided by
# the slim library
# (https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim).

from kaffe.tensorflow import Network
import tensorflow as tf
import numpy as np


class HGedHourglassModel(Network):

    def setup(self, is_training, n_classes):
        '''Network definition.

        Args:
          is_training: whether to update the running mean and variance of the batch normalisation layer.
                       If the batch size is small, it is better to keep the running mean and variance of 
                       the-pretrained model frozen.
        '''
        # 384x384-->192x192
        (self.feed('data')
             # .pad([[0,0], [2,2], [2,2], [0,0]], name='pad_1')
             .conv(7, 7, 64, 1, 1, biased=True, relu=False, name='conv1_')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='conv1__')
             .relu(name='conv1')
         # .max_pool(3, 3, 2, 2, name='pool1')
         # .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='res2a_branch1')
         # .tf_batch_normalization(is_training=is_training, activation_fn=None, name='bn2a_branch1')
         )
# res1
        (self.feed('conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='res1_batch1')
             .relu(name='res1_relu1')
             .conv(1, 1, 32, 1, 1, biased=True, relu=False, name='res1_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='res1_batch2')
             .relu(name='res1_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 32, 1, 1, biased=True, relu=False, name='res1_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='res1_batch3')
             .relu(name='res1_relu3')
             .conv(1, 1, 64, 1, 1, biased=True, relu=False, name='res1_conv3'))

        (self.feed('conv1')
             .conv(1, 1, 64, 1, 1, biased=True, relu=False, name='res1_skip'))
        # 192x192-->96x96
        (self.feed('res1_conv3',
                   'res1_skip')
         .add(name='Res1'))

# pool1
        (self.feed('Res1')
         .max_pool(2, 2, 2, 2, name='pool1'))

# res2
        (self.feed('pool1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='res2_batch1')
             .relu(name='res2_relu1')
             .conv(1, 1, 32, 1, 1, biased=True, relu=False, name='res2_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='res2_batch2')
             .relu(name='res2_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 32, 1, 1, biased=True, relu=False, name='res2_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='res2_batch3')
             .relu(name='res2_relu3')
             .conv(1, 1, 64, 1, 1, biased=True, relu=False, name='res2_conv3'))

        (self.feed('pool1')
             .conv(1, 1, 64, 1, 1, biased=True, relu=False, name='res2_skip'))

        (self.feed('res2_conv3',
                   'res2_skip')
         .add(name='Res2'))
# res3
        (self.feed('Res2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='res3_batch1')
             .relu(name='res3_relu1')
             .conv(1, 1, 64, 1, 1, biased=True, relu=False, name='res3_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='res3_batch2')
             .relu(name='res3_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 64, 1, 1, biased=True, relu=False, name='res3_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='res3_batch3')
             .relu(name='res3_relu3')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='res3_conv3'))

        (self.feed('Res2')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='res3_skip'))

        (self.feed('res3_conv3',
                   'res3_skip')
         .add(name='Res3'))



# pool1
        (self.feed('Res3')
         .max_pool(2, 2, 2, 2, name='pool2'))

# res4
        (self.feed('pool2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='res4_batch1')
             .relu(name='res4_relu1')
             .conv(1, 1, 64, 1, 1, biased=True, relu=False, name='res4_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='res4_batch2')
             .relu(name='res4_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 64, 1, 1, biased=True, relu=False, name='res4_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='res4_batch3')
             .relu(name='res4_relu3')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='res4_conv3'))

        (self.feed('pool2')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='res4_skip'))

        (self.feed('res4_conv3',
                   'res4_skip')
         .add(name='Res4'))

# res5
        (self.feed('Res4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='res5_batch1')
             .relu(name='res5_relu1')
             .conv(1, 1, 64, 1, 1, biased=True, relu=False, name='res5_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='res5_batch2')
             .relu(name='res5_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 64, 1, 1, biased=True, relu=False, name='res5_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='res5_batch3')
             .relu(name='res5_relu3')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='res5_conv3'))

        (self.feed('Res4')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='res5_skip'))

        (self.feed('res5_conv3',
                   'res5_skip')
         .add(name='Res5'))

# res6
        (self.feed('Res5')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='res6_batch1')
             .relu(name='res6_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='res6_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='res6_batch2')
             .relu(name='res6_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='res6_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='res6_batch3')
             .relu(name='res6_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='res6_conv3'))

        (self.feed('Res5')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='res6_skip'))

        (self.feed('res6_conv3',
                   'res6_skip')
         .add(name='Res6'))































#######################################  HG1  #########################################
                 ####################  Hourglass  #####################
# res1
        (self.feed('Res6')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res1_batch1')
             .relu(name='HG1_res1_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG1_res1_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res1_batch2')
             .relu(name='HG1_res1_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG1_res1_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res1_batch3')
             .relu(name='HG1_res1_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_res1_conv3'))

        (self.feed('Res6')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_res1_skip'))

        (self.feed('HG1_res1_conv3',
                   'HG1_res1_skip')
         .add(name='HG1_res1'))

# resPool1
        (self.feed('HG1_res1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_resPool1_batch1')
             .relu(name='HG1_resPool1_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG1_resPool1_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_resPool1_batch2')
             .relu(name='HG1_resPool1_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG1_resPool1_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_resPool1_batch3')
             .relu(name='HG1_resPool1_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_resPool1_conv3'))

        (self.feed('HG1_res1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_resPool1_skip'))

        (self.feed('HG1_res1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_resPool1_batch4')
             .relu(name='HG1_resPool1_relu4')
             .max_pool(2, 2, 2, 2, name='HG1_resPool1_pool')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='HG1_resPool1_conv4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_resPool1_batch5')
             .relu(name='HG1_resPool1_relu5')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='HG1_resPool1_conv5')
             .upsample(64, 64, name='HG1_resPool1_upSample'))


        (self.feed('HG1_resPool1_conv3',
                   'HG1_resPool1_skip',
                   'HG1_resPool1_upSample')
         .add(name='HG1_resPool1'))



# resPool2
        (self.feed('HG1_resPool1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_resPool2_batch1')
             .relu(name='HG1_resPool2_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG1_resPool2_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_resPool2_batch2')
             .relu(name='HG1_resPool2_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG1_resPool2_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_resPool2_batch3')
             .relu(name='HG1_resPool2_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_resPool2_conv3'))

        (self.feed('HG1_resPool1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_resPool2_skip'))

        (self.feed('HG1_resPool1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_resPool2_batch4')
             .relu(name='HG1_resPool2_relu4')
             .max_pool(2, 2, 2, 2, name='HG1_resPool2_pool')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='HG1_resPool2_conv4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_resPool2_batch5')
             .relu(name='HG1_resPool2_relu5')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='HG1_resPool2_conv5')
             .upsample(64, 64, name='HG1_resPool2_upSample'))


        (self.feed('HG1_resPool2_conv3',
                   'HG1_resPool2_skip',
                   'HG1_resPool2_upSample')
         .add(name='HG1_resPool2'))

# pool1
        (self.feed('Res6')
             .max_pool(2, 2, 2, 2, name='HG1_pool1'))


# res2
        (self.feed('HG1_pool1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res2_batch1')
             .relu(name='HG1_res2_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG1_res2_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res2_batch2')
             .relu(name='HG1_res2_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG1_res2_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res2_batch3')
             .relu(name='HG1_res2_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_res2_conv3'))

        (self.feed('HG1_pool1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_res2_skip'))

        (self.feed('HG1_res2_conv3',
                   'HG1_res2_skip')
         .add(name='HG1_res2'))

# res3
        (self.feed('HG1_res2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res3_batch1')
             .relu(name='HG1_res3_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG1_res3_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res3_batch2')
             .relu(name='HG1_res3_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG1_res3_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res3_batch3')
             .relu(name='HG1_res3_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_res3_conv3'))

        (self.feed('HG1_res2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_res3_skip'))

        (self.feed('HG1_res3_conv3',
                   'HG1_res3_skip')
         .add(name='HG1_res3'))

# resPool3
        (self.feed('HG1_res3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_resPool3_batch1')
             .relu(name='HG1_resPool3_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG1_resPool3_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_resPool3_batch2')
             .relu(name='HG1_resPool3_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG1_resPool3_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_resPool3_batch3')
             .relu(name='HG1_resPool3_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_resPool3_conv3'))

        (self.feed('HG1_res3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_resPool3_skip'))

        (self.feed('HG1_res3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_resPool3_batch4')
             .relu(name='HG1_resPool3_relu4')
             .max_pool(2, 2, 2, 2, name='HG1_resPool3_pool')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='HG1_resPool3_conv4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_resPool3_batch5')
             .relu(name='HG1_resPool3_relu5')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='HG1_resPool3_conv5')
             .upsample(32, 32, name='HG1_resPool3_upSample'))


        (self.feed('HG1_resPool3_conv3',
                   'HG1_resPool3_skip',
                   'HG1_resPool3_upSample')
         .add(name='HG1_resPool3'))




# pool2
        (self.feed('HG1_res2')
             .max_pool(2, 2, 2, 2, name='HG1_pool2'))


# res4
        (self.feed('HG1_pool2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res4_batch1')
             .relu(name='HG1_res4_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG1_res4_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res4_batch2')
             .relu(name='HG1_res4_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG1_res4_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res4_batch3')
             .relu(name='HG1_res4_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_res4_conv3'))

        (self.feed('HG1_pool2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_res4_skip'))

        (self.feed('HG1_res4_conv3',
                   'HG1_res4_skip')
         .add(name='HG1_res4'))
# id:013 max-pooling
        # (self.feed('HG1_pool3')
        #      .max_pool(2, 2, 2, 2, name='HG1_pool4'))


# res5
        (self.feed('HG1_res4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res5_batch1')
             .relu(name='HG1_res5_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG1_res5_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res5_batch2')
             .relu(name='HG1_res5_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG1_res5_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res5_batch3')
             .relu(name='HG1_res5_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_res5_conv3'))

        (self.feed('HG1_res4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_res5_skip'))

        (self.feed('HG1_res5_conv3',
                   'HG1_res5_skip')
         .add(name='HG1_res5'))


# pool3
        (self.feed('HG1_res4')
             .max_pool(2, 2, 2, 2, name='HG1_pool3'))


# res6
        (self.feed('HG1_pool3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res6_batch1')
             .relu(name='HG1_res6_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG1_res6_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res6_batch2')
             .relu(name='HG1_res6_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG1_res6_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res6_batch3')
             .relu(name='HG1_res6_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_res6_conv3'))

        (self.feed('HG1_pool3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_res6_skip'))

        (self.feed('HG1_res6_conv3',
                   'HG1_res6_skip')
         .add(name='HG1_res6'))

# res7
        (self.feed('HG1_res6')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res7_batch1')
             .relu(name='HG1_res7_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG1_res7_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res7_batch2')
             .relu(name='HG1_res7_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG1_res7_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res7_batch3')
             .relu(name='HG1_res7_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_res7_conv3'))

        (self.feed('HG1_res6')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_res7_skip'))

        (self.feed('HG1_res7_conv3',
                   'HG1_res7_skip')
         .add(name='HG1_res7'))


# pool4
        (self.feed('HG1_res6')
             .max_pool(2, 2, 2, 2, name='HG1_pool4'))

# res8
        (self.feed('HG1_pool4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res8_batch1')
             .relu(name='HG1_res8_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG1_res8_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res8_batch2')
             .relu(name='HG1_res8_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG1_res8_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res8_batch3')
             .relu(name='HG1_res8_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_res8_conv3'))

        (self.feed('HG1_pool4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_res8_skip'))

        (self.feed('HG1_res8_conv3',
                   'HG1_res8_skip')
         .add(name='HG1_res8'))

# res9
        (self.feed('HG1_res8')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res9_batch1')
             .relu(name='HG1_res9_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG1_res9_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res9_batch2')
             .relu(name='HG1_res9_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG1_res9_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res9_batch3')
             .relu(name='HG1_res9_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_res9_conv3'))

        (self.feed('HG1_res8')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_res9_skip'))

        (self.feed('HG1_res9_conv3',
                   'HG1_res9_skip')
         .add(name='HG1_res9'))

# res10
        (self.feed('HG1_res9')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res10_batch1')
             .relu(name='HG1_res10_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG1_res10_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res10_batch2')
             .relu(name='HG1_res10_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG1_res10_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res10_batch3')
             .relu(name='HG1_res10_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_res10_conv3'))

        (self.feed('HG1_res9')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_res10_skip'))

        (self.feed('HG1_res10_conv3',
                   'HG1_res10_skip')
         .add(name='HG1_res10'))


# upsample1
        (self.feed('HG1_res10')
             .upsample(8, 8, name='HG1_upSample1'))

# upsample1 + up1(Hg_res7)
        (self.feed('HG1_upSample1',
                   'HG1_res7')
         .add(name='HG1_add1'))


# res11
        (self.feed('HG1_add1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res11_batch1')
             .relu(name='HG1_res11_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG1_res11_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res11_batch2')
             .relu(name='HG1_res11_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG1_res11_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res11_batch3')
             .relu(name='HG1_res11_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_res11_conv3'))

        (self.feed('HG1_add1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_res11_skip'))

        (self.feed('HG1_res11_conv3',
                   'HG1_res11_skip')
         .add(name='HG1_res11'))


# upsample2
        (self.feed('HG1_res11')
             .upsample(16, 16, name='HG1_upSample2'))

# upsample2 + up1(Hg_res5)
        (self.feed('HG1_upSample2',
                   'HG1_res5')
         .add(name='HG1_add2'))


# res12
        (self.feed('HG1_add2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res12_batch1')
             .relu(name='HG1_res12_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG1_res12_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res12_batch2')
             .relu(name='HG1_res12_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG1_res12_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res12_batch3')
             .relu(name='HG1_res12_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_res12_conv3'))

        (self.feed('HG1_add2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_res12_skip'))

        (self.feed('HG1_res12_conv3',
                   'HG1_res12_skip')
         .add(name='HG1_res12'))


# upsample3
        (self.feed('HG1_res12')
             .upsample(32, 32, name='HG1_upSample3'))

# upsample3 + HG1_resPool3
        (self.feed('HG1_upSample3',
                   'HG1_resPool3')
         .add(name='HG1_add3'))


# res13
        (self.feed('HG1_add3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res13_batch1')
             .relu(name='HG1_res13_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG1_res13_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res13_batch2')
             .relu(name='HG1_res13_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG1_res13_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res13_batch3')
             .relu(name='HG1_res13_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_res13_conv3'))

        (self.feed('HG1_add3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_res13_skip'))

        (self.feed('HG1_res13_conv3',
                   'HG1_res13_skip')
         .add(name='HG1_res13'))


# upsample4
        (self.feed('HG1_res13')
             .upsample(64, 64, name='HG1_upSample4'))

# upsample4 + HG1_resPool2
        (self.feed('HG1_upSample4',
                   'HG1_resPool2')
         .add(name='HG1_add4'))

###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  Hourglass  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

################################# HG1 postprocess #################
# Linear layer to produce first set of predictions
# ll1
        (self.feed('HG1_add4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_linearfunc1_conv1', padding='SAME')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_linearfunc1_batch1')
             .relu(name='HG1_linearfunc1_relu'))
# ll2
        (self.feed('HG1_linearfunc1_relu')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_linearfunc2_conv1', padding='SAME')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_linearfunc2_batch1')
             .relu(name='HG1_linearfunc2_relu'))

# att itersize=3
# att U input: ll2
        (self.feed('HG1_linearfunc2_relu')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG1_U', padding='SAME'))

        # self.HG1_U = (self.feed('HG1_linearfunc2_relu')
        #      .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG1_U', padding='SAME'))
# att i=1 conv  C(1)
        # with tf.variable_scope('HG1_share_param', reuse=False):
        #     (self.feed('HG1_U')
        #          .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG1_spConv1', padding='SAME'))

        with tf.variable_scope('HG1_share_param', reuse=False):
            (self.feed('HG1_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG1_spConv1', padding='SAME'))
# att Qtemp
        (self.feed('HG1_spConv1',
                   'HG1_U')
         .add(name='HG1_Qtemp1_add')
         .sigmoid(name='HG1_Qtemp1'))
        # (self.add([self.HG1_spConv1, self.HG1_U], name = 'HG1_Qtemp1_add')
        #     .sigmoid(name='HG1_Qtemp1'))

# att i=2 conv  C(2)
        with tf.variable_scope('HG1_share_param', reuse=True):
            (self.feed('HG1_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False,name='HG1_spConv1',  padding='SAME'))
# att Qtemp2
        (self.feed('HG1_spConv1',
                   'HG1_U')
         .add(name='HG1_Qtemp2_add')
         .sigmoid(name='HG1_Qtemp2'))
        # self.add([HG1_spConv2, HG1_U], name = 'HG1_Qtemp2_add')
        # (self.feed('HG1_Qtemp2_add')
        #     .sigmoid(name='HG1_Qtemp2'))

# att i=3 conv  C(3)
        with tf.variable_scope('HG1_share_param', reuse=True):
            (self.feed('HG1_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG1_spConv1' , padding='SAME'))
# att Qtemp
        (self.feed('HG1_spConv1',
                   'HG1_U')
         .add(name='HG1_Qtemp3_add')
         .sigmoid(name='HG1_Qtemp3'))
        # self.add([HG1_spConv3, HG1_U], name = 'HG1_Qtemp3_add')
        # (self.feed('HG1_Qtemp3_add')
        #     .sigmoid(name='HG1_Qtemp3'))

# att pfeat
        (self.feed('HG1_Qtemp3')
         .replicate(256, 3, name='HG1_pfeat_replicate'))### dim =1?

        (self.feed('HG1_linearfunc2_relu')
            .printLayer(name='printLayer1'))

        (self.feed('HG1_pfeat_replicate')
            .printLayer(name='printLayer2'))

        (self.feed('HG1_Qtemp3')
            .printLayer(name='printLayer3'))


        (self.feed('HG1_linearfunc2_relu',
                   'HG1_pfeat_replicate')
         .multiply2(name='HG1_pfeat_multiply'))

# tmpOut
        (self.feed('HG1_pfeat_multiply')
             .conv(1, 1, n_classes, 1, 1, biased=True, relu=False, name='HG1_Heatmap', padding='SAME'))

###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ HG1 postprocess ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

############################# generate input for next HG ##########

# outmap
        (self.feed('HG1_Heatmap')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_outmap', padding='SAME'))
# ll3
        (self.feed('HG1_linearfunc1_relu')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_linearfunc3_conv1', padding='SAME')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_linearfunc3_batch1')
             .relu(name='HG1_linearfunc3_relu'))
# tmointer
        (self.feed('Res6',
                   'HG1_outmap',
                   'HG1_linearfunc3_relu')
         .add(name='HG2_input'))

###^^^^^^^^^^^^^^^^^^^^^^^^^^^ generate input for next HG ^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

























#######################################  HG2  #########################################
                 ####################  Hourglass  #####################
# res1
        (self.feed('HG2_input')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res1_batch1')
             .relu(name='HG2_res1_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG2_res1_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res1_batch2')
             .relu(name='HG2_res1_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG2_res1_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res1_batch3')
             .relu(name='HG2_res1_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res1_conv3'))

        (self.feed('HG2_input')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res1_skip'))

        (self.feed('HG2_res1_conv3',
                   'HG2_res1_skip')
         .add(name='HG2_res1'))

# resPool1
        (self.feed('HG2_res1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_resPool1_batch1')
             .relu(name='HG2_resPool1_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG2_resPool1_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_resPool1_batch2')
             .relu(name='HG2_resPool1_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG2_resPool1_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_resPool1_batch3')
             .relu(name='HG2_resPool1_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_resPool1_conv3'))

        (self.feed('HG2_res1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_resPool1_skip'))

        (self.feed('HG2_res1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_resPool1_batch4')
             .relu(name='HG2_resPool1_relu4')
             .max_pool(2, 2, 2, 2, name='HG2_resPool1_pool')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='HG2_resPool1_conv4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_resPool1_batch5')
             .relu(name='HG2_resPool1_relu5')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='HG2_resPool1_conv5')
             .upsample(64, 64, name='HG2_resPool1_upSample'))


        (self.feed('HG2_resPool1_conv3',
                   'HG2_resPool1_skip',
                   'HG2_resPool1_upSample')
         .add(name='HG2_resPool1'))



# resPool2
        (self.feed('HG2_resPool1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_resPool2_batch1')
             .relu(name='HG2_resPool2_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG2_resPool2_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_resPool2_batch2')
             .relu(name='HG2_resPool2_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG2_resPool2_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_resPool2_batch3')
             .relu(name='HG2_resPool2_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_resPool2_conv3'))

        (self.feed('HG2_resPool1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_resPool2_skip'))

        (self.feed('HG2_resPool1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_resPool2_batch4')
             .relu(name='HG2_resPool2_relu4')
             .max_pool(2, 2, 2, 2, name='HG2_resPool2_pool')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='HG2_resPool2_conv4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_resPool2_batch5')
             .relu(name='HG2_resPool2_relu5')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='HG2_resPool2_conv5')
             .upsample(64, 64, name='HG2_resPool2_upSample'))


        (self.feed('HG2_resPool2_conv3',
                   'HG2_resPool2_skip',
                   'HG2_resPool2_upSample')
         .add(name='HG2_resPool2'))

# pool1
        (self.feed('HG2_input')
             .max_pool(2, 2, 2, 2, name='HG2_pool1'))


# res2
        (self.feed('HG2_pool1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res2_batch1')
             .relu(name='HG2_res2_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG2_res2_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res2_batch2')
             .relu(name='HG2_res2_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG2_res2_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res2_batch3')
             .relu(name='HG2_res2_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res2_conv3'))

        (self.feed('HG2_pool1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res2_skip'))

        (self.feed('HG2_res2_conv3',
                   'HG2_res2_skip')
         .add(name='HG2_res2'))

# res3
        (self.feed('HG2_res2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res3_batch1')
             .relu(name='HG2_res3_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG2_res3_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res3_batch2')
             .relu(name='HG2_res3_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG2_res3_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res3_batch3')
             .relu(name='HG2_res3_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res3_conv3'))

        (self.feed('HG2_res2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res3_skip'))

        (self.feed('HG2_res3_conv3',
                   'HG2_res3_skip')
         .add(name='HG2_res3'))

# resPool3
        (self.feed('HG2_res3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_resPool3_batch1')
             .relu(name='HG2_resPool3_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG2_resPool3_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_resPool3_batch2')
             .relu(name='HG2_resPool3_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG2_resPool3_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_resPool3_batch3')
             .relu(name='HG2_resPool3_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_resPool3_conv3'))

        (self.feed('HG2_res3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_resPool3_skip'))

        (self.feed('HG2_res3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_resPool3_batch4')
             .relu(name='HG2_resPool3_relu4')
             .max_pool(2, 2, 2, 2, name='HG2_resPool3_pool')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='HG2_resPool3_conv4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_resPool3_batch5')
             .relu(name='HG2_resPool3_relu5')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='HG2_resPool3_conv5')
             .upsample(32, 32, name='HG2_resPool3_upSample'))


        (self.feed('HG2_resPool3_conv3',
                   'HG2_resPool3_skip',
                   'HG2_resPool3_upSample')
         .add(name='HG2_resPool3'))




# pool2
        (self.feed('HG2_res2')
             .max_pool(2, 2, 2, 2, name='HG2_pool2'))


# res4
        (self.feed('HG2_pool2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res4_batch1')
             .relu(name='HG2_res4_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG2_res4_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res4_batch2')
             .relu(name='HG2_res4_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG2_res4_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res4_batch3')
             .relu(name='HG2_res4_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res4_conv3'))

        (self.feed('HG2_pool2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res4_skip'))

        (self.feed('HG2_res4_conv3',
                   'HG2_res4_skip')
         .add(name='HG2_res4'))
# id:013 max-pooling
        # (self.feed('HG2_pool3')
        #      .max_pool(2, 2, 2, 2, name='HG2_pool4'))


# res5
        (self.feed('HG2_res4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res5_batch1')
             .relu(name='HG2_res5_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG2_res5_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res5_batch2')
             .relu(name='HG2_res5_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG2_res5_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res5_batch3')
             .relu(name='HG2_res5_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res5_conv3'))

        (self.feed('HG2_res4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res5_skip'))

        (self.feed('HG2_res5_conv3',
                   'HG2_res5_skip')
         .add(name='HG2_res5'))


# pool3
        (self.feed('HG2_res4')
             .max_pool(2, 2, 2, 2, name='HG2_pool3'))


# res6
        (self.feed('HG2_pool3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res6_batch1')
             .relu(name='HG2_res6_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG2_res6_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res6_batch2')
             .relu(name='HG2_res6_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG2_res6_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res6_batch3')
             .relu(name='HG2_res6_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res6_conv3'))

        (self.feed('HG2_pool3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res6_skip'))

        (self.feed('HG2_res6_conv3',
                   'HG2_res6_skip')
         .add(name='HG2_res6'))

# res7
        (self.feed('HG2_res6')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res7_batch1')
             .relu(name='HG2_res7_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG2_res7_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res7_batch2')
             .relu(name='HG2_res7_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG2_res7_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res7_batch3')
             .relu(name='HG2_res7_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res7_conv3'))

        (self.feed('HG2_res6')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res7_skip'))

        (self.feed('HG2_res7_conv3',
                   'HG2_res7_skip')
         .add(name='HG2_res7'))


# pool4
        (self.feed('HG2_res6')
             .max_pool(2, 2, 2, 2, name='HG2_pool4'))

# res8
        (self.feed('HG2_pool4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res8_batch1')
             .relu(name='HG2_res8_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG2_res8_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res8_batch2')
             .relu(name='HG2_res8_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG2_res8_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res8_batch3')
             .relu(name='HG2_res8_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res8_conv3'))

        (self.feed('HG2_pool4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res8_skip'))

        (self.feed('HG2_res8_conv3',
                   'HG2_res8_skip')
         .add(name='HG2_res8'))

# res9
        (self.feed('HG2_res8')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res9_batch1')
             .relu(name='HG2_res9_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG2_res9_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res9_batch2')
             .relu(name='HG2_res9_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG2_res9_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res9_batch3')
             .relu(name='HG2_res9_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res9_conv3'))

        (self.feed('HG2_res8')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res9_skip'))

        (self.feed('HG2_res9_conv3',
                   'HG2_res9_skip')
         .add(name='HG2_res9'))

# res10
        (self.feed('HG2_res9')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res10_batch1')
             .relu(name='HG2_res10_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG2_res10_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res10_batch2')
             .relu(name='HG2_res10_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG2_res10_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res10_batch3')
             .relu(name='HG2_res10_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res10_conv3'))

        (self.feed('HG2_res9')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res10_skip'))

        (self.feed('HG2_res10_conv3',
                   'HG2_res10_skip')
         .add(name='HG2_res10'))


# upsample1
        (self.feed('HG2_res10')
             .upsample(8, 8, name='HG2_upSample1'))

# upsample1 + up1(Hg_res7)
        (self.feed('HG2_upSample1',
                   'HG2_res7')
         .add(name='HG2_add1'))


# res11
        (self.feed('HG2_add1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res11_batch1')
             .relu(name='HG2_res11_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG2_res11_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res11_batch2')
             .relu(name='HG2_res11_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG2_res11_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res11_batch3')
             .relu(name='HG2_res11_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res11_conv3'))

        (self.feed('HG2_add1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res11_skip'))

        (self.feed('HG2_res11_conv3',
                   'HG2_res11_skip')
         .add(name='HG2_res11'))


# upsample2
        (self.feed('HG2_res11')
             .upsample(16, 16, name='HG2_upSample2'))

# upsample2 + up1(Hg_res5)
        (self.feed('HG2_upSample2',
                   'HG2_res5')
         .add(name='HG2_add2'))


# res12
        (self.feed('HG2_add2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res12_batch1')
             .relu(name='HG2_res12_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG2_res12_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res12_batch2')
             .relu(name='HG2_res12_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG2_res12_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res12_batch3')
             .relu(name='HG2_res12_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res12_conv3'))

        (self.feed('HG2_add2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res12_skip'))

        (self.feed('HG2_res12_conv3',
                   'HG2_res12_skip')
         .add(name='HG2_res12'))


# upsample3
        (self.feed('HG2_res12')
             .upsample(32, 32, name='HG2_upSample3'))

# upsample3 + HG2_resPool3
        (self.feed('HG2_upSample3',
                   'HG2_resPool3')
         .add(name='HG2_add3'))


# res13
        (self.feed('HG2_add3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res13_batch1')
             .relu(name='HG2_res13_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG2_res13_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res13_batch2')
             .relu(name='HG2_res13_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG2_res13_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res13_batch3')
             .relu(name='HG2_res13_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res13_conv3'))

        (self.feed('HG2_add3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res13_skip'))

        (self.feed('HG2_res13_conv3',
                   'HG2_res13_skip')
         .add(name='HG2_res13'))


# upsample4
        (self.feed('HG2_res13')
             .upsample(64, 64, name='HG2_upSample4'))

# upsample4 + HG2_resPool2
        (self.feed('HG2_upSample4',
                   'HG2_resPool2')
         .add(name='HG2_add4'))

###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  Hourglass  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

################################# HG2 postprocess #################
# Linear layer to produce first set of predictions
# ll1
        (self.feed('HG2_add4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_linearfunc1_conv1', padding='SAME')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_linearfunc1_batch1')
             .relu(name='HG2_linearfunc1_relu'))
# ll2
        (self.feed('HG2_linearfunc1_relu')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_linearfunc2_conv1', padding='SAME')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_linearfunc2_batch1')
             .relu(name='HG2_linearfunc2_relu'))

# att itersize=3
# att U input: ll2
        (self.feed('HG2_linearfunc2_relu')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG2_U', padding='SAME'))
# att i=1 conv  C(1)
        with tf.variable_scope('HG2_share_param', reuse=False):
            (self.feed('HG2_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG2_spConv1', padding='SAME'))
# att Qtemp
        (self.feed('HG2_spConv1',
                   'HG2_U')
         .add(name='HG2_Qtemp1_add')
         .sigmoid(name='HG2_Qtemp1'))

# att i=2 conv  C(2)
        with tf.variable_scope('HG2_share_param', reuse=True):
            (self.feed('HG2_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG2_spConv1', padding='SAME'))
# att Qtemp2
        (self.feed('HG2_spConv1',
                   'HG2_U')
         .add(name='HG2_Qtemp2_add')
         .sigmoid(name='HG2_Qtemp2'))

# att i=3 conv  C(3)
        with tf.variable_scope('HG2_share_param', reuse=True):
            (self.feed('HG2_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG2_spConv1', padding='SAME'))
# att Qtemp
        (self.feed('HG2_spConv1',
                   'HG2_U')
         .add(name='HG2_Qtemp3_add')
         .sigmoid(name='HG2_Qtemp3'))
# att pfeat
        (self.feed('HG2_Qtemp3')
         .replicate(256, 3, name='HG2_pfeat_replicate'))

        (self.feed('HG2_linearfunc2_relu',
                   'HG2_pfeat_replicate')
         .multiply2(name='HG2_pfeat_multiply'))

# tmpOut
        (self.feed('HG2_pfeat_multiply')
             .conv(1, 1, n_classes, 1, 1, biased=True, relu=False, name='HG2_Heatmap', padding='SAME'))

###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ HG2 postprocess ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

############################# generate input for next HG ##########

# outmap
        (self.feed('HG2_Heatmap')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_outmap', padding='SAME'))
# ll3
        (self.feed('HG2_linearfunc1_relu')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_linearfunc3_conv1', padding='SAME')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_linearfunc3_batch1')
             .relu(name='HG2_linearfunc3_relu'))
# tmointer
        (self.feed('HG2_input',
                   'HG2_outmap',
                   'HG2_linearfunc3_relu')
         .add(name='HG3_input'))

###^^^^^^^^^^^^^^^^^^^^^^^^^^^ generate input for next HG ^^^^^^^^^^^^^^^^^^^^^^^^^^^^###



































































































































































































































































#######################################  HG3  #########################################
                 ####################  Hourglass  #####################
# res1
        (self.feed('HG3_input')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res1_batch1')
             .relu(name='HG3_res1_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG3_res1_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res1_batch2')
             .relu(name='HG3_res1_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG3_res1_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res1_batch3')
             .relu(name='HG3_res1_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res1_conv3'))

        (self.feed('HG3_input')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res1_skip'))

        (self.feed('HG3_res1_conv3',
                   'HG3_res1_skip')
         .add(name='HG3_res1'))

# resPool1
        (self.feed('HG3_res1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_resPool1_batch1')
             .relu(name='HG3_resPool1_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG3_resPool1_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_resPool1_batch2')
             .relu(name='HG3_resPool1_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG3_resPool1_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_resPool1_batch3')
             .relu(name='HG3_resPool1_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_resPool1_conv3'))

        (self.feed('HG3_res1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_resPool1_skip'))

        (self.feed('HG3_res1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_resPool1_batch4')
             .relu(name='HG3_resPool1_relu4')
             .max_pool(2, 2, 2, 2, name='HG3_resPool1_pool')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='HG3_resPool1_conv4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_resPool1_batch5')
             .relu(name='HG3_resPool1_relu5')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='HG3_resPool1_conv5')
             .upsample(64, 64, name='HG3_resPool1_upSample'))


        (self.feed('HG3_resPool1_conv3',
                   'HG3_resPool1_skip',
                   'HG3_resPool1_upSample')
         .add(name='HG3_resPool1'))



# resPool2
        (self.feed('HG3_resPool1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_resPool2_batch1')
             .relu(name='HG3_resPool2_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG3_resPool2_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_resPool2_batch2')
             .relu(name='HG3_resPool2_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG3_resPool2_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_resPool2_batch3')
             .relu(name='HG3_resPool2_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_resPool2_conv3'))

        (self.feed('HG3_resPool1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_resPool2_skip'))

        (self.feed('HG3_resPool1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_resPool2_batch4')
             .relu(name='HG3_resPool2_relu4')
             .max_pool(2, 2, 2, 2, name='HG3_resPool2_pool')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='HG3_resPool2_conv4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_resPool2_batch5')
             .relu(name='HG3_resPool2_relu5')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='HG3_resPool2_conv5')
             .upsample(64, 64, name='HG3_resPool2_upSample'))


        (self.feed('HG3_resPool2_conv3',
                   'HG3_resPool2_skip',
                   'HG3_resPool2_upSample')
         .add(name='HG3_resPool2'))

# pool1
        (self.feed('HG3_input')
             .max_pool(2, 2, 2, 2, name='HG3_pool1'))


# res2
        (self.feed('HG3_pool1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res2_batch1')
             .relu(name='HG3_res2_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG3_res2_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res2_batch2')
             .relu(name='HG3_res2_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG3_res2_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res2_batch3')
             .relu(name='HG3_res2_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res2_conv3'))

        (self.feed('HG3_pool1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res2_skip'))

        (self.feed('HG3_res2_conv3',
                   'HG3_res2_skip')
         .add(name='HG3_res2'))

# res3
        (self.feed('HG3_res2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res3_batch1')
             .relu(name='HG3_res3_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG3_res3_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res3_batch2')
             .relu(name='HG3_res3_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG3_res3_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res3_batch3')
             .relu(name='HG3_res3_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res3_conv3'))

        (self.feed('HG3_res2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res3_skip'))

        (self.feed('HG3_res3_conv3',
                   'HG3_res3_skip')
         .add(name='HG3_res3'))

# resPool3
        (self.feed('HG3_res3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_resPool3_batch1')
             .relu(name='HG3_resPool3_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG3_resPool3_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_resPool3_batch2')
             .relu(name='HG3_resPool3_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG3_resPool3_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_resPool3_batch3')
             .relu(name='HG3_resPool3_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_resPool3_conv3'))

        (self.feed('HG3_res3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_resPool3_skip'))

        (self.feed('HG3_res3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_resPool3_batch4')
             .relu(name='HG3_resPool3_relu4')
             .max_pool(2, 2, 2, 2, name='HG3_resPool3_pool')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='HG3_resPool3_conv4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_resPool3_batch5')
             .relu(name='HG3_resPool3_relu5')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='HG3_resPool3_conv5')
             .upsample(32, 32, name='HG3_resPool3_upSample'))


        (self.feed('HG3_resPool3_conv3',
                   'HG3_resPool3_skip',
                   'HG3_resPool3_upSample')
         .add(name='HG3_resPool3'))




# pool2
        (self.feed('HG3_res2')
             .max_pool(2, 2, 2, 2, name='HG3_pool2'))


# res4
        (self.feed('HG3_pool2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res4_batch1')
             .relu(name='HG3_res4_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG3_res4_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res4_batch2')
             .relu(name='HG3_res4_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG3_res4_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res4_batch3')
             .relu(name='HG3_res4_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res4_conv3'))

        (self.feed('HG3_pool2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res4_skip'))

        (self.feed('HG3_res4_conv3',
                   'HG3_res4_skip')
         .add(name='HG3_res4'))
# id:013 max-pooling
        # (self.feed('HG3_pool3')
        #      .max_pool(2, 2, 2, 2, name='HG3_pool4'))


# res5
        (self.feed('HG3_res4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res5_batch1')
             .relu(name='HG3_res5_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG3_res5_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res5_batch2')
             .relu(name='HG3_res5_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG3_res5_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res5_batch3')
             .relu(name='HG3_res5_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res5_conv3'))

        (self.feed('HG3_res4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res5_skip'))

        (self.feed('HG3_res5_conv3',
                   'HG3_res5_skip')
         .add(name='HG3_res5'))


# pool3
        (self.feed('HG3_res4')
             .max_pool(2, 2, 2, 2, name='HG3_pool3'))


# res6
        (self.feed('HG3_pool3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res6_batch1')
             .relu(name='HG3_res6_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG3_res6_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res6_batch2')
             .relu(name='HG3_res6_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG3_res6_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res6_batch3')
             .relu(name='HG3_res6_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res6_conv3'))

        (self.feed('HG3_pool3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res6_skip'))

        (self.feed('HG3_res6_conv3',
                   'HG3_res6_skip')
         .add(name='HG3_res6'))

# res7
        (self.feed('HG3_res6')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res7_batch1')
             .relu(name='HG3_res7_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG3_res7_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res7_batch2')
             .relu(name='HG3_res7_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG3_res7_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res7_batch3')
             .relu(name='HG3_res7_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res7_conv3'))

        (self.feed('HG3_res6')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res7_skip'))

        (self.feed('HG3_res7_conv3',
                   'HG3_res7_skip')
         .add(name='HG3_res7'))


# pool4
        (self.feed('HG3_res6')
             .max_pool(2, 2, 2, 2, name='HG3_pool4'))

# res8
        (self.feed('HG3_pool4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res8_batch1')
             .relu(name='HG3_res8_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG3_res8_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res8_batch2')
             .relu(name='HG3_res8_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG3_res8_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res8_batch3')
             .relu(name='HG3_res8_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res8_conv3'))

        (self.feed('HG3_pool4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res8_skip'))

        (self.feed('HG3_res8_conv3',
                   'HG3_res8_skip')
         .add(name='HG3_res8'))

# res9
        (self.feed('HG3_res8')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res9_batch1')
             .relu(name='HG3_res9_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG3_res9_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res9_batch2')
             .relu(name='HG3_res9_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG3_res9_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res9_batch3')
             .relu(name='HG3_res9_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res9_conv3'))

        (self.feed('HG3_res8')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res9_skip'))

        (self.feed('HG3_res9_conv3',
                   'HG3_res9_skip')
         .add(name='HG3_res9'))

# res10
        (self.feed('HG3_res9')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res10_batch1')
             .relu(name='HG3_res10_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG3_res10_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res10_batch2')
             .relu(name='HG3_res10_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG3_res10_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res10_batch3')
             .relu(name='HG3_res10_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res10_conv3'))

        (self.feed('HG3_res9')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res10_skip'))

        (self.feed('HG3_res10_conv3',
                   'HG3_res10_skip')
         .add(name='HG3_res10'))


# upsample1
        (self.feed('HG3_res10')
             .upsample(8, 8, name='HG3_upSample1'))

# upsample1 + up1(Hg_res7)
        (self.feed('HG3_upSample1',
                   'HG3_res7')
         .add(name='HG3_add1'))


# res11
        (self.feed('HG3_add1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res11_batch1')
             .relu(name='HG3_res11_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG3_res11_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res11_batch2')
             .relu(name='HG3_res11_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG3_res11_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res11_batch3')
             .relu(name='HG3_res11_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res11_conv3'))

        (self.feed('HG3_add1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res11_skip'))

        (self.feed('HG3_res11_conv3',
                   'HG3_res11_skip')
         .add(name='HG3_res11'))


# upsample2
        (self.feed('HG3_res11')
             .upsample(16, 16, name='HG3_upSample2'))

# upsample2 + up1(Hg_res5)
        (self.feed('HG3_upSample2',
                   'HG3_res5')
         .add(name='HG3_add2'))


# res12
        (self.feed('HG3_add2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res12_batch1')
             .relu(name='HG3_res12_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG3_res12_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res12_batch2')
             .relu(name='HG3_res12_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG3_res12_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res12_batch3')
             .relu(name='HG3_res12_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res12_conv3'))

        (self.feed('HG3_add2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res12_skip'))

        (self.feed('HG3_res12_conv3',
                   'HG3_res12_skip')
         .add(name='HG3_res12'))


# upsample3
        (self.feed('HG3_res12')
             .upsample(32, 32, name='HG3_upSample3'))

# upsample3 + HG3_resPool3
        (self.feed('HG3_upSample3',
                   'HG3_resPool3')
         .add(name='HG3_add3'))


# res13
        (self.feed('HG3_add3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res13_batch1')
             .relu(name='HG3_res13_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG3_res13_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res13_batch2')
             .relu(name='HG3_res13_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG3_res13_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res13_batch3')
             .relu(name='HG3_res13_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res13_conv3'))

        (self.feed('HG3_add3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res13_skip'))

        (self.feed('HG3_res13_conv3',
                   'HG3_res13_skip')
         .add(name='HG3_res13'))


# upsample4
        (self.feed('HG3_res13')
             .upsample(64, 64, name='HG3_upSample4'))

# upsample4 + HG3_resPool2
        (self.feed('HG3_upSample4',
                   'HG3_resPool2')
         .add(name='HG3_add4'))

###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  Hourglass  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

################################# HG3 postprocess #################
# Linear layer to produce first set of predictions
# ll1
        (self.feed('HG3_add4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_linearfunc1_conv1', padding='SAME')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_linearfunc1_batch1')
             .relu(name='HG3_linearfunc1_relu'))
# ll2
        (self.feed('HG3_linearfunc1_relu')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_linearfunc2_conv1', padding='SAME')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_linearfunc2_batch1')
             .relu(name='HG3_linearfunc2_relu'))

# att itersize=3
# att U input: ll2
        (self.feed('HG3_linearfunc2_relu')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG3_U', padding='SAME'))
# att i=1 conv  C(1)
        with tf.variable_scope('HG3_share_param', reuse=False):
            (self.feed('HG3_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG3_spConv1', padding='SAME'))
# att Qtemp
        (self.feed('HG3_spConv1',
                   'HG3_U')
         .add(name='HG3_Qtemp1_add')
         .sigmoid(name='HG3_Qtemp1'))

# att i=2 conv  C(2)
        with tf.variable_scope('HG3_share_param', reuse=True):
            (self.feed('HG3_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG3_spConv1', padding='SAME'))
# att Qtemp2
        (self.feed('HG3_spConv1',
                   'HG3_U')
         .add(name='HG3_Qtemp2_add')
         .sigmoid(name='HG3_Qtemp2'))

# att i=3 conv  C(3)
        with tf.variable_scope('HG3_share_param', reuse=True):
            (self.feed('HG3_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG3_spConv1', padding='SAME'))
# att Qtemp
        (self.feed('HG3_spConv1',
                   'HG3_U')
         .add(name='HG3_Qtemp3_add')
         .sigmoid(name='HG3_Qtemp3'))
# att pfeat
        (self.feed('HG3_Qtemp3')
         .replicate(256, 3, name='HG3_pfeat_replicate'))

        (self.feed('HG3_linearfunc2_relu',
                   'HG3_pfeat_replicate')
         .multiply2(name='HG3_pfeat_multiply'))

# tmpOut
        (self.feed('HG3_pfeat_multiply')
             .conv(1, 1, n_classes, 1, 1, biased=True, relu=False, name='HG3_Heatmap', padding='SAME'))

###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ HG3 postprocess ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

############################# generate input for next HG ##########

# outmap
        (self.feed('HG3_Heatmap')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_outmap', padding='SAME'))
# ll3
        (self.feed('HG3_linearfunc1_relu')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_linearfunc3_conv1', padding='SAME')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_linearfunc3_batch1')
             .relu(name='HG3_linearfunc3_relu'))
# tmointer
        (self.feed('HG3_input',
                   'HG3_outmap',
                   'HG3_linearfunc3_relu')
         .add(name='HG4_input'))

###^^^^^^^^^^^^^^^^^^^^^^^^^^^ generate input for next HG ^^^^^^^^^^^^^^^^^^^^^^^^^^^^###



























































#######################################  HG4  #########################################
                 ####################  Hourglass  #####################
# res1
        (self.feed('HG4_input')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res1_batch1')
             .relu(name='HG4_res1_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG4_res1_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res1_batch2')
             .relu(name='HG4_res1_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG4_res1_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res1_batch3')
             .relu(name='HG4_res1_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res1_conv3'))

        (self.feed('HG4_input')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res1_skip'))

        (self.feed('HG4_res1_conv3',
                   'HG4_res1_skip')
         .add(name='HG4_res1'))

# resPool1
        (self.feed('HG4_res1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_resPool1_batch1')
             .relu(name='HG4_resPool1_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG4_resPool1_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_resPool1_batch2')
             .relu(name='HG4_resPool1_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG4_resPool1_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_resPool1_batch3')
             .relu(name='HG4_resPool1_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_resPool1_conv3'))

        (self.feed('HG4_res1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_resPool1_skip'))

        (self.feed('HG4_res1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_resPool1_batch4')
             .relu(name='HG4_resPool1_relu4')
             .max_pool(2, 2, 2, 2, name='HG4_resPool1_pool')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='HG4_resPool1_conv4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_resPool1_batch5')
             .relu(name='HG4_resPool1_relu5')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='HG4_resPool1_conv5')
             .upsample(64, 64, name='HG4_resPool1_upSample'))


        (self.feed('HG4_resPool1_conv3',
                   'HG4_resPool1_skip',
                   'HG4_resPool1_upSample')
         .add(name='HG4_resPool1'))



# resPool2
        (self.feed('HG4_resPool1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_resPool2_batch1')
             .relu(name='HG4_resPool2_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG4_resPool2_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_resPool2_batch2')
             .relu(name='HG4_resPool2_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG4_resPool2_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_resPool2_batch3')
             .relu(name='HG4_resPool2_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_resPool2_conv3'))

        (self.feed('HG4_resPool1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_resPool2_skip'))

        (self.feed('HG4_resPool1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_resPool2_batch4')
             .relu(name='HG4_resPool2_relu4')
             .max_pool(2, 2, 2, 2, name='HG4_resPool2_pool')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='HG4_resPool2_conv4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_resPool2_batch5')
             .relu(name='HG4_resPool2_relu5')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='HG4_resPool2_conv5')
             .upsample(64, 64, name='HG4_resPool2_upSample'))


        (self.feed('HG4_resPool2_conv3',
                   'HG4_resPool2_skip',
                   'HG4_resPool2_upSample')
         .add(name='HG4_resPool2'))

# pool1
        (self.feed('HG4_input')
             .max_pool(2, 2, 2, 2, name='HG4_pool1'))


# res2
        (self.feed('HG4_pool1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res2_batch1')
             .relu(name='HG4_res2_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG4_res2_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res2_batch2')
             .relu(name='HG4_res2_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG4_res2_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res2_batch3')
             .relu(name='HG4_res2_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res2_conv3'))

        (self.feed('HG4_pool1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res2_skip'))

        (self.feed('HG4_res2_conv3',
                   'HG4_res2_skip')
         .add(name='HG4_res2'))

# res3
        (self.feed('HG4_res2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res3_batch1')
             .relu(name='HG4_res3_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG4_res3_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res3_batch2')
             .relu(name='HG4_res3_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG4_res3_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res3_batch3')
             .relu(name='HG4_res3_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res3_conv3'))

        (self.feed('HG4_res2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res3_skip'))

        (self.feed('HG4_res3_conv3',
                   'HG4_res3_skip')
         .add(name='HG4_res3'))

# resPool3
        (self.feed('HG4_res3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_resPool3_batch1')
             .relu(name='HG4_resPool3_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG4_resPool3_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_resPool3_batch2')
             .relu(name='HG4_resPool3_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG4_resPool3_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_resPool3_batch3')
             .relu(name='HG4_resPool3_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_resPool3_conv3'))

        (self.feed('HG4_res3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_resPool3_skip'))

        (self.feed('HG4_res3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_resPool3_batch4')
             .relu(name='HG4_resPool3_relu4')
             .max_pool(2, 2, 2, 2, name='HG4_resPool3_pool')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='HG4_resPool3_conv4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_resPool3_batch5')
             .relu(name='HG4_resPool3_relu5')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='HG4_resPool3_conv5')
             .upsample(32, 32, name='HG4_resPool3_upSample'))


        (self.feed('HG4_resPool3_conv3',
                   'HG4_resPool3_skip',
                   'HG4_resPool3_upSample')
         .add(name='HG4_resPool3'))




# pool2
        (self.feed('HG4_res2')
             .max_pool(2, 2, 2, 2, name='HG4_pool2'))


# res4
        (self.feed('HG4_pool2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res4_batch1')
             .relu(name='HG4_res4_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG4_res4_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res4_batch2')
             .relu(name='HG4_res4_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG4_res4_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res4_batch3')
             .relu(name='HG4_res4_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res4_conv3'))

        (self.feed('HG4_pool2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res4_skip'))

        (self.feed('HG4_res4_conv3',
                   'HG4_res4_skip')
         .add(name='HG4_res4'))
# id:013 max-pooling
        # (self.feed('HG4_pool3')
        #      .max_pool(2, 2, 2, 2, name='HG4_pool4'))


# res5
        (self.feed('HG4_res4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res5_batch1')
             .relu(name='HG4_res5_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG4_res5_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res5_batch2')
             .relu(name='HG4_res5_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG4_res5_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res5_batch3')
             .relu(name='HG4_res5_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res5_conv3'))

        (self.feed('HG4_res4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res5_skip'))

        (self.feed('HG4_res5_conv3',
                   'HG4_res5_skip')
         .add(name='HG4_res5'))


# pool3
        (self.feed('HG4_res4')
             .max_pool(2, 2, 2, 2, name='HG4_pool3'))


# res6
        (self.feed('HG4_pool3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res6_batch1')
             .relu(name='HG4_res6_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG4_res6_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res6_batch2')
             .relu(name='HG4_res6_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG4_res6_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res6_batch3')
             .relu(name='HG4_res6_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res6_conv3'))

        (self.feed('HG4_pool3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res6_skip'))

        (self.feed('HG4_res6_conv3',
                   'HG4_res6_skip')
         .add(name='HG4_res6'))

# res7
        (self.feed('HG4_res6')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res7_batch1')
             .relu(name='HG4_res7_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG4_res7_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res7_batch2')
             .relu(name='HG4_res7_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG4_res7_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res7_batch3')
             .relu(name='HG4_res7_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res7_conv3'))

        (self.feed('HG4_res6')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res7_skip'))

        (self.feed('HG4_res7_conv3',
                   'HG4_res7_skip')
         .add(name='HG4_res7'))


# pool4
        (self.feed('HG4_res6')
             .max_pool(2, 2, 2, 2, name='HG4_pool4'))

# res8
        (self.feed('HG4_pool4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res8_batch1')
             .relu(name='HG4_res8_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG4_res8_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res8_batch2')
             .relu(name='HG4_res8_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG4_res8_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res8_batch3')
             .relu(name='HG4_res8_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res8_conv3'))

        (self.feed('HG4_pool4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res8_skip'))

        (self.feed('HG4_res8_conv3',
                   'HG4_res8_skip')
         .add(name='HG4_res8'))

# res9
        (self.feed('HG4_res8')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res9_batch1')
             .relu(name='HG4_res9_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG4_res9_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res9_batch2')
             .relu(name='HG4_res9_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG4_res9_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res9_batch3')
             .relu(name='HG4_res9_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res9_conv3'))

        (self.feed('HG4_res8')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res9_skip'))

        (self.feed('HG4_res9_conv3',
                   'HG4_res9_skip')
         .add(name='HG4_res9'))

# res10
        (self.feed('HG4_res9')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res10_batch1')
             .relu(name='HG4_res10_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG4_res10_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res10_batch2')
             .relu(name='HG4_res10_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG4_res10_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res10_batch3')
             .relu(name='HG4_res10_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res10_conv3'))

        (self.feed('HG4_res9')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res10_skip'))

        (self.feed('HG4_res10_conv3',
                   'HG4_res10_skip')
         .add(name='HG4_res10'))


# upsample1
        (self.feed('HG4_res10')
             .upsample(8, 8, name='HG4_upSample1'))

# upsample1 + up1(Hg_res7)
        (self.feed('HG4_upSample1',
                   'HG4_res7')
         .add(name='HG4_add1'))


# res11
        (self.feed('HG4_add1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res11_batch1')
             .relu(name='HG4_res11_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG4_res11_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res11_batch2')
             .relu(name='HG4_res11_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG4_res11_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res11_batch3')
             .relu(name='HG4_res11_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res11_conv3'))

        (self.feed('HG4_add1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res11_skip'))

        (self.feed('HG4_res11_conv3',
                   'HG4_res11_skip')
         .add(name='HG4_res11'))


# upsample2
        (self.feed('HG4_res11')
             .upsample(16, 16, name='HG4_upSample2'))

# upsample2 + up1(Hg_res5)
        (self.feed('HG4_upSample2',
                   'HG4_res5')
         .add(name='HG4_add2'))


# res12
        (self.feed('HG4_add2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res12_batch1')
             .relu(name='HG4_res12_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG4_res12_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res12_batch2')
             .relu(name='HG4_res12_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG4_res12_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res12_batch3')
             .relu(name='HG4_res12_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res12_conv3'))

        (self.feed('HG4_add2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res12_skip'))

        (self.feed('HG4_res12_conv3',
                   'HG4_res12_skip')
         .add(name='HG4_res12'))


# upsample3
        (self.feed('HG4_res12')
             .upsample(32, 32, name='HG4_upSample3'))

# upsample3 + HG4_resPool3
        (self.feed('HG4_upSample3',
                   'HG4_resPool3')
         .add(name='HG4_add3'))


# res13
        (self.feed('HG4_add3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res13_batch1')
             .relu(name='HG4_res13_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG4_res13_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res13_batch2')
             .relu(name='HG4_res13_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG4_res13_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res13_batch3')
             .relu(name='HG4_res13_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res13_conv3'))

        (self.feed('HG4_add3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res13_skip'))

        (self.feed('HG4_res13_conv3',
                   'HG4_res13_skip')
         .add(name='HG4_res13'))


# upsample4
        (self.feed('HG4_res13')
             .upsample(64, 64, name='HG4_upSample4'))

# upsample4 + HG4_resPool2
        (self.feed('HG4_upSample4',
                   'HG4_resPool2')
         .add(name='HG4_add4'))

###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  Hourglass  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

################################# HG4 postprocess #################
# Linear layer to produce first set of predictions
# ll1
        (self.feed('HG4_add4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_linearfunc1_conv1', padding='SAME')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_linearfunc1_batch1')
             .relu(name='HG4_linearfunc1_relu'))
# ll2
        (self.feed('HG4_linearfunc1_relu')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_linearfunc2_conv1', padding='SAME')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_linearfunc2_batch1')
             .relu(name='HG4_linearfunc2_relu'))

# att itersize=3
# att U input: ll2
        (self.feed('HG4_linearfunc2_relu')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG4_U', padding='SAME'))
# att i=1 conv  C(1)
        with tf.variable_scope('HG4_share_param', reuse=False):
            (self.feed('HG4_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG4_spConv1', padding='SAME'))
# att Qtemp
        (self.feed('HG4_spConv1',
                   'HG4_U')
         .add(name='HG4_Qtemp1_add')
         .sigmoid(name='HG4_Qtemp1'))

# att i=2 conv  C(2)
        with tf.variable_scope('HG4_share_param', reuse=True):
            (self.feed('HG4_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG4_spConv1', padding='SAME'))
# att Qtemp2
        (self.feed('HG4_spConv1',
                   'HG4_U')
         .add(name='HG4_Qtemp2_add')
         .sigmoid(name='HG4_Qtemp2'))

# att i=3 conv  C(3)
        with tf.variable_scope('HG4_share_param', reuse=True):
            (self.feed('HG4_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG4_spConv1', padding='SAME'))
# att Qtemp
        (self.feed('HG4_spConv1',
                   'HG4_U')
         .add(name='HG4_Qtemp3_add')
         .sigmoid(name='HG4_Qtemp3'))
# att pfeat
        (self.feed('HG4_Qtemp3')
         .replicate(256, 3, name='HG4_pfeat_replicate'))

        (self.feed('HG4_linearfunc2_relu',
                   'HG4_pfeat_replicate')
         .multiply2(name='HG4_pfeat_multiply'))

# tmpOut
        (self.feed('HG4_pfeat_multiply')
             .conv(1, 1, n_classes, 1, 1, biased=True, relu=False, name='HG4_Heatmap', padding='SAME'))

###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ HG4 postprocess ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

############################# generate input for next HG ##########

# outmap
        (self.feed('HG4_Heatmap')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_outmap', padding='SAME'))
# ll3
        (self.feed('HG4_linearfunc1_relu')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_linearfunc3_conv1', padding='SAME')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_linearfunc3_batch1')
             .relu(name='HG4_linearfunc3_relu'))
# tmointer
        (self.feed('HG4_input',
                   'HG4_outmap',
                   'HG4_linearfunc3_relu')
         .add(name='HG5_input'))

###^^^^^^^^^^^^^^^^^^^^^^^^^^^ generate input for next HG ^^^^^^^^^^^^^^^^^^^^^^^^^^^^###


























































































#######################################  HG5  #########################################
                 ####################  Hourglass  #####################
# res1
        (self.feed('HG5_input')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res1_batch1')
             .relu(name='HG5_res1_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG5_res1_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res1_batch2')
             .relu(name='HG5_res1_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG5_res1_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res1_batch3')
             .relu(name='HG5_res1_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res1_conv3'))

        (self.feed('HG5_input')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res1_skip'))

        (self.feed('HG5_res1_conv3',
                   'HG5_res1_skip')
         .add(name='HG5_res1'))

# resPool1
        (self.feed('HG5_res1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_resPool1_batch1')
             .relu(name='HG5_resPool1_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG5_resPool1_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_resPool1_batch2')
             .relu(name='HG5_resPool1_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG5_resPool1_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_resPool1_batch3')
             .relu(name='HG5_resPool1_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_resPool1_conv3'))

        (self.feed('HG5_res1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_resPool1_skip'))

        (self.feed('HG5_res1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_resPool1_batch4')
             .relu(name='HG5_resPool1_relu4')
             .max_pool(2, 2, 2, 2, name='HG5_resPool1_pool')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='HG5_resPool1_conv4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_resPool1_batch5')
             .relu(name='HG5_resPool1_relu5')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='HG5_resPool1_conv5')
             .upsample(64, 64, name='HG5_resPool1_upSample'))


        (self.feed('HG5_resPool1_conv3',
                   'HG5_resPool1_skip',
                   'HG5_resPool1_upSample')
         .add(name='HG5_resPool1'))



# resPool2
        (self.feed('HG5_resPool1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_resPool2_batch1')
             .relu(name='HG5_resPool2_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG5_resPool2_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_resPool2_batch2')
             .relu(name='HG5_resPool2_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG5_resPool2_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_resPool2_batch3')
             .relu(name='HG5_resPool2_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_resPool2_conv3'))

        (self.feed('HG5_resPool1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_resPool2_skip'))

        (self.feed('HG5_resPool1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_resPool2_batch4')
             .relu(name='HG5_resPool2_relu4')
             .max_pool(2, 2, 2, 2, name='HG5_resPool2_pool')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='HG5_resPool2_conv4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_resPool2_batch5')
             .relu(name='HG5_resPool2_relu5')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='HG5_resPool2_conv5')
             .upsample(64, 64, name='HG5_resPool2_upSample'))


        (self.feed('HG5_resPool2_conv3',
                   'HG5_resPool2_skip',
                   'HG5_resPool2_upSample')
         .add(name='HG5_resPool2'))

# pool1
        (self.feed('HG5_input')
             .max_pool(2, 2, 2, 2, name='HG5_pool1'))


# res2
        (self.feed('HG5_pool1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res2_batch1')
             .relu(name='HG5_res2_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG5_res2_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res2_batch2')
             .relu(name='HG5_res2_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG5_res2_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res2_batch3')
             .relu(name='HG5_res2_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res2_conv3'))

        (self.feed('HG5_pool1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res2_skip'))

        (self.feed('HG5_res2_conv3',
                   'HG5_res2_skip')
         .add(name='HG5_res2'))

# res3
        (self.feed('HG5_res2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res3_batch1')
             .relu(name='HG5_res3_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG5_res3_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res3_batch2')
             .relu(name='HG5_res3_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG5_res3_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res3_batch3')
             .relu(name='HG5_res3_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res3_conv3'))

        (self.feed('HG5_res2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res3_skip'))

        (self.feed('HG5_res3_conv3',
                   'HG5_res3_skip')
         .add(name='HG5_res3'))

# resPool3
        (self.feed('HG5_res3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_resPool3_batch1')
             .relu(name='HG5_resPool3_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG5_resPool3_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_resPool3_batch2')
             .relu(name='HG5_resPool3_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG5_resPool3_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_resPool3_batch3')
             .relu(name='HG5_resPool3_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_resPool3_conv3'))

        (self.feed('HG5_res3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_resPool3_skip'))

        (self.feed('HG5_res3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_resPool3_batch4')
             .relu(name='HG5_resPool3_relu4')
             .max_pool(2, 2, 2, 2, name='HG5_resPool3_pool')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='HG5_resPool3_conv4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_resPool3_batch5')
             .relu(name='HG5_resPool3_relu5')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='HG5_resPool3_conv5')
             .upsample(32, 32, name='HG5_resPool3_upSample'))


        (self.feed('HG5_resPool3_conv3',
                   'HG5_resPool3_skip',
                   'HG5_resPool3_upSample')
         .add(name='HG5_resPool3'))




# pool2
        (self.feed('HG5_res2')
             .max_pool(2, 2, 2, 2, name='HG5_pool2'))


# res4
        (self.feed('HG5_pool2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res4_batch1')
             .relu(name='HG5_res4_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG5_res4_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res4_batch2')
             .relu(name='HG5_res4_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG5_res4_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res4_batch3')
             .relu(name='HG5_res4_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res4_conv3'))

        (self.feed('HG5_pool2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res4_skip'))

        (self.feed('HG5_res4_conv3',
                   'HG5_res4_skip')
         .add(name='HG5_res4'))
# id:013 max-pooling
        # (self.feed('HG5_pool3')
        #      .max_pool(2, 2, 2, 2, name='HG5_pool4'))


# res5
        (self.feed('HG5_res4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res5_batch1')
             .relu(name='HG5_res5_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG5_res5_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res5_batch2')
             .relu(name='HG5_res5_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG5_res5_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res5_batch3')
             .relu(name='HG5_res5_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res5_conv3'))

        (self.feed('HG5_res4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res5_skip'))

        (self.feed('HG5_res5_conv3',
                   'HG5_res5_skip')
         .add(name='HG5_res5'))


# pool3
        (self.feed('HG5_res4')
             .max_pool(2, 2, 2, 2, name='HG5_pool3'))


# res6
        (self.feed('HG5_pool3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res6_batch1')
             .relu(name='HG5_res6_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG5_res6_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res6_batch2')
             .relu(name='HG5_res6_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG5_res6_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res6_batch3')
             .relu(name='HG5_res6_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res6_conv3'))

        (self.feed('HG5_pool3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res6_skip'))

        (self.feed('HG5_res6_conv3',
                   'HG5_res6_skip')
         .add(name='HG5_res6'))

# res7
        (self.feed('HG5_res6')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res7_batch1')
             .relu(name='HG5_res7_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG5_res7_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res7_batch2')
             .relu(name='HG5_res7_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG5_res7_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res7_batch3')
             .relu(name='HG5_res7_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res7_conv3'))

        (self.feed('HG5_res6')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res7_skip'))

        (self.feed('HG5_res7_conv3',
                   'HG5_res7_skip')
         .add(name='HG5_res7'))


# pool4
        (self.feed('HG5_res6')
             .max_pool(2, 2, 2, 2, name='HG5_pool4'))

# res8
        (self.feed('HG5_pool4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res8_batch1')
             .relu(name='HG5_res8_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG5_res8_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res8_batch2')
             .relu(name='HG5_res8_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG5_res8_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res8_batch3')
             .relu(name='HG5_res8_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res8_conv3'))

        (self.feed('HG5_pool4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res8_skip'))

        (self.feed('HG5_res8_conv3',
                   'HG5_res8_skip')
         .add(name='HG5_res8'))

# res9
        (self.feed('HG5_res8')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res9_batch1')
             .relu(name='HG5_res9_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG5_res9_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res9_batch2')
             .relu(name='HG5_res9_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG5_res9_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res9_batch3')
             .relu(name='HG5_res9_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res9_conv3'))

        (self.feed('HG5_res8')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res9_skip'))

        (self.feed('HG5_res9_conv3',
                   'HG5_res9_skip')
         .add(name='HG5_res9'))

# res10
        (self.feed('HG5_res9')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res10_batch1')
             .relu(name='HG5_res10_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG5_res10_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res10_batch2')
             .relu(name='HG5_res10_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG5_res10_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res10_batch3')
             .relu(name='HG5_res10_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res10_conv3'))

        (self.feed('HG5_res9')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res10_skip'))

        (self.feed('HG5_res10_conv3',
                   'HG5_res10_skip')
         .add(name='HG5_res10'))


# upsample1
        (self.feed('HG5_res10')
             .upsample(8, 8, name='HG5_upSample1'))

# upsample1 + up1(Hg_res7)
        (self.feed('HG5_upSample1',
                   'HG5_res7')
         .add(name='HG5_add1'))


# res11
        (self.feed('HG5_add1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res11_batch1')
             .relu(name='HG5_res11_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG5_res11_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res11_batch2')
             .relu(name='HG5_res11_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG5_res11_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res11_batch3')
             .relu(name='HG5_res11_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res11_conv3'))

        (self.feed('HG5_add1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res11_skip'))

        (self.feed('HG5_res11_conv3',
                   'HG5_res11_skip')
         .add(name='HG5_res11'))


# upsample2
        (self.feed('HG5_res11')
             .upsample(16, 16, name='HG5_upSample2'))

# upsample2 + up1(Hg_res5)
        (self.feed('HG5_upSample2',
                   'HG5_res5')
         .add(name='HG5_add2'))


# res12
        (self.feed('HG5_add2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res12_batch1')
             .relu(name='HG5_res12_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG5_res12_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res12_batch2')
             .relu(name='HG5_res12_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG5_res12_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res12_batch3')
             .relu(name='HG5_res12_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res12_conv3'))

        (self.feed('HG5_add2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res12_skip'))

        (self.feed('HG5_res12_conv3',
                   'HG5_res12_skip')
         .add(name='HG5_res12'))


# upsample3
        (self.feed('HG5_res12')
             .upsample(32, 32, name='HG5_upSample3'))

# upsample3 + HG5_resPool3
        (self.feed('HG5_upSample3',
                   'HG5_resPool3')
         .add(name='HG5_add3'))


# res13
        (self.feed('HG5_add3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res13_batch1')
             .relu(name='HG5_res13_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG5_res13_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res13_batch2')
             .relu(name='HG5_res13_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG5_res13_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res13_batch3')
             .relu(name='HG5_res13_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res13_conv3'))

        (self.feed('HG5_add3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res13_skip'))

        (self.feed('HG5_res13_conv3',
                   'HG5_res13_skip')
         .add(name='HG5_res13'))


# upsample4
        (self.feed('HG5_res13')
             .upsample(64, 64, name='HG5_upSample4'))

# upsample4 + HG5_resPool2
        (self.feed('HG5_upSample4',
                   'HG5_resPool2')
         .add(name='HG5_add4'))

###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  Hourglass  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

################################# HG5 postprocess #################
# Linear layer to produce first set of predictions
# ll1
        (self.feed('HG5_add4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_linearfunc1_conv1', padding='SAME')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_linearfunc1_batch1')
             .relu(name='HG5_linearfunc1_relu'))
# ll2
        (self.feed('HG5_linearfunc1_relu')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_linearfunc2_conv1', padding='SAME')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_linearfunc2_batch1')
             .relu(name='HG5_linearfunc2_relu'))

# att itersize=3

# att U input: ll2
        (self.feed('HG5_linearfunc2_relu')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG5_att_U', padding='SAME'))
# att i=1 conv  C(1)
        with tf.variable_scope('HG5_att_share_param', reuse=False):
            (self.feed('HG5_att_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_att_spConv1', padding='SAME'))
# att Qtemp
        (self.feed('HG5_att_spConv1',
                   'HG5_att_U')
         .add(name='HG5_att_Qtemp1_add')
         .sigmoid(name='HG5_att_Qtemp1'))

# att i=2 conv  C(2)
        with tf.variable_scope('HG5_att_share_param', reuse=True):
            (self.feed('HG5_att_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_att_spConv1', padding='SAME'))
# att Qtemp2
        (self.feed('HG5_att_spConv1',
                   'HG5_att_U')
         .add(name='HG5_att_Qtemp2_add')
         .sigmoid(name='HG5_att_Qtemp2'))

# att i=3 conv  C(3)
        with tf.variable_scope('HG5_att_share_param', reuse=True):
            (self.feed('HG5_att_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_att_spConv1', padding='SAME'))
# att Qtemp
        (self.feed('HG5_att_spConv1',
                   'HG5_att_U')
         .add(name='HG5_att_Qtemp3_add')
         .sigmoid(name='HG5_att_Qtemp3'))
# att pfeat
        (self.feed('HG5_att_Qtemp3')
         .replicate(256, 3, name='HG5_att_pfeat_replicate'))

        (self.feed('HG5_linearfunc2_relu',
                   'HG5_att_pfeat_replicate')
         .multiply2(name='HG5_att_pfeat_multiply'))

# tmpOut 
        # (self.feed('HG5_att_pfeat_multiply')
        #      .conv(1, 1, n_classes, 1, 1, biased=True, relu=False, name='HG5_att_Heatmap', padding='SAME'))









# attention map for part1
# tmpOut1 att U input: HG5_att_pfeat_multiply
        (self.feed('HG5_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut1_U', padding='SAME'))
# tmpOut1 att i=1 conv  C(1)
        with tf.variable_scope('HG5_tmpOut1_share_param', reuse=False):
            (self.feed('HG5_tmpOut1_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut1_spConv1', padding='SAME'))
# tmpOut1 att Qtemp
        (self.feed('HG5_tmpOut1_spConv1',
                   'HG5_tmpOut1_U')
         .add(name='HG5_tmpOut1_Qtemp1_add')
         .sigmoid(name='HG5_tmpOut1_Qtemp1'))

# tmpOut1 att i=2 conv  C(2)
        with tf.variable_scope('HG5_tmpOut1_share_param', reuse=True):
            (self.feed('HG5_tmpOut1_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut1_spConv1', padding='SAME'))
# tmpOut1 att Qtemp2
        (self.feed('HG5_tmpOut1_spConv1',
                   'HG5_tmpOut1_U')
         .add(name='HG5_tmpOut1_Qtemp2_add')
         .sigmoid(name='HG5_tmpOut1_Qtemp2'))

# tmpOut1 att i=3 conv  C(3)
        with tf.variable_scope('HG5_tmpOut1_share_param', reuse=True):
            (self.feed('HG5_tmpOut1_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut1_spConv1', padding='SAME'))
# tmpOut1 att Qtemp
        (self.feed('HG5_tmpOut1_spConv1',
                   'HG5_tmpOut1_U')
         .add(name='HG5_tmpOut1_Qtemp3_add')
         .sigmoid(name='HG5_tmpOut1_Qtemp3'))
# tmpOut1 att pfeat
        (self.feed('HG5_tmpOut1_Qtemp3')
         .replicate(256, 3, name='HG5_tmpOut1_pfeat_replicate'))

        (self.feed('HG5_att_pfeat_multiply',
                   'HG5_tmpOut1_pfeat_replicate')
         .multiply2(name='HG5_tmpOut1_pfeat_multiply'))

        (self.feed('HG5_tmpOut1_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut1_s', padding='SAME'))




# attention map for part2
# tmpOut2 att U input: HG5_att_pfeat_multiply
        (self.feed('HG5_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut2_U', padding='SAME'))
# tmpOut2 att i=1 conv  C(1)
        with tf.variable_scope('HG5_tmpOut2_share_param', reuse=False):
            (self.feed('HG5_tmpOut2_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut2_spConv1', padding='SAME'))
# tmpOut2 att Qtemp
        (self.feed('HG5_tmpOut2_spConv1',
                   'HG5_tmpOut2_U')
         .add(name='HG5_tmpOut2_Qtemp1_add')
         .sigmoid(name='HG5_tmpOut2_Qtemp1'))

# tmpOut2 att i=2 conv  C(2)
        with tf.variable_scope('HG5_tmpOut2_share_param', reuse=True):
            (self.feed('HG5_tmpOut2_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut2_spConv1', padding='SAME'))
# tmpOut2 att Qtemp2
        (self.feed('HG5_tmpOut2_spConv1',
                   'HG5_tmpOut2_U')
         .add(name='HG5_tmpOut2_Qtemp2_add')
         .sigmoid(name='HG5_tmpOut2_Qtemp2'))

# tmpOut2 att i=3 conv  C(3)
        with tf.variable_scope('HG5_tmpOut2_share_param', reuse=True):
            (self.feed('HG5_tmpOut2_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut2_spConv1', padding='SAME'))
# tmpOut2 att Qtemp
        (self.feed('HG5_tmpOut2_spConv1',
                   'HG5_tmpOut2_U')
         .add(name='HG5_tmpOut2_Qtemp3_add')
         .sigmoid(name='HG5_tmpOut2_Qtemp3'))
# tmpOut2 att pfeat
        (self.feed('HG5_tmpOut2_Qtemp3')
         .replicate(256, 3, name='HG5_tmpOut2_pfeat_replicate'))

        (self.feed('HG5_att_pfeat_multiply',
                   'HG5_tmpOut2_pfeat_replicate')
         .multiply2(name='HG5_tmpOut2_pfeat_multiply'))

        (self.feed('HG5_tmpOut2_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut2_s', padding='SAME'))




# attention map for part3
# tmpOut3 att U input: HG5_att_pfeat_multiply
        (self.feed('HG5_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut3_U', padding='SAME'))
# tmpOut3 att i=1 conv  C(1)
        with tf.variable_scope('HG5_tmpOut3_share_param', reuse=False):
            (self.feed('HG5_tmpOut3_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut3_spConv1', padding='SAME'))
# tmpOut3 att Qtemp
        (self.feed('HG5_tmpOut3_spConv1',
                   'HG5_tmpOut3_U')
         .add(name='HG5_tmpOut3_Qtemp1_add')
         .sigmoid(name='HG5_tmpOut3_Qtemp1'))

# tmpOut3 att i=2 conv  C(2)
        with tf.variable_scope('HG5_tmpOut3_share_param', reuse=True):
            (self.feed('HG5_tmpOut3_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut3_spConv1', padding='SAME'))
# tmpOut3 att Qtemp2
        (self.feed('HG5_tmpOut3_spConv1',
                   'HG5_tmpOut3_U')
         .add(name='HG5_tmpOut3_Qtemp2_add')
         .sigmoid(name='HG5_tmpOut3_Qtemp2'))

# tmpOut3 att i=3 conv  C(3)
        with tf.variable_scope('HG5_tmpOut3_share_param', reuse=True):
            (self.feed('HG5_tmpOut3_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut3_spConv1', padding='SAME'))
# tmpOut3 att Qtemp
        (self.feed('HG5_tmpOut3_spConv1',
                   'HG5_tmpOut3_U')
         .add(name='HG5_tmpOut3_Qtemp3_add')
         .sigmoid(name='HG5_tmpOut3_Qtemp3'))
# tmpOut3 att pfeat
        (self.feed('HG5_tmpOut3_Qtemp3')
         .replicate(256, 3, name='HG5_tmpOut3_pfeat_replicate'))

        (self.feed('HG5_att_pfeat_multiply',
                   'HG5_tmpOut3_pfeat_replicate')
         .multiply2(name='HG5_tmpOut3_pfeat_multiply'))

        (self.feed('HG5_tmpOut3_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut3_s', padding='SAME'))






# attention map for part4
# tmpOut4 att U input: HG5_att_pfeat_multiply
        (self.feed('HG5_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut4_U', padding='SAME'))
# tmpOut4 att i=1 conv  C(1)
        with tf.variable_scope('HG5_tmpOut4_share_param', reuse=False):
            (self.feed('HG5_tmpOut4_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut4_spConv1', padding='SAME'))
# tmpOut4 att Qtemp
        (self.feed('HG5_tmpOut4_spConv1',
                   'HG5_tmpOut4_U')
         .add(name='HG5_tmpOut4_Qtemp1_add')
         .sigmoid(name='HG5_tmpOut4_Qtemp1'))

# tmpOut4 att i=2 conv  C(2)
        with tf.variable_scope('HG5_tmpOut4_share_param', reuse=True):
            (self.feed('HG5_tmpOut4_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut4_spConv1', padding='SAME'))
# tmpOut4 att Qtemp2
        (self.feed('HG5_tmpOut4_spConv1',
                   'HG5_tmpOut4_U')
         .add(name='HG5_tmpOut4_Qtemp2_add')
         .sigmoid(name='HG5_tmpOut4_Qtemp2'))

# tmpOut4 att i=3 conv  C(3)
        with tf.variable_scope('HG5_tmpOut4_share_param', reuse=True):
            (self.feed('HG5_tmpOut4_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut4_spConv1', padding='SAME'))
# tmpOut4 att Qtemp
        (self.feed('HG5_tmpOut4_spConv1',
                   'HG5_tmpOut4_U')
         .add(name='HG5_tmpOut4_Qtemp3_add')
         .sigmoid(name='HG5_tmpOut4_Qtemp3'))
# tmpOut4 att pfeat
        (self.feed('HG5_tmpOut4_Qtemp3')
         .replicate(256, 3, name='HG5_tmpOut4_pfeat_replicate'))

        (self.feed('HG5_att_pfeat_multiply',
                   'HG5_tmpOut4_pfeat_replicate')
         .multiply2(name='HG5_tmpOut4_pfeat_multiply'))

        (self.feed('HG5_tmpOut4_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut4_s', padding='SAME'))









# attention map for part5
# tmpOut5 att U input: HG5_att_pfeat_multiply
        (self.feed('HG5_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut5_U', padding='SAME'))
# tmpOut5 att i=1 conv  C(1)
        with tf.variable_scope('HG5_tmpOut5_share_param', reuse=False):
            (self.feed('HG5_tmpOut5_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut5_spConv1', padding='SAME'))
# tmpOut5 att Qtemp
        (self.feed('HG5_tmpOut5_spConv1',
                   'HG5_tmpOut5_U')
         .add(name='HG5_tmpOut5_Qtemp1_add')
         .sigmoid(name='HG5_tmpOut5_Qtemp1'))

# tmpOut5 att i=2 conv  C(2)
        with tf.variable_scope('HG5_tmpOut5_share_param', reuse=True):
            (self.feed('HG5_tmpOut5_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut5_spConv1', padding='SAME'))
# tmpOut5 att Qtemp2
        (self.feed('HG5_tmpOut5_spConv1',
                   'HG5_tmpOut5_U')
         .add(name='HG5_tmpOut5_Qtemp2_add')
         .sigmoid(name='HG5_tmpOut5_Qtemp2'))

# tmpOut5 att i=3 conv  C(3)
        with tf.variable_scope('HG5_tmpOut5_share_param', reuse=True):
            (self.feed('HG5_tmpOut5_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut5_spConv1', padding='SAME'))
# tmpOut5 att Qtemp
        (self.feed('HG5_tmpOut5_spConv1',
                   'HG5_tmpOut5_U')
         .add(name='HG5_tmpOut5_Qtemp3_add')
         .sigmoid(name='HG5_tmpOut5_Qtemp3'))
# tmpOut5 att pfeat
        (self.feed('HG5_tmpOut5_Qtemp3')
         .replicate(256, 3, name='HG5_tmpOut5_pfeat_replicate'))

        (self.feed('HG5_att_pfeat_multiply',
                   'HG5_tmpOut5_pfeat_replicate')
         .multiply2(name='HG5_tmpOut5_pfeat_multiply'))

        (self.feed('HG5_tmpOut5_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut5_s', padding='SAME'))










# attention map for part6
# tmpOut6 att U input: HG5_att_pfeat_multiply
        (self.feed('HG5_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut6_U', padding='SAME'))
# tmpOut6 att i=1 conv  C(1)
        with tf.variable_scope('HG5_tmpOut6_share_param', reuse=False):
            (self.feed('HG5_tmpOut6_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut6_spConv1', padding='SAME'))
# tmpOut6 att Qtemp
        (self.feed('HG5_tmpOut6_spConv1',
                   'HG5_tmpOut6_U')
         .add(name='HG5_tmpOut6_Qtemp1_add')
         .sigmoid(name='HG5_tmpOut6_Qtemp1'))

# tmpOut6 att i=2 conv  C(2)
        with tf.variable_scope('HG5_tmpOut6_share_param', reuse=True):
            (self.feed('HG5_tmpOut6_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut6_spConv1', padding='SAME'))
# tmpOut6 att Qtemp2
        (self.feed('HG5_tmpOut6_spConv1',
                   'HG5_tmpOut6_U')
         .add(name='HG5_tmpOut6_Qtemp2_add')
         .sigmoid(name='HG5_tmpOut6_Qtemp2'))

# tmpOut6 att i=3 conv  C(3)
        with tf.variable_scope('HG5_tmpOut6_share_param', reuse=True):
            (self.feed('HG5_tmpOut6_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut6_spConv1', padding='SAME'))
# tmpOut6 att Qtemp
        (self.feed('HG5_tmpOut6_spConv1',
                   'HG5_tmpOut6_U')
         .add(name='HG5_tmpOut6_Qtemp3_add')
         .sigmoid(name='HG5_tmpOut6_Qtemp3'))
# tmpOut6 att pfeat
        (self.feed('HG5_tmpOut6_Qtemp3')
         .replicate(256, 3, name='HG5_tmpOut6_pfeat_replicate'))

        (self.feed('HG5_att_pfeat_multiply',
                   'HG5_tmpOut6_pfeat_replicate')
         .multiply2(name='HG5_tmpOut6_pfeat_multiply'))

        (self.feed('HG5_tmpOut6_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut6_s', padding='SAME'))





# attention map for part7
# tmpOut7 att U input: HG5_att_pfeat_multiply
        (self.feed('HG5_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut7_U', padding='SAME'))
# tmpOut7 att i=1 conv  C(1)
        with tf.variable_scope('HG5_tmpOut7_share_param', reuse=False):
            (self.feed('HG5_tmpOut7_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut7_spConv1', padding='SAME'))
# tmpOut7 att Qtemp
        (self.feed('HG5_tmpOut7_spConv1',
                   'HG5_tmpOut7_U')
         .add(name='HG5_tmpOut7_Qtemp1_add')
         .sigmoid(name='HG5_tmpOut7_Qtemp1'))

# tmpOut7 att i=2 conv  C(2)
        with tf.variable_scope('HG5_tmpOut7_share_param', reuse=True):
            (self.feed('HG5_tmpOut7_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut7_spConv1', padding='SAME'))
# tmpOut7 att Qtemp2
        (self.feed('HG5_tmpOut7_spConv1',
                   'HG5_tmpOut7_U')
         .add(name='HG5_tmpOut7_Qtemp2_add')
         .sigmoid(name='HG5_tmpOut7_Qtemp2'))

# tmpOut7 att i=3 conv  C(3)
        with tf.variable_scope('HG5_tmpOut7_share_param', reuse=True):
            (self.feed('HG5_tmpOut7_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut7_spConv1', padding='SAME'))
# tmpOut7 att Qtemp
        (self.feed('HG5_tmpOut7_spConv1',
                   'HG5_tmpOut7_U')
         .add(name='HG5_tmpOut7_Qtemp3_add')
         .sigmoid(name='HG5_tmpOut7_Qtemp3'))
# tmpOut7 att pfeat
        (self.feed('HG5_tmpOut7_Qtemp3')
         .replicate(256, 3, name='HG5_tmpOut7_pfeat_replicate'))

        (self.feed('HG5_att_pfeat_multiply',
                   'HG5_tmpOut7_pfeat_replicate')
         .multiply2(name='HG5_tmpOut7_pfeat_multiply'))

        (self.feed('HG5_tmpOut7_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut7_s', padding='SAME'))








# attention map for part8
# tmpOut8 att U input: HG5_att_pfeat_multiply
        (self.feed('HG5_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut8_U', padding='SAME'))
# tmpOut8 att i=1 conv  C(1)
        with tf.variable_scope('HG5_tmpOut8_share_param', reuse=False):
            (self.feed('HG5_tmpOut8_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut8_spConv1', padding='SAME'))
# tmpOut8 att Qtemp
        (self.feed('HG5_tmpOut8_spConv1',
                   'HG5_tmpOut8_U')
         .add(name='HG5_tmpOut8_Qtemp1_add')
         .sigmoid(name='HG5_tmpOut8_Qtemp1'))

# tmpOut8 att i=2 conv  C(2)
        with tf.variable_scope('HG5_tmpOut8_share_param', reuse=True):
            (self.feed('HG5_tmpOut8_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut8_spConv1', padding='SAME'))
# tmpOut8 att Qtemp2
        (self.feed('HG5_tmpOut8_spConv1',
                   'HG5_tmpOut8_U')
         .add(name='HG5_tmpOut8_Qtemp2_add')
         .sigmoid(name='HG5_tmpOut8_Qtemp2'))

# tmpOut8 att i=3 conv  C(3)
        with tf.variable_scope('HG5_tmpOut8_share_param', reuse=True):
            (self.feed('HG5_tmpOut8_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut8_spConv1', padding='SAME'))
# tmpOut8 att Qtemp
        (self.feed('HG5_tmpOut8_spConv1',
                   'HG5_tmpOut8_U')
         .add(name='HG5_tmpOut8_Qtemp3_add')
         .sigmoid(name='HG5_tmpOut8_Qtemp3'))
# tmpOut8 att pfeat
        (self.feed('HG5_tmpOut8_Qtemp3')
         .replicate(256, 3, name='HG5_tmpOut8_pfeat_replicate'))

        (self.feed('HG5_att_pfeat_multiply',
                   'HG5_tmpOut8_pfeat_replicate')
         .multiply2(name='HG5_tmpOut8_pfeat_multiply'))

        (self.feed('HG5_tmpOut8_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut8_s', padding='SAME'))











# attention map for part9
# tmpOut9 att U input: HG5_att_pfeat_multiply
        (self.feed('HG5_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut9_U', padding='SAME'))
# tmpOut9 att i=1 conv  C(1)
        with tf.variable_scope('HG5_tmpOut9_share_param', reuse=False):
            (self.feed('HG5_tmpOut9_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut9_spConv1', padding='SAME'))
# tmpOut9 att Qtemp
        (self.feed('HG5_tmpOut9_spConv1',
                   'HG5_tmpOut9_U')
         .add(name='HG5_tmpOut9_Qtemp1_add')
         .sigmoid(name='HG5_tmpOut9_Qtemp1'))

# tmpOut9 att i=2 conv  C(2)
        with tf.variable_scope('HG5_tmpOut9_share_param', reuse=True):
            (self.feed('HG5_tmpOut9_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut9_spConv1', padding='SAME'))
# tmpOut9 att Qtemp2
        (self.feed('HG5_tmpOut9_spConv1',
                   'HG5_tmpOut9_U')
         .add(name='HG5_tmpOut9_Qtemp2_add')
         .sigmoid(name='HG5_tmpOut9_Qtemp2'))

# tmpOut9 att i=3 conv  C(3)
        with tf.variable_scope('HG5_tmpOut9_share_param', reuse=True):
            (self.feed('HG5_tmpOut9_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut9_spConv1', padding='SAME'))
# tmpOut9 att Qtemp
        (self.feed('HG5_tmpOut9_spConv1',
                   'HG5_tmpOut9_U')
         .add(name='HG5_tmpOut9_Qtemp3_add')
         .sigmoid(name='HG5_tmpOut9_Qtemp3'))
# tmpOut9 att pfeat
        (self.feed('HG5_tmpOut9_Qtemp3')
         .replicate(256, 3, name='HG5_tmpOut9_pfeat_replicate'))

        (self.feed('HG5_att_pfeat_multiply',
                   'HG5_tmpOut9_pfeat_replicate')
         .multiply2(name='HG5_tmpOut9_pfeat_multiply'))

        (self.feed('HG5_tmpOut9_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut9_s', padding='SAME'))












# attention map for part10
# tmpOut10 att U input: HG5_att_pfeat_multiply
        (self.feed('HG5_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut10_U', padding='SAME'))
# tmpOut10 att i=1 conv  C(1)
        with tf.variable_scope('HG5_tmpOut10_share_param', reuse=False):
            (self.feed('HG5_tmpOut10_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut10_spConv1', padding='SAME'))
# tmpOut10 att Qtemp
        (self.feed('HG5_tmpOut10_spConv1',
                   'HG5_tmpOut10_U')
         .add(name='HG5_tmpOut10_Qtemp1_add')
         .sigmoid(name='HG5_tmpOut10_Qtemp1'))

# tmpOut10 att i=2 conv  C(2)
        with tf.variable_scope('HG5_tmpOut10_share_param', reuse=True):
            (self.feed('HG5_tmpOut10_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut10_spConv1', padding='SAME'))
# tmpOut10 att Qtemp2
        (self.feed('HG5_tmpOut10_spConv1',
                   'HG5_tmpOut10_U')
         .add(name='HG5_tmpOut10_Qtemp2_add')
         .sigmoid(name='HG5_tmpOut10_Qtemp2'))

# tmpOut10 att i=3 conv  C(3)
        with tf.variable_scope('HG5_tmpOut10_share_param', reuse=True):
            (self.feed('HG5_tmpOut10_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut10_spConv1', padding='SAME'))
# tmpOut10 att Qtemp
        (self.feed('HG5_tmpOut10_spConv1',
                   'HG5_tmpOut10_U')
         .add(name='HG5_tmpOut10_Qtemp3_add')
         .sigmoid(name='HG5_tmpOut10_Qtemp3'))
# tmpOut10 att pfeat
        (self.feed('HG5_tmpOut10_Qtemp3')
         .replicate(256, 3, name='HG5_tmpOut10_pfeat_replicate'))

        (self.feed('HG5_att_pfeat_multiply',
                   'HG5_tmpOut10_pfeat_replicate')
         .multiply2(name='HG5_tmpOut10_pfeat_multiply'))

        (self.feed('HG5_tmpOut10_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut10_s', padding='SAME'))










# attention map for part11
# tmpOut11 att U input: HG5_att_pfeat_multiply
        (self.feed('HG5_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut11_U', padding='SAME'))
# tmpOut11 att i=1 conv  C(1)
        with tf.variable_scope('HG5_tmpOut11_share_param', reuse=False):
            (self.feed('HG5_tmpOut11_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut11_spConv1', padding='SAME'))
# tmpOut11 att Qtemp
        (self.feed('HG5_tmpOut11_spConv1',
                   'HG5_tmpOut11_U')
         .add(name='HG5_tmpOut11_Qtemp1_add')
         .sigmoid(name='HG5_tmpOut11_Qtemp1'))

# tmpOut11 att i=2 conv  C(2)
        with tf.variable_scope('HG5_tmpOut11_share_param', reuse=True):
            (self.feed('HG5_tmpOut11_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut11_spConv1', padding='SAME'))
# tmpOut11 att Qtemp2
        (self.feed('HG5_tmpOut11_spConv1',
                   'HG5_tmpOut11_U')
         .add(name='HG5_tmpOut11_Qtemp2_add')
         .sigmoid(name='HG5_tmpOut11_Qtemp2'))

# tmpOut11 att i=3 conv  C(3)
        with tf.variable_scope('HG5_tmpOut11_share_param', reuse=True):
            (self.feed('HG5_tmpOut11_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut11_spConv1', padding='SAME'))
# tmpOut11 att Qtemp
        (self.feed('HG5_tmpOut11_spConv1',
                   'HG5_tmpOut11_U')
         .add(name='HG5_tmpOut11_Qtemp3_add')
         .sigmoid(name='HG5_tmpOut11_Qtemp3'))
# tmpOut11 att pfeat
        (self.feed('HG5_tmpOut11_Qtemp3')
         .replicate(256, 3, name='HG5_tmpOut11_pfeat_replicate'))

        (self.feed('HG5_att_pfeat_multiply',
                   'HG5_tmpOut11_pfeat_replicate')
         .multiply2(name='HG5_tmpOut11_pfeat_multiply'))

        (self.feed('HG5_tmpOut11_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut11_s', padding='SAME'))











# attention map for part12
# tmpOut12 att U input: HG5_att_pfeat_multiply
        (self.feed('HG5_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut12_U', padding='SAME'))
# tmpOut12 att i=1 conv  C(1)
        with tf.variable_scope('HG5_tmpOut12_share_param', reuse=False):
            (self.feed('HG5_tmpOut12_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut12_spConv1', padding='SAME'))
# tmpOut12 att Qtemp
        (self.feed('HG5_tmpOut12_spConv1',
                   'HG5_tmpOut12_U')
         .add(name='HG5_tmpOut12_Qtemp1_add')
         .sigmoid(name='HG5_tmpOut12_Qtemp1'))

# tmpOut12 att i=2 conv  C(2)
        with tf.variable_scope('HG5_tmpOut12_share_param', reuse=True):
            (self.feed('HG5_tmpOut12_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut12_spConv1', padding='SAME'))
# tmpOut12 att Qtemp2
        (self.feed('HG5_tmpOut12_spConv1',
                   'HG5_tmpOut12_U')
         .add(name='HG5_tmpOut12_Qtemp2_add')
         .sigmoid(name='HG5_tmpOut12_Qtemp2'))

# tmpOut12 att i=3 conv  C(3)
        with tf.variable_scope('HG5_tmpOut12_share_param', reuse=True):
            (self.feed('HG5_tmpOut12_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut12_spConv1', padding='SAME'))
# tmpOut12 att Qtemp
        (self.feed('HG5_tmpOut12_spConv1',
                   'HG5_tmpOut12_U')
         .add(name='HG5_tmpOut12_Qtemp3_add')
         .sigmoid(name='HG5_tmpOut12_Qtemp3'))
# tmpOut12 att pfeat
        (self.feed('HG5_tmpOut12_Qtemp3')
         .replicate(256, 3, name='HG5_tmpOut12_pfeat_replicate'))

        (self.feed('HG5_att_pfeat_multiply',
                   'HG5_tmpOut12_pfeat_replicate')
         .multiply2(name='HG5_tmpOut12_pfeat_multiply'))

        (self.feed('HG5_tmpOut12_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut12_s', padding='SAME'))










# attention map for part13
# tmpOut13 att U input: HG5_att_pfeat_multiply
        (self.feed('HG5_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut13_U', padding='SAME'))
# tmpOut13 att i=1 conv  C(1)
        with tf.variable_scope('HG5_tmpOut13_share_param', reuse=False):
            (self.feed('HG5_tmpOut13_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut13_spConv1', padding='SAME'))
# tmpOut13 att Qtemp
        (self.feed('HG5_tmpOut13_spConv1',
                   'HG5_tmpOut13_U')
         .add(name='HG5_tmpOut13_Qtemp1_add')
         .sigmoid(name='HG5_tmpOut13_Qtemp1'))

# tmpOut13 att i=2 conv  C(2)
        with tf.variable_scope('HG5_tmpOut13_share_param', reuse=True):
            (self.feed('HG5_tmpOut13_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut13_spConv1', padding='SAME'))
# tmpOut13 att Qtemp2
        (self.feed('HG5_tmpOut13_spConv1',
                   'HG5_tmpOut13_U')
         .add(name='HG5_tmpOut13_Qtemp2_add')
         .sigmoid(name='HG5_tmpOut13_Qtemp2'))

# tmpOut13 att i=3 conv  C(3)
        with tf.variable_scope('HG5_tmpOut13_share_param', reuse=True):
            (self.feed('HG5_tmpOut13_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut13_spConv1', padding='SAME'))
# tmpOut13 att Qtemp
        (self.feed('HG5_tmpOut13_spConv1',
                   'HG5_tmpOut13_U')
         .add(name='HG5_tmpOut13_Qtemp3_add')
         .sigmoid(name='HG5_tmpOut13_Qtemp3'))
# tmpOut13 att pfeat
        (self.feed('HG5_tmpOut13_Qtemp3')
         .replicate(256, 3, name='HG5_tmpOut13_pfeat_replicate'))

        (self.feed('HG5_att_pfeat_multiply',
                   'HG5_tmpOut13_pfeat_replicate')
         .multiply2(name='HG5_tmpOut13_pfeat_multiply'))

        (self.feed('HG5_tmpOut13_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut13_s', padding='SAME'))










# attention map for part14
# tmpOut14 att U input: HG5_att_pfeat_multiply
        (self.feed('HG5_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut14_U', padding='SAME'))
# tmpOut14 att i=1 conv  C(1)
        with tf.variable_scope('HG5_tmpOut14_share_param', reuse=False):
            (self.feed('HG5_tmpOut14_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut14_spConv1', padding='SAME'))
# tmpOut14 att Qtemp
        (self.feed('HG5_tmpOut14_spConv1',
                   'HG5_tmpOut14_U')
         .add(name='HG5_tmpOut14_Qtemp1_add')
         .sigmoid(name='HG5_tmpOut14_Qtemp1'))

# tmpOut14 att i=2 conv  C(2)
        with tf.variable_scope('HG5_tmpOut14_share_param', reuse=True):
            (self.feed('HG5_tmpOut14_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut14_spConv1', padding='SAME'))
# tmpOut14 att Qtemp2
        (self.feed('HG5_tmpOut14_spConv1',
                   'HG5_tmpOut14_U')
         .add(name='HG5_tmpOut14_Qtemp2_add')
         .sigmoid(name='HG5_tmpOut14_Qtemp2'))

# tmpOut14 att i=3 conv  C(3)
        with tf.variable_scope('HG5_tmpOut14_share_param', reuse=True):
            (self.feed('HG5_tmpOut14_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut14_spConv1', padding='SAME'))
# tmpOut14 att Qtemp
        (self.feed('HG5_tmpOut14_spConv1',
                   'HG5_tmpOut14_U')
         .add(name='HG5_tmpOut14_Qtemp3_add')
         .sigmoid(name='HG5_tmpOut14_Qtemp3'))
# tmpOut14 att pfeat
        (self.feed('HG5_tmpOut14_Qtemp3')
         .replicate(256, 3, name='HG5_tmpOut14_pfeat_replicate'))

        (self.feed('HG5_att_pfeat_multiply',
                   'HG5_tmpOut14_pfeat_replicate')
         .multiply2(name='HG5_tmpOut14_pfeat_multiply'))

        (self.feed('HG5_tmpOut14_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut14_s', padding='SAME'))










# attention map for part15
# tmpOut15 att U input: HG5_att_pfeat_multiply
        (self.feed('HG5_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut15_U', padding='SAME'))
# tmpOut15 att i=1 conv  C(1)
        with tf.variable_scope('HG5_tmpOut15_share_param', reuse=False):
            (self.feed('HG5_tmpOut15_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut15_spConv1', padding='SAME'))
# tmpOut15 att Qtemp
        (self.feed('HG5_tmpOut15_spConv1',
                   'HG5_tmpOut15_U')
         .add(name='HG5_tmpOut15_Qtemp1_add')
         .sigmoid(name='HG5_tmpOut15_Qtemp1'))

# tmpOut15 att i=2 conv  C(2)
        with tf.variable_scope('HG5_tmpOut15_share_param', reuse=True):
            (self.feed('HG5_tmpOut15_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut15_spConv1', padding='SAME'))
# tmpOut15 att Qtemp2
        (self.feed('HG5_tmpOut15_spConv1',
                   'HG5_tmpOut15_U')
         .add(name='HG5_tmpOut15_Qtemp2_add')
         .sigmoid(name='HG5_tmpOut15_Qtemp2'))

# tmpOut15 att i=3 conv  C(3)
        with tf.variable_scope('HG5_tmpOut15_share_param', reuse=True):
            (self.feed('HG5_tmpOut15_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut15_spConv1', padding='SAME'))
# tmpOut15 att Qtemp
        (self.feed('HG5_tmpOut15_spConv1',
                   'HG5_tmpOut15_U')
         .add(name='HG5_tmpOut15_Qtemp3_add')
         .sigmoid(name='HG5_tmpOut15_Qtemp3'))
# tmpOut15 att pfeat
        (self.feed('HG5_tmpOut15_Qtemp3')
         .replicate(256, 3, name='HG5_tmpOut15_pfeat_replicate'))

        (self.feed('HG5_att_pfeat_multiply',
                   'HG5_tmpOut15_pfeat_replicate')
         .multiply2(name='HG5_tmpOut15_pfeat_multiply'))

        (self.feed('HG5_tmpOut15_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut15_s', padding='SAME'))










# attention map for part16
# tmpOut16 att U input: HG5_att_pfeat_multiply
        (self.feed('HG5_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut16_U', padding='SAME'))
# tmpOut16 att i=1 conv  C(1)
        with tf.variable_scope('HG5_tmpOut16_share_param', reuse=False):
            (self.feed('HG5_tmpOut16_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut16_spConv1', padding='SAME'))
# tmpOut16 att Qtemp
        (self.feed('HG5_tmpOut16_spConv1',
                   'HG5_tmpOut16_U')
         .add(name='HG5_tmpOut16_Qtemp1_add')
         .sigmoid(name='HG5_tmpOut16_Qtemp1'))

# tmpOut16 att i=2 conv  C(2)
        with tf.variable_scope('HG5_tmpOut16_share_param', reuse=True):
            (self.feed('HG5_tmpOut16_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut16_spConv1', padding='SAME'))
# tmpOut16 att Qtemp2
        (self.feed('HG5_tmpOut16_spConv1',
                   'HG5_tmpOut16_U')
         .add(name='HG5_tmpOut16_Qtemp2_add')
         .sigmoid(name='HG5_tmpOut16_Qtemp2'))

# tmpOut16 att i=3 conv  C(3)
        with tf.variable_scope('HG5_tmpOut16_share_param', reuse=True):
            (self.feed('HG5_tmpOut16_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut16_spConv1', padding='SAME'))
# tmpOut16 att Qtemp
        (self.feed('HG5_tmpOut16_spConv1',
                   'HG5_tmpOut16_U')
         .add(name='HG5_tmpOut16_Qtemp3_add')
         .sigmoid(name='HG5_tmpOut16_Qtemp3'))
# tmpOut16 att pfeat
        (self.feed('HG5_tmpOut16_Qtemp3')
         .replicate(256, 3, name='HG5_tmpOut16_pfeat_replicate'))

        (self.feed('HG5_att_pfeat_multiply',
                   'HG5_tmpOut16_pfeat_replicate')
         .multiply2(name='HG5_tmpOut16_pfeat_multiply'))

        (self.feed('HG5_tmpOut16_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG5_tmpOut16_s', padding='SAME'))


        (self.feed('HG5_tmpOut1_s',
                   'HG5_tmpOut2_s',
                   'HG5_tmpOut3_s',
                   'HG5_tmpOut4_s',
                   'HG5_tmpOut5_s',
                   'HG5_tmpOut6_s',
                   'HG5_tmpOut7_s',
                   'HG5_tmpOut8_s',
                   'HG5_tmpOut9_s',
                   'HG5_tmpOut10_s',
                   'HG5_tmpOut11_s',
                   'HG5_tmpOut12_s',
                   'HG5_tmpOut13_s',
                   'HG5_tmpOut14_s',
                   'HG5_tmpOut15_s',
                   'HG5_tmpOut16_s')
         .stack(axis = 3,  name='HG5_Heatmap'))        

        (self.feed('HG5_tmpOut1_s')
            .printLayer(name='printLayer14'))
        (self.feed('HG5_tmpOut2_s')
            .printLayer(name='printLayer15'))
        (self.feed('HG5_tmpOut3_s')
            .printLayer(name='printLayer16'))
        (self.feed('HG5_tmpOut4_s')
            .printLayer(name='printLayer17'))
        (self.feed('HG5_Heatmap')
            .printLayer(name='printLayer18'))
###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ HG5 postprocess ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

############################# generate input for next HG ##########

# outmap
        (self.feed('HG5_Heatmap')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_outmap', padding='SAME'))
# ll3
        (self.feed('HG5_linearfunc1_relu')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_linearfunc3_conv1', padding='SAME')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_linearfunc3_batch1')
             .relu(name='HG5_linearfunc3_relu'))
# tmointer
        (self.feed('HG5_input',
                   'HG5_outmap',
                   'HG5_linearfunc3_relu')
         .add(name='HG6_input'))

###^^^^^^^^^^^^^^^^^^^^^^^^^^^ generate input for next HG ^^^^^^^^^^^^^^^^^^^^^^^^^^^^###
































































































#######################################  HG6  #########################################
                 ####################  Hourglass  #####################
# res1
        (self.feed('HG6_input')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res1_batch1')
             .relu(name='HG6_res1_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG6_res1_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res1_batch2')
             .relu(name='HG6_res1_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG6_res1_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res1_batch3')
             .relu(name='HG6_res1_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res1_conv3'))

        (self.feed('HG6_input')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res1_skip'))

        (self.feed('HG6_res1_conv3',
                   'HG6_res1_skip')
         .add(name='HG6_res1'))

# resPool1
        (self.feed('HG6_res1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_resPool1_batch1')
             .relu(name='HG6_resPool1_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG6_resPool1_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_resPool1_batch2')
             .relu(name='HG6_resPool1_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG6_resPool1_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_resPool1_batch3')
             .relu(name='HG6_resPool1_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_resPool1_conv3'))

        (self.feed('HG6_res1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_resPool1_skip'))

        (self.feed('HG6_res1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_resPool1_batch4')
             .relu(name='HG6_resPool1_relu4')
             .max_pool(2, 2, 2, 2, name='HG6_resPool1_pool')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='HG6_resPool1_conv4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_resPool1_batch5')
             .relu(name='HG6_resPool1_relu5')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='HG6_resPool1_conv5')
             .upsample(64, 64, name='HG6_resPool1_upSample'))


        (self.feed('HG6_resPool1_conv3',
                   'HG6_resPool1_skip',
                   'HG6_resPool1_upSample')
         .add(name='HG6_resPool1'))



# resPool2
        (self.feed('HG6_resPool1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_resPool2_batch1')
             .relu(name='HG6_resPool2_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG6_resPool2_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_resPool2_batch2')
             .relu(name='HG6_resPool2_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG6_resPool2_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_resPool2_batch3')
             .relu(name='HG6_resPool2_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_resPool2_conv3'))

        (self.feed('HG6_resPool1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_resPool2_skip'))

        (self.feed('HG6_resPool1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_resPool2_batch4')
             .relu(name='HG6_resPool2_relu4')
             .max_pool(2, 2, 2, 2, name='HG6_resPool2_pool')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='HG6_resPool2_conv4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_resPool2_batch5')
             .relu(name='HG6_resPool2_relu5')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='HG6_resPool2_conv5')
             .upsample(64, 64, name='HG6_resPool2_upSample'))


        (self.feed('HG6_resPool2_conv3',
                   'HG6_resPool2_skip',
                   'HG6_resPool2_upSample')
         .add(name='HG6_resPool2'))

# pool1
        (self.feed('HG6_input')
             .max_pool(2, 2, 2, 2, name='HG6_pool1'))


# res2
        (self.feed('HG6_pool1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res2_batch1')
             .relu(name='HG6_res2_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG6_res2_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res2_batch2')
             .relu(name='HG6_res2_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG6_res2_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res2_batch3')
             .relu(name='HG6_res2_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res2_conv3'))

        (self.feed('HG6_pool1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res2_skip'))

        (self.feed('HG6_res2_conv3',
                   'HG6_res2_skip')
         .add(name='HG6_res2'))

# res3
        (self.feed('HG6_res2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res3_batch1')
             .relu(name='HG6_res3_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG6_res3_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res3_batch2')
             .relu(name='HG6_res3_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG6_res3_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res3_batch3')
             .relu(name='HG6_res3_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res3_conv3'))

        (self.feed('HG6_res2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res3_skip'))

        (self.feed('HG6_res3_conv3',
                   'HG6_res3_skip')
         .add(name='HG6_res3'))

# resPool3
        (self.feed('HG6_res3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_resPool3_batch1')
             .relu(name='HG6_resPool3_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG6_resPool3_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_resPool3_batch2')
             .relu(name='HG6_resPool3_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG6_resPool3_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_resPool3_batch3')
             .relu(name='HG6_resPool3_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_resPool3_conv3'))

        (self.feed('HG6_res3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_resPool3_skip'))

        (self.feed('HG6_res3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_resPool3_batch4')
             .relu(name='HG6_resPool3_relu4')
             .max_pool(2, 2, 2, 2, name='HG6_resPool3_pool')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='HG6_resPool3_conv4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_resPool3_batch5')
             .relu(name='HG6_resPool3_relu5')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='HG6_resPool3_conv5')
             .upsample(32, 32, name='HG6_resPool3_upSample'))


        (self.feed('HG6_resPool3_conv3',
                   'HG6_resPool3_skip',
                   'HG6_resPool3_upSample')
         .add(name='HG6_resPool3'))




# pool2
        (self.feed('HG6_res2')
             .max_pool(2, 2, 2, 2, name='HG6_pool2'))


# res4
        (self.feed('HG6_pool2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res4_batch1')
             .relu(name='HG6_res4_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG6_res4_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res4_batch2')
             .relu(name='HG6_res4_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG6_res4_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res4_batch3')
             .relu(name='HG6_res4_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res4_conv3'))

        (self.feed('HG6_pool2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res4_skip'))

        (self.feed('HG6_res4_conv3',
                   'HG6_res4_skip')
         .add(name='HG6_res4'))
# id:013 max-pooling
        # (self.feed('HG6_pool3')
        #      .max_pool(2, 2, 2, 2, name='HG6_pool4'))


# res5
        (self.feed('HG6_res4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res5_batch1')
             .relu(name='HG6_res5_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG6_res5_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res5_batch2')
             .relu(name='HG6_res5_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG6_res5_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res5_batch3')
             .relu(name='HG6_res5_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res5_conv3'))

        (self.feed('HG6_res4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res5_skip'))

        (self.feed('HG6_res5_conv3',
                   'HG6_res5_skip')
         .add(name='HG6_res5'))


# pool3
        (self.feed('HG6_res4')
             .max_pool(2, 2, 2, 2, name='HG6_pool3'))


# res6
        (self.feed('HG6_pool3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res6_batch1')
             .relu(name='HG6_res6_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG6_res6_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res6_batch2')
             .relu(name='HG6_res6_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG6_res6_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res6_batch3')
             .relu(name='HG6_res6_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res6_conv3'))

        (self.feed('HG6_pool3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res6_skip'))

        (self.feed('HG6_res6_conv3',
                   'HG6_res6_skip')
         .add(name='HG6_res6'))

# res7
        (self.feed('HG6_res6')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res7_batch1')
             .relu(name='HG6_res7_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG6_res7_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res7_batch2')
             .relu(name='HG6_res7_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG6_res7_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res7_batch3')
             .relu(name='HG6_res7_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res7_conv3'))

        (self.feed('HG6_res6')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res7_skip'))

        (self.feed('HG6_res7_conv3',
                   'HG6_res7_skip')
         .add(name='HG6_res7'))


# pool4
        (self.feed('HG6_res6')
             .max_pool(2, 2, 2, 2, name='HG6_pool4'))

# res8
        (self.feed('HG6_pool4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res8_batch1')
             .relu(name='HG6_res8_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG6_res8_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res8_batch2')
             .relu(name='HG6_res8_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG6_res8_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res8_batch3')
             .relu(name='HG6_res8_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res8_conv3'))

        (self.feed('HG6_pool4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res8_skip'))

        (self.feed('HG6_res8_conv3',
                   'HG6_res8_skip')
         .add(name='HG6_res8'))

# res9
        (self.feed('HG6_res8')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res9_batch1')
             .relu(name='HG6_res9_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG6_res9_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res9_batch2')
             .relu(name='HG6_res9_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG6_res9_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res9_batch3')
             .relu(name='HG6_res9_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res9_conv3'))

        (self.feed('HG6_res8')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res9_skip'))

        (self.feed('HG6_res9_conv3',
                   'HG6_res9_skip')
         .add(name='HG6_res9'))

# res10
        (self.feed('HG6_res9')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res10_batch1')
             .relu(name='HG6_res10_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG6_res10_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res10_batch2')
             .relu(name='HG6_res10_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG6_res10_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res10_batch3')
             .relu(name='HG6_res10_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res10_conv3'))

        (self.feed('HG6_res9')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res10_skip'))

        (self.feed('HG6_res10_conv3',
                   'HG6_res10_skip')
         .add(name='HG6_res10'))


# upsample1
        (self.feed('HG6_res10')
             .upsample(8, 8, name='HG6_upSample1'))

# upsample1 + up1(Hg_res7)
        (self.feed('HG6_upSample1',
                   'HG6_res7')
         .add(name='HG6_add1'))


# res11
        (self.feed('HG6_add1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res11_batch1')
             .relu(name='HG6_res11_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG6_res11_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res11_batch2')
             .relu(name='HG6_res11_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG6_res11_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res11_batch3')
             .relu(name='HG6_res11_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res11_conv3'))

        (self.feed('HG6_add1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res11_skip'))

        (self.feed('HG6_res11_conv3',
                   'HG6_res11_skip')
         .add(name='HG6_res11'))


# upsample2
        (self.feed('HG6_res11')
             .upsample(16, 16, name='HG6_upSample2'))

# upsample2 + up1(Hg_res5)
        (self.feed('HG6_upSample2',
                   'HG6_res5')
         .add(name='HG6_add2'))


# res12
        (self.feed('HG6_add2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res12_batch1')
             .relu(name='HG6_res12_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG6_res12_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res12_batch2')
             .relu(name='HG6_res12_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG6_res12_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res12_batch3')
             .relu(name='HG6_res12_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res12_conv3'))

        (self.feed('HG6_add2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res12_skip'))

        (self.feed('HG6_res12_conv3',
                   'HG6_res12_skip')
         .add(name='HG6_res12'))


# upsample3
        (self.feed('HG6_res12')
             .upsample(32, 32, name='HG6_upSample3'))

# upsample3 + HG6_resPool3
        (self.feed('HG6_upSample3',
                   'HG6_resPool3')
         .add(name='HG6_add3'))


# res13
        (self.feed('HG6_add3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res13_batch1')
             .relu(name='HG6_res13_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG6_res13_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res13_batch2')
             .relu(name='HG6_res13_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG6_res13_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res13_batch3')
             .relu(name='HG6_res13_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res13_conv3'))

        (self.feed('HG6_add3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res13_skip'))

        (self.feed('HG6_res13_conv3',
                   'HG6_res13_skip')
         .add(name='HG6_res13'))


# upsample4
        (self.feed('HG6_res13')
             .upsample(64, 64, name='HG6_upSample4'))

# upsample4 + HG6_resPool2
        (self.feed('HG6_upSample4',
                   'HG6_resPool2')
         .add(name='HG6_add4'))

###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  Hourglass  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

################################# HG6 postprocess #################
# Linear layer to produce first set of predictions
# ll1
        (self.feed('HG6_add4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_linearfunc1_conv1', padding='SAME')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_linearfunc1_batch1')
             .relu(name='HG6_linearfunc1_relu'))
# ll2
        (self.feed('HG6_linearfunc1_relu')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_linearfunc2_conv1', padding='SAME')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_linearfunc2_batch1')
             .relu(name='HG6_linearfunc2_relu'))

# att itersize=3

# att U input: ll2
        (self.feed('HG6_linearfunc2_relu')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG6_att_U', padding='SAME'))
# att i=1 conv  C(1)
        with tf.variable_scope('HG6_att_share_param', reuse=False):
            (self.feed('HG6_att_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_att_spConv1', padding='SAME'))
# att Qtemp
        (self.feed('HG6_att_spConv1',
                   'HG6_att_U')
         .add(name='HG6_att_Qtemp1_add')
         .sigmoid(name='HG6_att_Qtemp1'))

# att i=2 conv  C(2)
        with tf.variable_scope('HG6_att_share_param', reuse=True):
            (self.feed('HG6_att_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_att_spConv1', padding='SAME'))
# att Qtemp2
        (self.feed('HG6_att_spConv1',
                   'HG6_att_U')
         .add(name='HG6_att_Qtemp2_add')
         .sigmoid(name='HG6_att_Qtemp2'))

# att i=3 conv  C(3)
        with tf.variable_scope('HG6_att_share_param', reuse=True):
            (self.feed('HG6_att_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_att_spConv1', padding='SAME'))
# att Qtemp
        (self.feed('HG6_att_spConv1',
                   'HG6_att_U')
         .add(name='HG6_att_Qtemp3_add')
         .sigmoid(name='HG6_att_Qtemp3'))
# att pfeat
        (self.feed('HG6_att_Qtemp3')
         .replicate(256, 3, name='HG6_att_pfeat_replicate'))

        (self.feed('HG6_linearfunc2_relu',
                   'HG6_att_pfeat_replicate')
         .multiply2(name='HG6_att_pfeat_multiply'))

# tmpOut 
        # (self.feed('HG6_att_pfeat_multiply')
        #      .conv(1, 1, n_classes, 1, 1, biased=True, relu=False, name='HG6_att_Heatmap', padding='SAME'))










# # tmpOut att U input: HG6_att_pfeat_multiply
#         (self.feed('HG6_att_pfeat_multiply')
#              .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut_U', padding='SAME'))
# # tmpOut att i=1 conv  C(1)
#         with tf.variable_scope('HG6_tmpOut_share_param', reuse=False):
#             (self.feed('HG6_tmpOut_U')
#                  .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut_spConv1', padding='SAME'))
# # tmpOut att Qtemp
#         (self.feed('HG6_tmpOut_spConv1',
#                    'HG6_tmpOut_U')
#          .add(name='HG6_tmpOut_Qtemp1_add')
#          .sigmoid(name='HG6_tmpOut_Qtemp1'))

# # tmpOut att i=2 conv  C(2)
#         with tf.variable_scope('HG6_tmpOut_share_param', reuse=True):
#             (self.feed('HG6_tmpOut_Qtemp1')
#                  .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut_spConv1', padding='SAME'))
# # tmpOut att Qtemp2
#         (self.feed('HG6_tmpOut_spConv1',
#                    'HG6_tmpOut_U')
#          .add(name='HG6_tmpOut_Qtemp2_add')
#          .sigmoid(name='HG6_tmpOut_Qtemp2'))

# # tmpOut att i=3 conv  C(3)
#         with tf.variable_scope('HG6_tmpOut_share_param', reuse=True):
#             (self.feed('HG6_tmpOut_Qtemp2')
#                  .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut_spConv1', padding='SAME'))
# # tmpOut att Qtemp
#         (self.feed('HG6_tmpOut_spConv1',
#                    'HG6_tmpOut_U')
#          .add(name='HG6_tmpOut_Qtemp3_add')
#          .sigmoid(name='HG6_tmpOut_Qtemp3'))
# # tmpOut att pfeat
#         (self.feed('HG6_tmpOut_Qtemp3')
#          .replicate(256, 3, name='HG6_tmpOut_pfeat_replicate'))

#         (self.feed('HG6_att_pfeat_multiply',
#                    'HG6_tmpOut_pfeat_replicate')
#          .multiply2(name='HG6_tmpOut_pfeat_multiply'))

#         (self.feed('HG6_tmpOut_pfeat_multiply')
#              .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut_s', padding='SAME'))

#         (self.feed('HG6_tmpOut_s')
#          .replicate(16, 3, name='HG6_Heatmap'))        









# attention map for part1
# tmpOut1 att U input: HG6_att_pfeat_multiply
        (self.feed('HG6_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut1_U', padding='SAME'))
# tmpOut1 att i=1 conv  C(1)
        with tf.variable_scope('HG6_tmpOut1_share_param', reuse=False):
            (self.feed('HG6_tmpOut1_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut1_spConv1', padding='SAME'))
# tmpOut1 att Qtemp
        (self.feed('HG6_tmpOut1_spConv1',
                   'HG6_tmpOut1_U')
         .add(name='HG6_tmpOut1_Qtemp1_add')
         .sigmoid(name='HG6_tmpOut1_Qtemp1'))

# tmpOut1 att i=2 conv  C(2)
        with tf.variable_scope('HG6_tmpOut1_share_param', reuse=True):
            (self.feed('HG6_tmpOut1_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut1_spConv1', padding='SAME'))
# tmpOut1 att Qtemp2
        (self.feed('HG6_tmpOut1_spConv1',
                   'HG6_tmpOut1_U')
         .add(name='HG6_tmpOut1_Qtemp2_add')
         .sigmoid(name='HG6_tmpOut1_Qtemp2'))

# tmpOut1 att i=3 conv  C(3)
        with tf.variable_scope('HG6_tmpOut1_share_param', reuse=True):
            (self.feed('HG6_tmpOut1_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut1_spConv1', padding='SAME'))
# tmpOut1 att Qtemp
        (self.feed('HG6_tmpOut1_spConv1',
                   'HG6_tmpOut1_U')
         .add(name='HG6_tmpOut1_Qtemp3_add')
         .sigmoid(name='HG6_tmpOut1_Qtemp3'))
# tmpOut1 att pfeat
        (self.feed('HG6_tmpOut1_Qtemp3')
         .replicate(256, 3, name='HG6_tmpOut1_pfeat_replicate'))

        (self.feed('HG6_att_pfeat_multiply',
                   'HG6_tmpOut1_pfeat_replicate')
         .multiply2(name='HG6_tmpOut1_pfeat_multiply'))

        (self.feed('HG6_tmpOut1_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut1_s', padding='SAME'))




# attention map for part2
# tmpOut2 att U input: HG6_att_pfeat_multiply
        (self.feed('HG6_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut2_U', padding='SAME'))
# tmpOut2 att i=1 conv  C(1)
        with tf.variable_scope('HG6_tmpOut2_share_param', reuse=False):
            (self.feed('HG6_tmpOut2_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut2_spConv1', padding='SAME'))
# tmpOut2 att Qtemp
        (self.feed('HG6_tmpOut2_spConv1',
                   'HG6_tmpOut2_U')
         .add(name='HG6_tmpOut2_Qtemp1_add')
         .sigmoid(name='HG6_tmpOut2_Qtemp1'))

# tmpOut2 att i=2 conv  C(2)
        with tf.variable_scope('HG6_tmpOut2_share_param', reuse=True):
            (self.feed('HG6_tmpOut2_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut2_spConv1', padding='SAME'))
# tmpOut2 att Qtemp2
        (self.feed('HG6_tmpOut2_spConv1',
                   'HG6_tmpOut2_U')
         .add(name='HG6_tmpOut2_Qtemp2_add')
         .sigmoid(name='HG6_tmpOut2_Qtemp2'))

# tmpOut2 att i=3 conv  C(3)
        with tf.variable_scope('HG6_tmpOut2_share_param', reuse=True):
            (self.feed('HG6_tmpOut2_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut2_spConv1', padding='SAME'))
# tmpOut2 att Qtemp
        (self.feed('HG6_tmpOut2_spConv1',
                   'HG6_tmpOut2_U')
         .add(name='HG6_tmpOut2_Qtemp3_add')
         .sigmoid(name='HG6_tmpOut2_Qtemp3'))
# tmpOut2 att pfeat
        (self.feed('HG6_tmpOut2_Qtemp3')
         .replicate(256, 3, name='HG6_tmpOut2_pfeat_replicate'))

        (self.feed('HG6_att_pfeat_multiply',
                   'HG6_tmpOut2_pfeat_replicate')
         .multiply2(name='HG6_tmpOut2_pfeat_multiply'))

        (self.feed('HG6_tmpOut2_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut2_s', padding='SAME'))




# attention map for part3
# tmpOut3 att U input: HG6_att_pfeat_multiply
        (self.feed('HG6_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut3_U', padding='SAME'))
# tmpOut3 att i=1 conv  C(1)
        with tf.variable_scope('HG6_tmpOut3_share_param', reuse=False):
            (self.feed('HG6_tmpOut3_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut3_spConv1', padding='SAME'))
# tmpOut3 att Qtemp
        (self.feed('HG6_tmpOut3_spConv1',
                   'HG6_tmpOut3_U')
         .add(name='HG6_tmpOut3_Qtemp1_add')
         .sigmoid(name='HG6_tmpOut3_Qtemp1'))

# tmpOut3 att i=2 conv  C(2)
        with tf.variable_scope('HG6_tmpOut3_share_param', reuse=True):
            (self.feed('HG6_tmpOut3_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut3_spConv1', padding='SAME'))
# tmpOut3 att Qtemp2
        (self.feed('HG6_tmpOut3_spConv1',
                   'HG6_tmpOut3_U')
         .add(name='HG6_tmpOut3_Qtemp2_add')
         .sigmoid(name='HG6_tmpOut3_Qtemp2'))

# tmpOut3 att i=3 conv  C(3)
        with tf.variable_scope('HG6_tmpOut3_share_param', reuse=True):
            (self.feed('HG6_tmpOut3_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut3_spConv1', padding='SAME'))
# tmpOut3 att Qtemp
        (self.feed('HG6_tmpOut3_spConv1',
                   'HG6_tmpOut3_U')
         .add(name='HG6_tmpOut3_Qtemp3_add')
         .sigmoid(name='HG6_tmpOut3_Qtemp3'))
# tmpOut3 att pfeat
        (self.feed('HG6_tmpOut3_Qtemp3')
         .replicate(256, 3, name='HG6_tmpOut3_pfeat_replicate'))

        (self.feed('HG6_att_pfeat_multiply',
                   'HG6_tmpOut3_pfeat_replicate')
         .multiply2(name='HG6_tmpOut3_pfeat_multiply'))

        (self.feed('HG6_tmpOut3_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut3_s', padding='SAME'))






# attention map for part4
# tmpOut4 att U input: HG6_att_pfeat_multiply
        (self.feed('HG6_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut4_U', padding='SAME'))
# tmpOut4 att i=1 conv  C(1)
        with tf.variable_scope('HG6_tmpOut4_share_param', reuse=False):
            (self.feed('HG6_tmpOut4_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut4_spConv1', padding='SAME'))
# tmpOut4 att Qtemp
        (self.feed('HG6_tmpOut4_spConv1',
                   'HG6_tmpOut4_U')
         .add(name='HG6_tmpOut4_Qtemp1_add')
         .sigmoid(name='HG6_tmpOut4_Qtemp1'))

# tmpOut4 att i=2 conv  C(2)
        with tf.variable_scope('HG6_tmpOut4_share_param', reuse=True):
            (self.feed('HG6_tmpOut4_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut4_spConv1', padding='SAME'))
# tmpOut4 att Qtemp2
        (self.feed('HG6_tmpOut4_spConv1',
                   'HG6_tmpOut4_U')
         .add(name='HG6_tmpOut4_Qtemp2_add')
         .sigmoid(name='HG6_tmpOut4_Qtemp2'))

# tmpOut4 att i=3 conv  C(3)
        with tf.variable_scope('HG6_tmpOut4_share_param', reuse=True):
            (self.feed('HG6_tmpOut4_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut4_spConv1', padding='SAME'))
# tmpOut4 att Qtemp
        (self.feed('HG6_tmpOut4_spConv1',
                   'HG6_tmpOut4_U')
         .add(name='HG6_tmpOut4_Qtemp3_add')
         .sigmoid(name='HG6_tmpOut4_Qtemp3'))
# tmpOut4 att pfeat
        (self.feed('HG6_tmpOut4_Qtemp3')
         .replicate(256, 3, name='HG6_tmpOut4_pfeat_replicate'))

        (self.feed('HG6_att_pfeat_multiply',
                   'HG6_tmpOut4_pfeat_replicate')
         .multiply2(name='HG6_tmpOut4_pfeat_multiply'))

        (self.feed('HG6_tmpOut4_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut4_s', padding='SAME'))









# attention map for part5
# tmpOut5 att U input: HG6_att_pfeat_multiply
        (self.feed('HG6_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut5_U', padding='SAME'))
# tmpOut5 att i=1 conv  C(1)
        with tf.variable_scope('HG6_tmpOut5_share_param', reuse=False):
            (self.feed('HG6_tmpOut5_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut5_spConv1', padding='SAME'))
# tmpOut5 att Qtemp
        (self.feed('HG6_tmpOut5_spConv1',
                   'HG6_tmpOut5_U')
         .add(name='HG6_tmpOut5_Qtemp1_add')
         .sigmoid(name='HG6_tmpOut5_Qtemp1'))

# tmpOut5 att i=2 conv  C(2)
        with tf.variable_scope('HG6_tmpOut5_share_param', reuse=True):
            (self.feed('HG6_tmpOut5_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut5_spConv1', padding='SAME'))
# tmpOut5 att Qtemp2
        (self.feed('HG6_tmpOut5_spConv1',
                   'HG6_tmpOut5_U')
         .add(name='HG6_tmpOut5_Qtemp2_add')
         .sigmoid(name='HG6_tmpOut5_Qtemp2'))

# tmpOut5 att i=3 conv  C(3)
        with tf.variable_scope('HG6_tmpOut5_share_param', reuse=True):
            (self.feed('HG6_tmpOut5_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut5_spConv1', padding='SAME'))
# tmpOut5 att Qtemp
        (self.feed('HG6_tmpOut5_spConv1',
                   'HG6_tmpOut5_U')
         .add(name='HG6_tmpOut5_Qtemp3_add')
         .sigmoid(name='HG6_tmpOut5_Qtemp3'))
# tmpOut5 att pfeat
        (self.feed('HG6_tmpOut5_Qtemp3')
         .replicate(256, 3, name='HG6_tmpOut5_pfeat_replicate'))

        (self.feed('HG6_att_pfeat_multiply',
                   'HG6_tmpOut5_pfeat_replicate')
         .multiply2(name='HG6_tmpOut5_pfeat_multiply'))

        (self.feed('HG6_tmpOut5_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut5_s', padding='SAME'))










# attention map for part6
# tmpOut6 att U input: HG6_att_pfeat_multiply
        (self.feed('HG6_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut6_U', padding='SAME'))
# tmpOut6 att i=1 conv  C(1)
        with tf.variable_scope('HG6_tmpOut6_share_param', reuse=False):
            (self.feed('HG6_tmpOut6_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut6_spConv1', padding='SAME'))
# tmpOut6 att Qtemp
        (self.feed('HG6_tmpOut6_spConv1',
                   'HG6_tmpOut6_U')
         .add(name='HG6_tmpOut6_Qtemp1_add')
         .sigmoid(name='HG6_tmpOut6_Qtemp1'))

# tmpOut6 att i=2 conv  C(2)
        with tf.variable_scope('HG6_tmpOut6_share_param', reuse=True):
            (self.feed('HG6_tmpOut6_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut6_spConv1', padding='SAME'))
# tmpOut6 att Qtemp2
        (self.feed('HG6_tmpOut6_spConv1',
                   'HG6_tmpOut6_U')
         .add(name='HG6_tmpOut6_Qtemp2_add')
         .sigmoid(name='HG6_tmpOut6_Qtemp2'))

# tmpOut6 att i=3 conv  C(3)
        with tf.variable_scope('HG6_tmpOut6_share_param', reuse=True):
            (self.feed('HG6_tmpOut6_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut6_spConv1', padding='SAME'))
# tmpOut6 att Qtemp
        (self.feed('HG6_tmpOut6_spConv1',
                   'HG6_tmpOut6_U')
         .add(name='HG6_tmpOut6_Qtemp3_add')
         .sigmoid(name='HG6_tmpOut6_Qtemp3'))
# tmpOut6 att pfeat
        (self.feed('HG6_tmpOut6_Qtemp3')
         .replicate(256, 3, name='HG6_tmpOut6_pfeat_replicate'))

        (self.feed('HG6_att_pfeat_multiply',
                   'HG6_tmpOut6_pfeat_replicate')
         .multiply2(name='HG6_tmpOut6_pfeat_multiply'))

        (self.feed('HG6_tmpOut6_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut6_s', padding='SAME'))





# attention map for part7
# tmpOut7 att U input: HG6_att_pfeat_multiply
        (self.feed('HG6_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut7_U', padding='SAME'))
# tmpOut7 att i=1 conv  C(1)
        with tf.variable_scope('HG6_tmpOut7_share_param', reuse=False):
            (self.feed('HG6_tmpOut7_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut7_spConv1', padding='SAME'))
# tmpOut7 att Qtemp
        (self.feed('HG6_tmpOut7_spConv1',
                   'HG6_tmpOut7_U')
         .add(name='HG6_tmpOut7_Qtemp1_add')
         .sigmoid(name='HG6_tmpOut7_Qtemp1'))

# tmpOut7 att i=2 conv  C(2)
        with tf.variable_scope('HG6_tmpOut7_share_param', reuse=True):
            (self.feed('HG6_tmpOut7_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut7_spConv1', padding='SAME'))
# tmpOut7 att Qtemp2
        (self.feed('HG6_tmpOut7_spConv1',
                   'HG6_tmpOut7_U')
         .add(name='HG6_tmpOut7_Qtemp2_add')
         .sigmoid(name='HG6_tmpOut7_Qtemp2'))

# tmpOut7 att i=3 conv  C(3)
        with tf.variable_scope('HG6_tmpOut7_share_param', reuse=True):
            (self.feed('HG6_tmpOut7_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut7_spConv1', padding='SAME'))
# tmpOut7 att Qtemp
        (self.feed('HG6_tmpOut7_spConv1',
                   'HG6_tmpOut7_U')
         .add(name='HG6_tmpOut7_Qtemp3_add')
         .sigmoid(name='HG6_tmpOut7_Qtemp3'))
# tmpOut7 att pfeat
        (self.feed('HG6_tmpOut7_Qtemp3')
         .replicate(256, 3, name='HG6_tmpOut7_pfeat_replicate'))

        (self.feed('HG6_att_pfeat_multiply',
                   'HG6_tmpOut7_pfeat_replicate')
         .multiply2(name='HG6_tmpOut7_pfeat_multiply'))

        (self.feed('HG6_tmpOut7_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut7_s', padding='SAME'))








# attention map for part8
# tmpOut8 att U input: HG6_att_pfeat_multiply
        (self.feed('HG6_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut8_U', padding='SAME'))
# tmpOut8 att i=1 conv  C(1)
        with tf.variable_scope('HG6_tmpOut8_share_param', reuse=False):
            (self.feed('HG6_tmpOut8_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut8_spConv1', padding='SAME'))
# tmpOut8 att Qtemp
        (self.feed('HG6_tmpOut8_spConv1',
                   'HG6_tmpOut8_U')
         .add(name='HG6_tmpOut8_Qtemp1_add')
         .sigmoid(name='HG6_tmpOut8_Qtemp1'))

# tmpOut8 att i=2 conv  C(2)
        with tf.variable_scope('HG6_tmpOut8_share_param', reuse=True):
            (self.feed('HG6_tmpOut8_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut8_spConv1', padding='SAME'))
# tmpOut8 att Qtemp2
        (self.feed('HG6_tmpOut8_spConv1',
                   'HG6_tmpOut8_U')
         .add(name='HG6_tmpOut8_Qtemp2_add')
         .sigmoid(name='HG6_tmpOut8_Qtemp2'))

# tmpOut8 att i=3 conv  C(3)
        with tf.variable_scope('HG6_tmpOut8_share_param', reuse=True):
            (self.feed('HG6_tmpOut8_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut8_spConv1', padding='SAME'))
# tmpOut8 att Qtemp
        (self.feed('HG6_tmpOut8_spConv1',
                   'HG6_tmpOut8_U')
         .add(name='HG6_tmpOut8_Qtemp3_add')
         .sigmoid(name='HG6_tmpOut8_Qtemp3'))
# tmpOut8 att pfeat
        (self.feed('HG6_tmpOut8_Qtemp3')
         .replicate(256, 3, name='HG6_tmpOut8_pfeat_replicate'))

        (self.feed('HG6_att_pfeat_multiply',
                   'HG6_tmpOut8_pfeat_replicate')
         .multiply2(name='HG6_tmpOut8_pfeat_multiply'))

        (self.feed('HG6_tmpOut8_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut8_s', padding='SAME'))











# attention map for part9
# tmpOut9 att U input: HG6_att_pfeat_multiply
        (self.feed('HG6_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut9_U', padding='SAME'))
# tmpOut9 att i=1 conv  C(1)
        with tf.variable_scope('HG6_tmpOut9_share_param', reuse=False):
            (self.feed('HG6_tmpOut9_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut9_spConv1', padding='SAME'))
# tmpOut9 att Qtemp
        (self.feed('HG6_tmpOut9_spConv1',
                   'HG6_tmpOut9_U')
         .add(name='HG6_tmpOut9_Qtemp1_add')
         .sigmoid(name='HG6_tmpOut9_Qtemp1'))

# tmpOut9 att i=2 conv  C(2)
        with tf.variable_scope('HG6_tmpOut9_share_param', reuse=True):
            (self.feed('HG6_tmpOut9_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut9_spConv1', padding='SAME'))
# tmpOut9 att Qtemp2
        (self.feed('HG6_tmpOut9_spConv1',
                   'HG6_tmpOut9_U')
         .add(name='HG6_tmpOut9_Qtemp2_add')
         .sigmoid(name='HG6_tmpOut9_Qtemp2'))

# tmpOut9 att i=3 conv  C(3)
        with tf.variable_scope('HG6_tmpOut9_share_param', reuse=True):
            (self.feed('HG6_tmpOut9_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut9_spConv1', padding='SAME'))
# tmpOut9 att Qtemp
        (self.feed('HG6_tmpOut9_spConv1',
                   'HG6_tmpOut9_U')
         .add(name='HG6_tmpOut9_Qtemp3_add')
         .sigmoid(name='HG6_tmpOut9_Qtemp3'))
# tmpOut9 att pfeat
        (self.feed('HG6_tmpOut9_Qtemp3')
         .replicate(256, 3, name='HG6_tmpOut9_pfeat_replicate'))

        (self.feed('HG6_att_pfeat_multiply',
                   'HG6_tmpOut9_pfeat_replicate')
         .multiply2(name='HG6_tmpOut9_pfeat_multiply'))

        (self.feed('HG6_tmpOut9_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut9_s', padding='SAME'))












# attention map for part10
# tmpOut10 att U input: HG6_att_pfeat_multiply
        (self.feed('HG6_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut10_U', padding='SAME'))
# tmpOut10 att i=1 conv  C(1)
        with tf.variable_scope('HG6_tmpOut10_share_param', reuse=False):
            (self.feed('HG6_tmpOut10_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut10_spConv1', padding='SAME'))
# tmpOut10 att Qtemp
        (self.feed('HG6_tmpOut10_spConv1',
                   'HG6_tmpOut10_U')
         .add(name='HG6_tmpOut10_Qtemp1_add')
         .sigmoid(name='HG6_tmpOut10_Qtemp1'))

# tmpOut10 att i=2 conv  C(2)
        with tf.variable_scope('HG6_tmpOut10_share_param', reuse=True):
            (self.feed('HG6_tmpOut10_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut10_spConv1', padding='SAME'))
# tmpOut10 att Qtemp2
        (self.feed('HG6_tmpOut10_spConv1',
                   'HG6_tmpOut10_U')
         .add(name='HG6_tmpOut10_Qtemp2_add')
         .sigmoid(name='HG6_tmpOut10_Qtemp2'))

# tmpOut10 att i=3 conv  C(3)
        with tf.variable_scope('HG6_tmpOut10_share_param', reuse=True):
            (self.feed('HG6_tmpOut10_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut10_spConv1', padding='SAME'))
# tmpOut10 att Qtemp
        (self.feed('HG6_tmpOut10_spConv1',
                   'HG6_tmpOut10_U')
         .add(name='HG6_tmpOut10_Qtemp3_add')
         .sigmoid(name='HG6_tmpOut10_Qtemp3'))
# tmpOut10 att pfeat
        (self.feed('HG6_tmpOut10_Qtemp3')
         .replicate(256, 3, name='HG6_tmpOut10_pfeat_replicate'))

        (self.feed('HG6_att_pfeat_multiply',
                   'HG6_tmpOut10_pfeat_replicate')
         .multiply2(name='HG6_tmpOut10_pfeat_multiply'))

        (self.feed('HG6_tmpOut10_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut10_s', padding='SAME'))










# attention map for part11
# tmpOut11 att U input: HG6_att_pfeat_multiply
        (self.feed('HG6_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut11_U', padding='SAME'))
# tmpOut11 att i=1 conv  C(1)
        with tf.variable_scope('HG6_tmpOut11_share_param', reuse=False):
            (self.feed('HG6_tmpOut11_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut11_spConv1', padding='SAME'))
# tmpOut11 att Qtemp
        (self.feed('HG6_tmpOut11_spConv1',
                   'HG6_tmpOut11_U')
         .add(name='HG6_tmpOut11_Qtemp1_add')
         .sigmoid(name='HG6_tmpOut11_Qtemp1'))

# tmpOut11 att i=2 conv  C(2)
        with tf.variable_scope('HG6_tmpOut11_share_param', reuse=True):
            (self.feed('HG6_tmpOut11_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut11_spConv1', padding='SAME'))
# tmpOut11 att Qtemp2
        (self.feed('HG6_tmpOut11_spConv1',
                   'HG6_tmpOut11_U')
         .add(name='HG6_tmpOut11_Qtemp2_add')
         .sigmoid(name='HG6_tmpOut11_Qtemp2'))

# tmpOut11 att i=3 conv  C(3)
        with tf.variable_scope('HG6_tmpOut11_share_param', reuse=True):
            (self.feed('HG6_tmpOut11_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut11_spConv1', padding='SAME'))
# tmpOut11 att Qtemp
        (self.feed('HG6_tmpOut11_spConv1',
                   'HG6_tmpOut11_U')
         .add(name='HG6_tmpOut11_Qtemp3_add')
         .sigmoid(name='HG6_tmpOut11_Qtemp3'))
# tmpOut11 att pfeat
        (self.feed('HG6_tmpOut11_Qtemp3')
         .replicate(256, 3, name='HG6_tmpOut11_pfeat_replicate'))

        (self.feed('HG6_att_pfeat_multiply',
                   'HG6_tmpOut11_pfeat_replicate')
         .multiply2(name='HG6_tmpOut11_pfeat_multiply'))

        (self.feed('HG6_tmpOut11_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut11_s', padding='SAME'))











# attention map for part12
# tmpOut12 att U input: HG6_att_pfeat_multiply
        (self.feed('HG6_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut12_U', padding='SAME'))
# tmpOut12 att i=1 conv  C(1)
        with tf.variable_scope('HG6_tmpOut12_share_param', reuse=False):
            (self.feed('HG6_tmpOut12_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut12_spConv1', padding='SAME'))
# tmpOut12 att Qtemp
        (self.feed('HG6_tmpOut12_spConv1',
                   'HG6_tmpOut12_U')
         .add(name='HG6_tmpOut12_Qtemp1_add')
         .sigmoid(name='HG6_tmpOut12_Qtemp1'))

# tmpOut12 att i=2 conv  C(2)
        with tf.variable_scope('HG6_tmpOut12_share_param', reuse=True):
            (self.feed('HG6_tmpOut12_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut12_spConv1', padding='SAME'))
# tmpOut12 att Qtemp2
        (self.feed('HG6_tmpOut12_spConv1',
                   'HG6_tmpOut12_U')
         .add(name='HG6_tmpOut12_Qtemp2_add')
         .sigmoid(name='HG6_tmpOut12_Qtemp2'))

# tmpOut12 att i=3 conv  C(3)
        with tf.variable_scope('HG6_tmpOut12_share_param', reuse=True):
            (self.feed('HG6_tmpOut12_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut12_spConv1', padding='SAME'))
# tmpOut12 att Qtemp
        (self.feed('HG6_tmpOut12_spConv1',
                   'HG6_tmpOut12_U')
         .add(name='HG6_tmpOut12_Qtemp3_add')
         .sigmoid(name='HG6_tmpOut12_Qtemp3'))
# tmpOut12 att pfeat
        (self.feed('HG6_tmpOut12_Qtemp3')
         .replicate(256, 3, name='HG6_tmpOut12_pfeat_replicate'))

        (self.feed('HG6_att_pfeat_multiply',
                   'HG6_tmpOut12_pfeat_replicate')
         .multiply2(name='HG6_tmpOut12_pfeat_multiply'))

        (self.feed('HG6_tmpOut12_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut12_s', padding='SAME'))










# attention map for part13
# tmpOut13 att U input: HG6_att_pfeat_multiply
        (self.feed('HG6_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut13_U', padding='SAME'))
# tmpOut13 att i=1 conv  C(1)
        with tf.variable_scope('HG6_tmpOut13_share_param', reuse=False):
            (self.feed('HG6_tmpOut13_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut13_spConv1', padding='SAME'))
# tmpOut13 att Qtemp
        (self.feed('HG6_tmpOut13_spConv1',
                   'HG6_tmpOut13_U')
         .add(name='HG6_tmpOut13_Qtemp1_add')
         .sigmoid(name='HG6_tmpOut13_Qtemp1'))

# tmpOut13 att i=2 conv  C(2)
        with tf.variable_scope('HG6_tmpOut13_share_param', reuse=True):
            (self.feed('HG6_tmpOut13_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut13_spConv1', padding='SAME'))
# tmpOut13 att Qtemp2
        (self.feed('HG6_tmpOut13_spConv1',
                   'HG6_tmpOut13_U')
         .add(name='HG6_tmpOut13_Qtemp2_add')
         .sigmoid(name='HG6_tmpOut13_Qtemp2'))

# tmpOut13 att i=3 conv  C(3)
        with tf.variable_scope('HG6_tmpOut13_share_param', reuse=True):
            (self.feed('HG6_tmpOut13_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut13_spConv1', padding='SAME'))
# tmpOut13 att Qtemp
        (self.feed('HG6_tmpOut13_spConv1',
                   'HG6_tmpOut13_U')
         .add(name='HG6_tmpOut13_Qtemp3_add')
         .sigmoid(name='HG6_tmpOut13_Qtemp3'))
# tmpOut13 att pfeat
        (self.feed('HG6_tmpOut13_Qtemp3')
         .replicate(256, 3, name='HG6_tmpOut13_pfeat_replicate'))

        (self.feed('HG6_att_pfeat_multiply',
                   'HG6_tmpOut13_pfeat_replicate')
         .multiply2(name='HG6_tmpOut13_pfeat_multiply'))

        (self.feed('HG6_tmpOut13_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut13_s', padding='SAME'))










# attention map for part14
# tmpOut14 att U input: HG6_att_pfeat_multiply
        (self.feed('HG6_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut14_U', padding='SAME'))
# tmpOut14 att i=1 conv  C(1)
        with tf.variable_scope('HG6_tmpOut14_share_param', reuse=False):
            (self.feed('HG6_tmpOut14_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut14_spConv1', padding='SAME'))
# tmpOut14 att Qtemp
        (self.feed('HG6_tmpOut14_spConv1',
                   'HG6_tmpOut14_U')
         .add(name='HG6_tmpOut14_Qtemp1_add')
         .sigmoid(name='HG6_tmpOut14_Qtemp1'))

# tmpOut14 att i=2 conv  C(2)
        with tf.variable_scope('HG6_tmpOut14_share_param', reuse=True):
            (self.feed('HG6_tmpOut14_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut14_spConv1', padding='SAME'))
# tmpOut14 att Qtemp2
        (self.feed('HG6_tmpOut14_spConv1',
                   'HG6_tmpOut14_U')
         .add(name='HG6_tmpOut14_Qtemp2_add')
         .sigmoid(name='HG6_tmpOut14_Qtemp2'))

# tmpOut14 att i=3 conv  C(3)
        with tf.variable_scope('HG6_tmpOut14_share_param', reuse=True):
            (self.feed('HG6_tmpOut14_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut14_spConv1', padding='SAME'))
# tmpOut14 att Qtemp
        (self.feed('HG6_tmpOut14_spConv1',
                   'HG6_tmpOut14_U')
         .add(name='HG6_tmpOut14_Qtemp3_add')
         .sigmoid(name='HG6_tmpOut14_Qtemp3'))
# tmpOut14 att pfeat
        (self.feed('HG6_tmpOut14_Qtemp3')
         .replicate(256, 3, name='HG6_tmpOut14_pfeat_replicate'))

        (self.feed('HG6_att_pfeat_multiply',
                   'HG6_tmpOut14_pfeat_replicate')
         .multiply2(name='HG6_tmpOut14_pfeat_multiply'))

        (self.feed('HG6_tmpOut14_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut14_s', padding='SAME'))










# attention map for part15
# tmpOut15 att U input: HG6_att_pfeat_multiply
        (self.feed('HG6_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut15_U', padding='SAME'))
# tmpOut15 att i=1 conv  C(1)
        with tf.variable_scope('HG6_tmpOut15_share_param', reuse=False):
            (self.feed('HG6_tmpOut15_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut15_spConv1', padding='SAME'))
# tmpOut15 att Qtemp
        (self.feed('HG6_tmpOut15_spConv1',
                   'HG6_tmpOut15_U')
         .add(name='HG6_tmpOut15_Qtemp1_add')
         .sigmoid(name='HG6_tmpOut15_Qtemp1'))

# tmpOut15 att i=2 conv  C(2)
        with tf.variable_scope('HG6_tmpOut15_share_param', reuse=True):
            (self.feed('HG6_tmpOut15_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut15_spConv1', padding='SAME'))
# tmpOut15 att Qtemp2
        (self.feed('HG6_tmpOut15_spConv1',
                   'HG6_tmpOut15_U')
         .add(name='HG6_tmpOut15_Qtemp2_add')
         .sigmoid(name='HG6_tmpOut15_Qtemp2'))

# tmpOut15 att i=3 conv  C(3)
        with tf.variable_scope('HG6_tmpOut15_share_param', reuse=True):
            (self.feed('HG6_tmpOut15_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut15_spConv1', padding='SAME'))
# tmpOut15 att Qtemp
        (self.feed('HG6_tmpOut15_spConv1',
                   'HG6_tmpOut15_U')
         .add(name='HG6_tmpOut15_Qtemp3_add')
         .sigmoid(name='HG6_tmpOut15_Qtemp3'))
# tmpOut15 att pfeat
        (self.feed('HG6_tmpOut15_Qtemp3')
         .replicate(256, 3, name='HG6_tmpOut15_pfeat_replicate'))

        (self.feed('HG6_att_pfeat_multiply',
                   'HG6_tmpOut15_pfeat_replicate')
         .multiply2(name='HG6_tmpOut15_pfeat_multiply'))

        (self.feed('HG6_tmpOut15_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut15_s', padding='SAME'))










# attention map for part16
# tmpOut16 att U input: HG6_att_pfeat_multiply
        (self.feed('HG6_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut16_U', padding='SAME'))
# tmpOut16 att i=1 conv  C(1)
        with tf.variable_scope('HG6_tmpOut16_share_param', reuse=False):
            (self.feed('HG6_tmpOut16_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut16_spConv1', padding='SAME'))
# tmpOut16 att Qtemp
        (self.feed('HG6_tmpOut16_spConv1',
                   'HG6_tmpOut16_U')
         .add(name='HG6_tmpOut16_Qtemp1_add')
         .sigmoid(name='HG6_tmpOut16_Qtemp1'))

# tmpOut16 att i=2 conv  C(2)
        with tf.variable_scope('HG6_tmpOut16_share_param', reuse=True):
            (self.feed('HG6_tmpOut16_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut16_spConv1', padding='SAME'))
# tmpOut16 att Qtemp2
        (self.feed('HG6_tmpOut16_spConv1',
                   'HG6_tmpOut16_U')
         .add(name='HG6_tmpOut16_Qtemp2_add')
         .sigmoid(name='HG6_tmpOut16_Qtemp2'))

# tmpOut16 att i=3 conv  C(3)
        with tf.variable_scope('HG6_tmpOut16_share_param', reuse=True):
            (self.feed('HG6_tmpOut16_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut16_spConv1', padding='SAME'))
# tmpOut16 att Qtemp
        (self.feed('HG6_tmpOut16_spConv1',
                   'HG6_tmpOut16_U')
         .add(name='HG6_tmpOut16_Qtemp3_add')
         .sigmoid(name='HG6_tmpOut16_Qtemp3'))
# tmpOut16 att pfeat
        (self.feed('HG6_tmpOut16_Qtemp3')
         .replicate(256, 3, name='HG6_tmpOut16_pfeat_replicate'))

        (self.feed('HG6_att_pfeat_multiply',
                   'HG6_tmpOut16_pfeat_replicate')
         .multiply2(name='HG6_tmpOut16_pfeat_multiply'))

        (self.feed('HG6_tmpOut16_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG6_tmpOut16_s', padding='SAME'))


        (self.feed('HG6_tmpOut1_s',
                   'HG6_tmpOut2_s',
                   'HG6_tmpOut3_s',
                   'HG6_tmpOut4_s',
                   'HG6_tmpOut5_s',
                   'HG6_tmpOut6_s',
                   'HG6_tmpOut7_s',
                   'HG6_tmpOut8_s',
                   'HG6_tmpOut9_s',
                   'HG6_tmpOut10_s',
                   'HG6_tmpOut11_s',
                   'HG6_tmpOut12_s',
                   'HG6_tmpOut13_s',
                   'HG6_tmpOut14_s',
                   'HG6_tmpOut15_s',
                   'HG6_tmpOut16_s')
         .stack(axis = 3,  name='HG6_Heatmap'))        








###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ HG6 postprocess ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

############################# generate input for next HG ##########

# outmap
        (self.feed('HG6_Heatmap')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_outmap', padding='SAME'))
# ll3
        (self.feed('HG6_linearfunc1_relu')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_linearfunc3_conv1', padding='SAME')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_linearfunc3_batch1')
             .relu(name='HG6_linearfunc3_relu'))
# tmointer
        (self.feed('HG6_input',
                   'HG6_outmap',
                   'HG6_linearfunc3_relu')
         .add(name='HG7_input'))

###^^^^^^^^^^^^^^^^^^^^^^^^^^^ generate input for next HG ^^^^^^^^^^^^^^^^^^^^^^^^^^^^###


















































































#######################################  HG7  #########################################
                 ####################  Hourglass  #####################
# res1
        (self.feed('HG7_input')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res1_batch1')
             .relu(name='HG7_res1_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG7_res1_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res1_batch2')
             .relu(name='HG7_res1_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG7_res1_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res1_batch3')
             .relu(name='HG7_res1_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res1_conv3'))

        (self.feed('HG7_input')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res1_skip'))

        (self.feed('HG7_res1_conv3',
                   'HG7_res1_skip')
         .add(name='HG7_res1'))

# resPool1
        (self.feed('HG7_res1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_resPool1_batch1')
             .relu(name='HG7_resPool1_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG7_resPool1_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_resPool1_batch2')
             .relu(name='HG7_resPool1_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG7_resPool1_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_resPool1_batch3')
             .relu(name='HG7_resPool1_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_resPool1_conv3'))

        (self.feed('HG7_res1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_resPool1_skip'))

        (self.feed('HG7_res1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_resPool1_batch4')
             .relu(name='HG7_resPool1_relu4')
             .max_pool(2, 2, 2, 2, name='HG7_resPool1_pool')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='HG7_resPool1_conv4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_resPool1_batch5')
             .relu(name='HG7_resPool1_relu5')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='HG7_resPool1_conv5')
             .upsample(64, 64, name='HG7_resPool1_upSample'))


        (self.feed('HG7_resPool1_conv3',
                   'HG7_resPool1_skip',
                   'HG7_resPool1_upSample')
         .add(name='HG7_resPool1'))



# resPool2
        (self.feed('HG7_resPool1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_resPool2_batch1')
             .relu(name='HG7_resPool2_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG7_resPool2_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_resPool2_batch2')
             .relu(name='HG7_resPool2_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG7_resPool2_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_resPool2_batch3')
             .relu(name='HG7_resPool2_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_resPool2_conv3'))

        (self.feed('HG7_resPool1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_resPool2_skip'))

        (self.feed('HG7_resPool1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_resPool2_batch4')
             .relu(name='HG7_resPool2_relu4')
             .max_pool(2, 2, 2, 2, name='HG7_resPool2_pool')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='HG7_resPool2_conv4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_resPool2_batch5')
             .relu(name='HG7_resPool2_relu5')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='HG7_resPool2_conv5')
             .upsample(64, 64, name='HG7_resPool2_upSample'))


        (self.feed('HG7_resPool2_conv3',
                   'HG7_resPool2_skip',
                   'HG7_resPool2_upSample')
         .add(name='HG7_resPool2'))

# pool1
        (self.feed('HG7_input')
             .max_pool(2, 2, 2, 2, name='HG7_pool1'))


# res2
        (self.feed('HG7_pool1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res2_batch1')
             .relu(name='HG7_res2_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG7_res2_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res2_batch2')
             .relu(name='HG7_res2_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG7_res2_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res2_batch3')
             .relu(name='HG7_res2_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res2_conv3'))

        (self.feed('HG7_pool1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res2_skip'))

        (self.feed('HG7_res2_conv3',
                   'HG7_res2_skip')
         .add(name='HG7_res2'))

# res3
        (self.feed('HG7_res2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res3_batch1')
             .relu(name='HG7_res3_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG7_res3_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res3_batch2')
             .relu(name='HG7_res3_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG7_res3_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res3_batch3')
             .relu(name='HG7_res3_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res3_conv3'))

        (self.feed('HG7_res2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res3_skip'))

        (self.feed('HG7_res3_conv3',
                   'HG7_res3_skip')
         .add(name='HG7_res3'))

# resPool3
        (self.feed('HG7_res3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_resPool3_batch1')
             .relu(name='HG7_resPool3_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG7_resPool3_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_resPool3_batch2')
             .relu(name='HG7_resPool3_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG7_resPool3_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_resPool3_batch3')
             .relu(name='HG7_resPool3_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_resPool3_conv3'))

        (self.feed('HG7_res3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_resPool3_skip'))

        (self.feed('HG7_res3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_resPool3_batch4')
             .relu(name='HG7_resPool3_relu4')
             .max_pool(2, 2, 2, 2, name='HG7_resPool3_pool')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='HG7_resPool3_conv4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_resPool3_batch5')
             .relu(name='HG7_resPool3_relu5')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='HG7_resPool3_conv5')
             .upsample(32, 32, name='HG7_resPool3_upSample'))


        (self.feed('HG7_resPool3_conv3',
                   'HG7_resPool3_skip',
                   'HG7_resPool3_upSample')
         .add(name='HG7_resPool3'))




# pool2
        (self.feed('HG7_res2')
             .max_pool(2, 2, 2, 2, name='HG7_pool2'))


# res4
        (self.feed('HG7_pool2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res4_batch1')
             .relu(name='HG7_res4_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG7_res4_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res4_batch2')
             .relu(name='HG7_res4_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG7_res4_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res4_batch3')
             .relu(name='HG7_res4_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res4_conv3'))

        (self.feed('HG7_pool2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res4_skip'))

        (self.feed('HG7_res4_conv3',
                   'HG7_res4_skip')
         .add(name='HG7_res4'))
# id:013 max-pooling
        # (self.feed('HG7_pool3')
        #      .max_pool(2, 2, 2, 2, name='HG7_pool4'))


# res5
        (self.feed('HG7_res4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res5_batch1')
             .relu(name='HG7_res5_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG7_res5_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res5_batch2')
             .relu(name='HG7_res5_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG7_res5_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res5_batch3')
             .relu(name='HG7_res5_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res5_conv3'))

        (self.feed('HG7_res4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res5_skip'))

        (self.feed('HG7_res5_conv3',
                   'HG7_res5_skip')
         .add(name='HG7_res5'))


# pool3
        (self.feed('HG7_res4')
             .max_pool(2, 2, 2, 2, name='HG7_pool3'))


# res6
        (self.feed('HG7_pool3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res6_batch1')
             .relu(name='HG7_res6_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG7_res6_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res6_batch2')
             .relu(name='HG7_res6_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG7_res6_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res6_batch3')
             .relu(name='HG7_res6_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res6_conv3'))

        (self.feed('HG7_pool3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res6_skip'))

        (self.feed('HG7_res6_conv3',
                   'HG7_res6_skip')
         .add(name='HG7_res6'))

# res7
        (self.feed('HG7_res6')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res7_batch1')
             .relu(name='HG7_res7_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG7_res7_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res7_batch2')
             .relu(name='HG7_res7_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG7_res7_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res7_batch3')
             .relu(name='HG7_res7_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res7_conv3'))

        (self.feed('HG7_res6')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res7_skip'))

        (self.feed('HG7_res7_conv3',
                   'HG7_res7_skip')
         .add(name='HG7_res7'))


# pool4
        (self.feed('HG7_res6')
             .max_pool(2, 2, 2, 2, name='HG7_pool4'))

# res8
        (self.feed('HG7_pool4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res8_batch1')
             .relu(name='HG7_res8_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG7_res8_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res8_batch2')
             .relu(name='HG7_res8_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG7_res8_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res8_batch3')
             .relu(name='HG7_res8_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res8_conv3'))

        (self.feed('HG7_pool4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res8_skip'))

        (self.feed('HG7_res8_conv3',
                   'HG7_res8_skip')
         .add(name='HG7_res8'))

# res9
        (self.feed('HG7_res8')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res9_batch1')
             .relu(name='HG7_res9_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG7_res9_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res9_batch2')
             .relu(name='HG7_res9_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG7_res9_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res9_batch3')
             .relu(name='HG7_res9_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res9_conv3'))

        (self.feed('HG7_res8')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res9_skip'))

        (self.feed('HG7_res9_conv3',
                   'HG7_res9_skip')
         .add(name='HG7_res9'))

# res10
        (self.feed('HG7_res9')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res10_batch1')
             .relu(name='HG7_res10_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG7_res10_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res10_batch2')
             .relu(name='HG7_res10_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG7_res10_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res10_batch3')
             .relu(name='HG7_res10_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res10_conv3'))

        (self.feed('HG7_res9')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res10_skip'))

        (self.feed('HG7_res10_conv3',
                   'HG7_res10_skip')
         .add(name='HG7_res10'))


# upsample1
        (self.feed('HG7_res10')
             .upsample(8, 8, name='HG7_upSample1'))

# upsample1 + up1(Hg_res7)
        (self.feed('HG7_upSample1',
                   'HG7_res7')
         .add(name='HG7_add1'))


# res11
        (self.feed('HG7_add1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res11_batch1')
             .relu(name='HG7_res11_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG7_res11_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res11_batch2')
             .relu(name='HG7_res11_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG7_res11_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res11_batch3')
             .relu(name='HG7_res11_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res11_conv3'))

        (self.feed('HG7_add1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res11_skip'))

        (self.feed('HG7_res11_conv3',
                   'HG7_res11_skip')
         .add(name='HG7_res11'))


# upsample2
        (self.feed('HG7_res11')
             .upsample(16, 16, name='HG7_upSample2'))

# upsample2 + up1(Hg_res5)
        (self.feed('HG7_upSample2',
                   'HG7_res5')
         .add(name='HG7_add2'))


# res12
        (self.feed('HG7_add2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res12_batch1')
             .relu(name='HG7_res12_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG7_res12_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res12_batch2')
             .relu(name='HG7_res12_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG7_res12_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res12_batch3')
             .relu(name='HG7_res12_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res12_conv3'))

        (self.feed('HG7_add2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res12_skip'))

        (self.feed('HG7_res12_conv3',
                   'HG7_res12_skip')
         .add(name='HG7_res12'))


# upsample3
        (self.feed('HG7_res12')
             .upsample(32, 32, name='HG7_upSample3'))

# upsample3 + HG7_resPool3
        (self.feed('HG7_upSample3',
                   'HG7_resPool3')
         .add(name='HG7_add3'))


# res13
        (self.feed('HG7_add3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res13_batch1')
             .relu(name='HG7_res13_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG7_res13_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res13_batch2')
             .relu(name='HG7_res13_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG7_res13_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res13_batch3')
             .relu(name='HG7_res13_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res13_conv3'))

        (self.feed('HG7_add3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res13_skip'))

        (self.feed('HG7_res13_conv3',
                   'HG7_res13_skip')
         .add(name='HG7_res13'))


# upsample4
        (self.feed('HG7_res13')
             .upsample(64, 64, name='HG7_upSample4'))

# upsample4 + HG7_resPool2
        (self.feed('HG7_upSample4',
                   'HG7_resPool2')
         .add(name='HG7_add4'))

###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  Hourglass  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

################################# HG7 postprocess #################
# Linear layer to produce first set of predictions
# ll1
        (self.feed('HG7_add4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_linearfunc1_conv1', padding='SAME')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_linearfunc1_batch1')
             .relu(name='HG7_linearfunc1_relu'))
# ll2
        (self.feed('HG7_linearfunc1_relu')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_linearfunc2_conv1', padding='SAME')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_linearfunc2_batch1')
             .relu(name='HG7_linearfunc2_relu'))

# att itersize=3

# att U input: ll2
        (self.feed('HG7_linearfunc2_relu')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG7_att_U', padding='SAME'))
# att i=1 conv  C(1)
        with tf.variable_scope('HG7_att_share_param', reuse=False):
            (self.feed('HG7_att_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_att_spConv1', padding='SAME'))
# att Qtemp
        (self.feed('HG7_att_spConv1',
                   'HG7_att_U')
         .add(name='HG7_att_Qtemp1_add')
         .sigmoid(name='HG7_att_Qtemp1'))

# att i=2 conv  C(2)
        with tf.variable_scope('HG7_att_share_param', reuse=True):
            (self.feed('HG7_att_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_att_spConv1', padding='SAME'))
# att Qtemp2
        (self.feed('HG7_att_spConv1',
                   'HG7_att_U')
         .add(name='HG7_att_Qtemp2_add')
         .sigmoid(name='HG7_att_Qtemp2'))

# att i=3 conv  C(3)
        with tf.variable_scope('HG7_att_share_param', reuse=True):
            (self.feed('HG7_att_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_att_spConv1', padding='SAME'))
# att Qtemp
        (self.feed('HG7_att_spConv1',
                   'HG7_att_U')
         .add(name='HG7_att_Qtemp3_add')
         .sigmoid(name='HG7_att_Qtemp3'))
# att pfeat
        (self.feed('HG7_att_Qtemp3')
         .replicate(256, 3, name='HG7_att_pfeat_replicate'))

        (self.feed('HG7_linearfunc2_relu',
                   'HG7_att_pfeat_replicate')
         .multiply2(name='HG7_att_pfeat_multiply'))

# tmpOut 
        # (self.feed('HG7_att_pfeat_multiply')
        #      .conv(1, 1, n_classes, 1, 1, biased=True, relu=False, name='HG7_att_Heatmap', padding='SAME'))










# # tmpOut att U input: HG7_att_pfeat_multiply
#         (self.feed('HG7_att_pfeat_multiply')
#              .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut_U', padding='SAME'))
# # tmpOut att i=1 conv  C(1)
#         with tf.variable_scope('HG7_tmpOut_share_param', reuse=False):
#             (self.feed('HG7_tmpOut_U')
#                  .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut_spConv1', padding='SAME'))
# # tmpOut att Qtemp
#         (self.feed('HG7_tmpOut_spConv1',
#                    'HG7_tmpOut_U')
#          .add(name='HG7_tmpOut_Qtemp1_add')
#          .sigmoid(name='HG7_tmpOut_Qtemp1'))

# # tmpOut att i=2 conv  C(2)
#         with tf.variable_scope('HG7_tmpOut_share_param', reuse=True):
#             (self.feed('HG7_tmpOut_Qtemp1')
#                  .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut_spConv1', padding='SAME'))
# # tmpOut att Qtemp2
#         (self.feed('HG7_tmpOut_spConv1',
#                    'HG7_tmpOut_U')
#          .add(name='HG7_tmpOut_Qtemp2_add')
#          .sigmoid(name='HG7_tmpOut_Qtemp2'))

# # tmpOut att i=3 conv  C(3)
#         with tf.variable_scope('HG7_tmpOut_share_param', reuse=True):
#             (self.feed('HG7_tmpOut_Qtemp2')
#                  .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut_spConv1', padding='SAME'))
# # tmpOut att Qtemp
#         (self.feed('HG7_tmpOut_spConv1',
#                    'HG7_tmpOut_U')
#          .add(name='HG7_tmpOut_Qtemp3_add')
#          .sigmoid(name='HG7_tmpOut_Qtemp3'))
# # tmpOut att pfeat
#         (self.feed('HG7_tmpOut_Qtemp3')
#          .replicate(256, 3, name='HG7_tmpOut_pfeat_replicate'))

#         (self.feed('HG7_att_pfeat_multiply',
#                    'HG7_tmpOut_pfeat_replicate')
#          .multiply2(name='HG7_tmpOut_pfeat_multiply'))

#         (self.feed('HG7_tmpOut_pfeat_multiply')
#              .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut_s', padding='SAME'))

#         (self.feed('HG7_tmpOut_s')
#          .replicate(16, 3, name='HG7_Heatmap'))        















# attention map for part1
# tmpOut1 att U input: HG7_att_pfeat_multiply
        (self.feed('HG7_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut1_U', padding='SAME'))
# tmpOut1 att i=1 conv  C(1)
        with tf.variable_scope('HG7_tmpOut1_share_param', reuse=False):
            (self.feed('HG7_tmpOut1_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut1_spConv1', padding='SAME'))
# tmpOut1 att Qtemp
        (self.feed('HG7_tmpOut1_spConv1',
                   'HG7_tmpOut1_U')
         .add(name='HG7_tmpOut1_Qtemp1_add')
         .sigmoid(name='HG7_tmpOut1_Qtemp1'))

# tmpOut1 att i=2 conv  C(2)
        with tf.variable_scope('HG7_tmpOut1_share_param', reuse=True):
            (self.feed('HG7_tmpOut1_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut1_spConv1', padding='SAME'))
# tmpOut1 att Qtemp2
        (self.feed('HG7_tmpOut1_spConv1',
                   'HG7_tmpOut1_U')
         .add(name='HG7_tmpOut1_Qtemp2_add')
         .sigmoid(name='HG7_tmpOut1_Qtemp2'))

# tmpOut1 att i=3 conv  C(3)
        with tf.variable_scope('HG7_tmpOut1_share_param', reuse=True):
            (self.feed('HG7_tmpOut1_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut1_spConv1', padding='SAME'))
# tmpOut1 att Qtemp
        (self.feed('HG7_tmpOut1_spConv1',
                   'HG7_tmpOut1_U')
         .add(name='HG7_tmpOut1_Qtemp3_add')
         .sigmoid(name='HG7_tmpOut1_Qtemp3'))
# tmpOut1 att pfeat
        (self.feed('HG7_tmpOut1_Qtemp3')
         .replicate(256, 3, name='HG7_tmpOut1_pfeat_replicate'))

        (self.feed('HG7_att_pfeat_multiply',
                   'HG7_tmpOut1_pfeat_replicate')
         .multiply2(name='HG7_tmpOut1_pfeat_multiply'))

        (self.feed('HG7_tmpOut1_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut1_s', padding='SAME'))




# attention map for part2
# tmpOut2 att U input: HG7_att_pfeat_multiply
        (self.feed('HG7_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut2_U', padding='SAME'))
# tmpOut2 att i=1 conv  C(1)
        with tf.variable_scope('HG7_tmpOut2_share_param', reuse=False):
            (self.feed('HG7_tmpOut2_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut2_spConv1', padding='SAME'))
# tmpOut2 att Qtemp
        (self.feed('HG7_tmpOut2_spConv1',
                   'HG7_tmpOut2_U')
         .add(name='HG7_tmpOut2_Qtemp1_add')
         .sigmoid(name='HG7_tmpOut2_Qtemp1'))

# tmpOut2 att i=2 conv  C(2)
        with tf.variable_scope('HG7_tmpOut2_share_param', reuse=True):
            (self.feed('HG7_tmpOut2_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut2_spConv1', padding='SAME'))
# tmpOut2 att Qtemp2
        (self.feed('HG7_tmpOut2_spConv1',
                   'HG7_tmpOut2_U')
         .add(name='HG7_tmpOut2_Qtemp2_add')
         .sigmoid(name='HG7_tmpOut2_Qtemp2'))

# tmpOut2 att i=3 conv  C(3)
        with tf.variable_scope('HG7_tmpOut2_share_param', reuse=True):
            (self.feed('HG7_tmpOut2_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut2_spConv1', padding='SAME'))
# tmpOut2 att Qtemp
        (self.feed('HG7_tmpOut2_spConv1',
                   'HG7_tmpOut2_U')
         .add(name='HG7_tmpOut2_Qtemp3_add')
         .sigmoid(name='HG7_tmpOut2_Qtemp3'))
# tmpOut2 att pfeat
        (self.feed('HG7_tmpOut2_Qtemp3')
         .replicate(256, 3, name='HG7_tmpOut2_pfeat_replicate'))

        (self.feed('HG7_att_pfeat_multiply',
                   'HG7_tmpOut2_pfeat_replicate')
         .multiply2(name='HG7_tmpOut2_pfeat_multiply'))

        (self.feed('HG7_tmpOut2_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut2_s', padding='SAME'))




# attention map for part3
# tmpOut3 att U input: HG7_att_pfeat_multiply
        (self.feed('HG7_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut3_U', padding='SAME'))
# tmpOut3 att i=1 conv  C(1)
        with tf.variable_scope('HG7_tmpOut3_share_param', reuse=False):
            (self.feed('HG7_tmpOut3_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut3_spConv1', padding='SAME'))
# tmpOut3 att Qtemp
        (self.feed('HG7_tmpOut3_spConv1',
                   'HG7_tmpOut3_U')
         .add(name='HG7_tmpOut3_Qtemp1_add')
         .sigmoid(name='HG7_tmpOut3_Qtemp1'))

# tmpOut3 att i=2 conv  C(2)
        with tf.variable_scope('HG7_tmpOut3_share_param', reuse=True):
            (self.feed('HG7_tmpOut3_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut3_spConv1', padding='SAME'))
# tmpOut3 att Qtemp2
        (self.feed('HG7_tmpOut3_spConv1',
                   'HG7_tmpOut3_U')
         .add(name='HG7_tmpOut3_Qtemp2_add')
         .sigmoid(name='HG7_tmpOut3_Qtemp2'))

# tmpOut3 att i=3 conv  C(3)
        with tf.variable_scope('HG7_tmpOut3_share_param', reuse=True):
            (self.feed('HG7_tmpOut3_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut3_spConv1', padding='SAME'))
# tmpOut3 att Qtemp
        (self.feed('HG7_tmpOut3_spConv1',
                   'HG7_tmpOut3_U')
         .add(name='HG7_tmpOut3_Qtemp3_add')
         .sigmoid(name='HG7_tmpOut3_Qtemp3'))
# tmpOut3 att pfeat
        (self.feed('HG7_tmpOut3_Qtemp3')
         .replicate(256, 3, name='HG7_tmpOut3_pfeat_replicate'))

        (self.feed('HG7_att_pfeat_multiply',
                   'HG7_tmpOut3_pfeat_replicate')
         .multiply2(name='HG7_tmpOut3_pfeat_multiply'))

        (self.feed('HG7_tmpOut3_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut3_s', padding='SAME'))






# attention map for part4
# tmpOut4 att U input: HG7_att_pfeat_multiply
        (self.feed('HG7_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut4_U', padding='SAME'))
# tmpOut4 att i=1 conv  C(1)
        with tf.variable_scope('HG7_tmpOut4_share_param', reuse=False):
            (self.feed('HG7_tmpOut4_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut4_spConv1', padding='SAME'))
# tmpOut4 att Qtemp
        (self.feed('HG7_tmpOut4_spConv1',
                   'HG7_tmpOut4_U')
         .add(name='HG7_tmpOut4_Qtemp1_add')
         .sigmoid(name='HG7_tmpOut4_Qtemp1'))

# tmpOut4 att i=2 conv  C(2)
        with tf.variable_scope('HG7_tmpOut4_share_param', reuse=True):
            (self.feed('HG7_tmpOut4_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut4_spConv1', padding='SAME'))
# tmpOut4 att Qtemp2
        (self.feed('HG7_tmpOut4_spConv1',
                   'HG7_tmpOut4_U')
         .add(name='HG7_tmpOut4_Qtemp2_add')
         .sigmoid(name='HG7_tmpOut4_Qtemp2'))

# tmpOut4 att i=3 conv  C(3)
        with tf.variable_scope('HG7_tmpOut4_share_param', reuse=True):
            (self.feed('HG7_tmpOut4_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut4_spConv1', padding='SAME'))
# tmpOut4 att Qtemp
        (self.feed('HG7_tmpOut4_spConv1',
                   'HG7_tmpOut4_U')
         .add(name='HG7_tmpOut4_Qtemp3_add')
         .sigmoid(name='HG7_tmpOut4_Qtemp3'))
# tmpOut4 att pfeat
        (self.feed('HG7_tmpOut4_Qtemp3')
         .replicate(256, 3, name='HG7_tmpOut4_pfeat_replicate'))

        (self.feed('HG7_att_pfeat_multiply',
                   'HG7_tmpOut4_pfeat_replicate')
         .multiply2(name='HG7_tmpOut4_pfeat_multiply'))

        (self.feed('HG7_tmpOut4_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut4_s', padding='SAME'))









# attention map for part5
# tmpOut5 att U input: HG7_att_pfeat_multiply
        (self.feed('HG7_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut5_U', padding='SAME'))
# tmpOut5 att i=1 conv  C(1)
        with tf.variable_scope('HG7_tmpOut5_share_param', reuse=False):
            (self.feed('HG7_tmpOut5_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut5_spConv1', padding='SAME'))
# tmpOut5 att Qtemp
        (self.feed('HG7_tmpOut5_spConv1',
                   'HG7_tmpOut5_U')
         .add(name='HG7_tmpOut5_Qtemp1_add')
         .sigmoid(name='HG7_tmpOut5_Qtemp1'))

# tmpOut5 att i=2 conv  C(2)
        with tf.variable_scope('HG7_tmpOut5_share_param', reuse=True):
            (self.feed('HG7_tmpOut5_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut5_spConv1', padding='SAME'))
# tmpOut5 att Qtemp2
        (self.feed('HG7_tmpOut5_spConv1',
                   'HG7_tmpOut5_U')
         .add(name='HG7_tmpOut5_Qtemp2_add')
         .sigmoid(name='HG7_tmpOut5_Qtemp2'))

# tmpOut5 att i=3 conv  C(3)
        with tf.variable_scope('HG7_tmpOut5_share_param', reuse=True):
            (self.feed('HG7_tmpOut5_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut5_spConv1', padding='SAME'))
# tmpOut5 att Qtemp
        (self.feed('HG7_tmpOut5_spConv1',
                   'HG7_tmpOut5_U')
         .add(name='HG7_tmpOut5_Qtemp3_add')
         .sigmoid(name='HG7_tmpOut5_Qtemp3'))
# tmpOut5 att pfeat
        (self.feed('HG7_tmpOut5_Qtemp3')
         .replicate(256, 3, name='HG7_tmpOut5_pfeat_replicate'))

        (self.feed('HG7_att_pfeat_multiply',
                   'HG7_tmpOut5_pfeat_replicate')
         .multiply2(name='HG7_tmpOut5_pfeat_multiply'))

        (self.feed('HG7_tmpOut5_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut5_s', padding='SAME'))










# attention map for part6
# tmpOut6 att U input: HG7_att_pfeat_multiply
        (self.feed('HG7_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut6_U', padding='SAME'))
# tmpOut6 att i=1 conv  C(1)
        with tf.variable_scope('HG7_tmpOut6_share_param', reuse=False):
            (self.feed('HG7_tmpOut6_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut6_spConv1', padding='SAME'))
# tmpOut6 att Qtemp
        (self.feed('HG7_tmpOut6_spConv1',
                   'HG7_tmpOut6_U')
         .add(name='HG7_tmpOut6_Qtemp1_add')
         .sigmoid(name='HG7_tmpOut6_Qtemp1'))

# tmpOut6 att i=2 conv  C(2)
        with tf.variable_scope('HG7_tmpOut6_share_param', reuse=True):
            (self.feed('HG7_tmpOut6_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut6_spConv1', padding='SAME'))
# tmpOut6 att Qtemp2
        (self.feed('HG7_tmpOut6_spConv1',
                   'HG7_tmpOut6_U')
         .add(name='HG7_tmpOut6_Qtemp2_add')
         .sigmoid(name='HG7_tmpOut6_Qtemp2'))

# tmpOut6 att i=3 conv  C(3)
        with tf.variable_scope('HG7_tmpOut6_share_param', reuse=True):
            (self.feed('HG7_tmpOut6_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut6_spConv1', padding='SAME'))
# tmpOut6 att Qtemp
        (self.feed('HG7_tmpOut6_spConv1',
                   'HG7_tmpOut6_U')
         .add(name='HG7_tmpOut6_Qtemp3_add')
         .sigmoid(name='HG7_tmpOut6_Qtemp3'))
# tmpOut6 att pfeat
        (self.feed('HG7_tmpOut6_Qtemp3')
         .replicate(256, 3, name='HG7_tmpOut6_pfeat_replicate'))

        (self.feed('HG7_att_pfeat_multiply',
                   'HG7_tmpOut6_pfeat_replicate')
         .multiply2(name='HG7_tmpOut6_pfeat_multiply'))

        (self.feed('HG7_tmpOut6_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut6_s', padding='SAME'))





# attention map for part7
# tmpOut7 att U input: HG7_att_pfeat_multiply
        (self.feed('HG7_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut7_U', padding='SAME'))
# tmpOut7 att i=1 conv  C(1)
        with tf.variable_scope('HG7_tmpOut7_share_param', reuse=False):
            (self.feed('HG7_tmpOut7_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut7_spConv1', padding='SAME'))
# tmpOut7 att Qtemp
        (self.feed('HG7_tmpOut7_spConv1',
                   'HG7_tmpOut7_U')
         .add(name='HG7_tmpOut7_Qtemp1_add')
         .sigmoid(name='HG7_tmpOut7_Qtemp1'))

# tmpOut7 att i=2 conv  C(2)
        with tf.variable_scope('HG7_tmpOut7_share_param', reuse=True):
            (self.feed('HG7_tmpOut7_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut7_spConv1', padding='SAME'))
# tmpOut7 att Qtemp2
        (self.feed('HG7_tmpOut7_spConv1',
                   'HG7_tmpOut7_U')
         .add(name='HG7_tmpOut7_Qtemp2_add')
         .sigmoid(name='HG7_tmpOut7_Qtemp2'))

# tmpOut7 att i=3 conv  C(3)
        with tf.variable_scope('HG7_tmpOut7_share_param', reuse=True):
            (self.feed('HG7_tmpOut7_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut7_spConv1', padding='SAME'))
# tmpOut7 att Qtemp
        (self.feed('HG7_tmpOut7_spConv1',
                   'HG7_tmpOut7_U')
         .add(name='HG7_tmpOut7_Qtemp3_add')
         .sigmoid(name='HG7_tmpOut7_Qtemp3'))
# tmpOut7 att pfeat
        (self.feed('HG7_tmpOut7_Qtemp3')
         .replicate(256, 3, name='HG7_tmpOut7_pfeat_replicate'))

        (self.feed('HG7_att_pfeat_multiply',
                   'HG7_tmpOut7_pfeat_replicate')
         .multiply2(name='HG7_tmpOut7_pfeat_multiply'))

        (self.feed('HG7_tmpOut7_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut7_s', padding='SAME'))








# attention map for part8
# tmpOut8 att U input: HG7_att_pfeat_multiply
        (self.feed('HG7_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut8_U', padding='SAME'))
# tmpOut8 att i=1 conv  C(1)
        with tf.variable_scope('HG7_tmpOut8_share_param', reuse=False):
            (self.feed('HG7_tmpOut8_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut8_spConv1', padding='SAME'))
# tmpOut8 att Qtemp
        (self.feed('HG7_tmpOut8_spConv1',
                   'HG7_tmpOut8_U')
         .add(name='HG7_tmpOut8_Qtemp1_add')
         .sigmoid(name='HG7_tmpOut8_Qtemp1'))

# tmpOut8 att i=2 conv  C(2)
        with tf.variable_scope('HG7_tmpOut8_share_param', reuse=True):
            (self.feed('HG7_tmpOut8_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut8_spConv1', padding='SAME'))
# tmpOut8 att Qtemp2
        (self.feed('HG7_tmpOut8_spConv1',
                   'HG7_tmpOut8_U')
         .add(name='HG7_tmpOut8_Qtemp2_add')
         .sigmoid(name='HG7_tmpOut8_Qtemp2'))

# tmpOut8 att i=3 conv  C(3)
        with tf.variable_scope('HG7_tmpOut8_share_param', reuse=True):
            (self.feed('HG7_tmpOut8_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut8_spConv1', padding='SAME'))
# tmpOut8 att Qtemp
        (self.feed('HG7_tmpOut8_spConv1',
                   'HG7_tmpOut8_U')
         .add(name='HG7_tmpOut8_Qtemp3_add')
         .sigmoid(name='HG7_tmpOut8_Qtemp3'))
# tmpOut8 att pfeat
        (self.feed('HG7_tmpOut8_Qtemp3')
         .replicate(256, 3, name='HG7_tmpOut8_pfeat_replicate'))

        (self.feed('HG7_att_pfeat_multiply',
                   'HG7_tmpOut8_pfeat_replicate')
         .multiply2(name='HG7_tmpOut8_pfeat_multiply'))

        (self.feed('HG7_tmpOut8_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut8_s', padding='SAME'))











# attention map for part9
# tmpOut9 att U input: HG7_att_pfeat_multiply
        (self.feed('HG7_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut9_U', padding='SAME'))
# tmpOut9 att i=1 conv  C(1)
        with tf.variable_scope('HG7_tmpOut9_share_param', reuse=False):
            (self.feed('HG7_tmpOut9_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut9_spConv1', padding='SAME'))
# tmpOut9 att Qtemp
        (self.feed('HG7_tmpOut9_spConv1',
                   'HG7_tmpOut9_U')
         .add(name='HG7_tmpOut9_Qtemp1_add')
         .sigmoid(name='HG7_tmpOut9_Qtemp1'))

# tmpOut9 att i=2 conv  C(2)
        with tf.variable_scope('HG7_tmpOut9_share_param', reuse=True):
            (self.feed('HG7_tmpOut9_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut9_spConv1', padding='SAME'))
# tmpOut9 att Qtemp2
        (self.feed('HG7_tmpOut9_spConv1',
                   'HG7_tmpOut9_U')
         .add(name='HG7_tmpOut9_Qtemp2_add')
         .sigmoid(name='HG7_tmpOut9_Qtemp2'))

# tmpOut9 att i=3 conv  C(3)
        with tf.variable_scope('HG7_tmpOut9_share_param', reuse=True):
            (self.feed('HG7_tmpOut9_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut9_spConv1', padding='SAME'))
# tmpOut9 att Qtemp
        (self.feed('HG7_tmpOut9_spConv1',
                   'HG7_tmpOut9_U')
         .add(name='HG7_tmpOut9_Qtemp3_add')
         .sigmoid(name='HG7_tmpOut9_Qtemp3'))
# tmpOut9 att pfeat
        (self.feed('HG7_tmpOut9_Qtemp3')
         .replicate(256, 3, name='HG7_tmpOut9_pfeat_replicate'))

        (self.feed('HG7_att_pfeat_multiply',
                   'HG7_tmpOut9_pfeat_replicate')
         .multiply2(name='HG7_tmpOut9_pfeat_multiply'))

        (self.feed('HG7_tmpOut9_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut9_s', padding='SAME'))












# attention map for part10
# tmpOut10 att U input: HG7_att_pfeat_multiply
        (self.feed('HG7_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut10_U', padding='SAME'))
# tmpOut10 att i=1 conv  C(1)
        with tf.variable_scope('HG7_tmpOut10_share_param', reuse=False):
            (self.feed('HG7_tmpOut10_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut10_spConv1', padding='SAME'))
# tmpOut10 att Qtemp
        (self.feed('HG7_tmpOut10_spConv1',
                   'HG7_tmpOut10_U')
         .add(name='HG7_tmpOut10_Qtemp1_add')
         .sigmoid(name='HG7_tmpOut10_Qtemp1'))

# tmpOut10 att i=2 conv  C(2)
        with tf.variable_scope('HG7_tmpOut10_share_param', reuse=True):
            (self.feed('HG7_tmpOut10_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut10_spConv1', padding='SAME'))
# tmpOut10 att Qtemp2
        (self.feed('HG7_tmpOut10_spConv1',
                   'HG7_tmpOut10_U')
         .add(name='HG7_tmpOut10_Qtemp2_add')
         .sigmoid(name='HG7_tmpOut10_Qtemp2'))

# tmpOut10 att i=3 conv  C(3)
        with tf.variable_scope('HG7_tmpOut10_share_param', reuse=True):
            (self.feed('HG7_tmpOut10_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut10_spConv1', padding='SAME'))
# tmpOut10 att Qtemp
        (self.feed('HG7_tmpOut10_spConv1',
                   'HG7_tmpOut10_U')
         .add(name='HG7_tmpOut10_Qtemp3_add')
         .sigmoid(name='HG7_tmpOut10_Qtemp3'))
# tmpOut10 att pfeat
        (self.feed('HG7_tmpOut10_Qtemp3')
         .replicate(256, 3, name='HG7_tmpOut10_pfeat_replicate'))

        (self.feed('HG7_att_pfeat_multiply',
                   'HG7_tmpOut10_pfeat_replicate')
         .multiply2(name='HG7_tmpOut10_pfeat_multiply'))

        (self.feed('HG7_tmpOut10_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut10_s', padding='SAME'))










# attention map for part11
# tmpOut11 att U input: HG7_att_pfeat_multiply
        (self.feed('HG7_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut11_U', padding='SAME'))
# tmpOut11 att i=1 conv  C(1)
        with tf.variable_scope('HG7_tmpOut11_share_param', reuse=False):
            (self.feed('HG7_tmpOut11_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut11_spConv1', padding='SAME'))
# tmpOut11 att Qtemp
        (self.feed('HG7_tmpOut11_spConv1',
                   'HG7_tmpOut11_U')
         .add(name='HG7_tmpOut11_Qtemp1_add')
         .sigmoid(name='HG7_tmpOut11_Qtemp1'))

# tmpOut11 att i=2 conv  C(2)
        with tf.variable_scope('HG7_tmpOut11_share_param', reuse=True):
            (self.feed('HG7_tmpOut11_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut11_spConv1', padding='SAME'))
# tmpOut11 att Qtemp2
        (self.feed('HG7_tmpOut11_spConv1',
                   'HG7_tmpOut11_U')
         .add(name='HG7_tmpOut11_Qtemp2_add')
         .sigmoid(name='HG7_tmpOut11_Qtemp2'))

# tmpOut11 att i=3 conv  C(3)
        with tf.variable_scope('HG7_tmpOut11_share_param', reuse=True):
            (self.feed('HG7_tmpOut11_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut11_spConv1', padding='SAME'))
# tmpOut11 att Qtemp
        (self.feed('HG7_tmpOut11_spConv1',
                   'HG7_tmpOut11_U')
         .add(name='HG7_tmpOut11_Qtemp3_add')
         .sigmoid(name='HG7_tmpOut11_Qtemp3'))
# tmpOut11 att pfeat
        (self.feed('HG7_tmpOut11_Qtemp3')
         .replicate(256, 3, name='HG7_tmpOut11_pfeat_replicate'))

        (self.feed('HG7_att_pfeat_multiply',
                   'HG7_tmpOut11_pfeat_replicate')
         .multiply2(name='HG7_tmpOut11_pfeat_multiply'))

        (self.feed('HG7_tmpOut11_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut11_s', padding='SAME'))











# attention map for part12
# tmpOut12 att U input: HG7_att_pfeat_multiply
        (self.feed('HG7_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut12_U', padding='SAME'))
# tmpOut12 att i=1 conv  C(1)
        with tf.variable_scope('HG7_tmpOut12_share_param', reuse=False):
            (self.feed('HG7_tmpOut12_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut12_spConv1', padding='SAME'))
# tmpOut12 att Qtemp
        (self.feed('HG7_tmpOut12_spConv1',
                   'HG7_tmpOut12_U')
         .add(name='HG7_tmpOut12_Qtemp1_add')
         .sigmoid(name='HG7_tmpOut12_Qtemp1'))

# tmpOut12 att i=2 conv  C(2)
        with tf.variable_scope('HG7_tmpOut12_share_param', reuse=True):
            (self.feed('HG7_tmpOut12_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut12_spConv1', padding='SAME'))
# tmpOut12 att Qtemp2
        (self.feed('HG7_tmpOut12_spConv1',
                   'HG7_tmpOut12_U')
         .add(name='HG7_tmpOut12_Qtemp2_add')
         .sigmoid(name='HG7_tmpOut12_Qtemp2'))

# tmpOut12 att i=3 conv  C(3)
        with tf.variable_scope('HG7_tmpOut12_share_param', reuse=True):
            (self.feed('HG7_tmpOut12_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut12_spConv1', padding='SAME'))
# tmpOut12 att Qtemp
        (self.feed('HG7_tmpOut12_spConv1',
                   'HG7_tmpOut12_U')
         .add(name='HG7_tmpOut12_Qtemp3_add')
         .sigmoid(name='HG7_tmpOut12_Qtemp3'))
# tmpOut12 att pfeat
        (self.feed('HG7_tmpOut12_Qtemp3')
         .replicate(256, 3, name='HG7_tmpOut12_pfeat_replicate'))

        (self.feed('HG7_att_pfeat_multiply',
                   'HG7_tmpOut12_pfeat_replicate')
         .multiply2(name='HG7_tmpOut12_pfeat_multiply'))

        (self.feed('HG7_tmpOut12_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut12_s', padding='SAME'))










# attention map for part13
# tmpOut13 att U input: HG7_att_pfeat_multiply
        (self.feed('HG7_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut13_U', padding='SAME'))
# tmpOut13 att i=1 conv  C(1)
        with tf.variable_scope('HG7_tmpOut13_share_param', reuse=False):
            (self.feed('HG7_tmpOut13_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut13_spConv1', padding='SAME'))
# tmpOut13 att Qtemp
        (self.feed('HG7_tmpOut13_spConv1',
                   'HG7_tmpOut13_U')
         .add(name='HG7_tmpOut13_Qtemp1_add')
         .sigmoid(name='HG7_tmpOut13_Qtemp1'))

# tmpOut13 att i=2 conv  C(2)
        with tf.variable_scope('HG7_tmpOut13_share_param', reuse=True):
            (self.feed('HG7_tmpOut13_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut13_spConv1', padding='SAME'))
# tmpOut13 att Qtemp2
        (self.feed('HG7_tmpOut13_spConv1',
                   'HG7_tmpOut13_U')
         .add(name='HG7_tmpOut13_Qtemp2_add')
         .sigmoid(name='HG7_tmpOut13_Qtemp2'))

# tmpOut13 att i=3 conv  C(3)
        with tf.variable_scope('HG7_tmpOut13_share_param', reuse=True):
            (self.feed('HG7_tmpOut13_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut13_spConv1', padding='SAME'))
# tmpOut13 att Qtemp
        (self.feed('HG7_tmpOut13_spConv1',
                   'HG7_tmpOut13_U')
         .add(name='HG7_tmpOut13_Qtemp3_add')
         .sigmoid(name='HG7_tmpOut13_Qtemp3'))
# tmpOut13 att pfeat
        (self.feed('HG7_tmpOut13_Qtemp3')
         .replicate(256, 3, name='HG7_tmpOut13_pfeat_replicate'))

        (self.feed('HG7_att_pfeat_multiply',
                   'HG7_tmpOut13_pfeat_replicate')
         .multiply2(name='HG7_tmpOut13_pfeat_multiply'))

        (self.feed('HG7_tmpOut13_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut13_s', padding='SAME'))










# attention map for part14
# tmpOut14 att U input: HG7_att_pfeat_multiply
        (self.feed('HG7_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut14_U', padding='SAME'))
# tmpOut14 att i=1 conv  C(1)
        with tf.variable_scope('HG7_tmpOut14_share_param', reuse=False):
            (self.feed('HG7_tmpOut14_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut14_spConv1', padding='SAME'))
# tmpOut14 att Qtemp
        (self.feed('HG7_tmpOut14_spConv1',
                   'HG7_tmpOut14_U')
         .add(name='HG7_tmpOut14_Qtemp1_add')
         .sigmoid(name='HG7_tmpOut14_Qtemp1'))

# tmpOut14 att i=2 conv  C(2)
        with tf.variable_scope('HG7_tmpOut14_share_param', reuse=True):
            (self.feed('HG7_tmpOut14_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut14_spConv1', padding='SAME'))
# tmpOut14 att Qtemp2
        (self.feed('HG7_tmpOut14_spConv1',
                   'HG7_tmpOut14_U')
         .add(name='HG7_tmpOut14_Qtemp2_add')
         .sigmoid(name='HG7_tmpOut14_Qtemp2'))

# tmpOut14 att i=3 conv  C(3)
        with tf.variable_scope('HG7_tmpOut14_share_param', reuse=True):
            (self.feed('HG7_tmpOut14_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut14_spConv1', padding='SAME'))
# tmpOut14 att Qtemp
        (self.feed('HG7_tmpOut14_spConv1',
                   'HG7_tmpOut14_U')
         .add(name='HG7_tmpOut14_Qtemp3_add')
         .sigmoid(name='HG7_tmpOut14_Qtemp3'))
# tmpOut14 att pfeat
        (self.feed('HG7_tmpOut14_Qtemp3')
         .replicate(256, 3, name='HG7_tmpOut14_pfeat_replicate'))

        (self.feed('HG7_att_pfeat_multiply',
                   'HG7_tmpOut14_pfeat_replicate')
         .multiply2(name='HG7_tmpOut14_pfeat_multiply'))

        (self.feed('HG7_tmpOut14_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut14_s', padding='SAME'))










# attention map for part15
# tmpOut15 att U input: HG7_att_pfeat_multiply
        (self.feed('HG7_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut15_U', padding='SAME'))
# tmpOut15 att i=1 conv  C(1)
        with tf.variable_scope('HG7_tmpOut15_share_param', reuse=False):
            (self.feed('HG7_tmpOut15_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut15_spConv1', padding='SAME'))
# tmpOut15 att Qtemp
        (self.feed('HG7_tmpOut15_spConv1',
                   'HG7_tmpOut15_U')
         .add(name='HG7_tmpOut15_Qtemp1_add')
         .sigmoid(name='HG7_tmpOut15_Qtemp1'))

# tmpOut15 att i=2 conv  C(2)
        with tf.variable_scope('HG7_tmpOut15_share_param', reuse=True):
            (self.feed('HG7_tmpOut15_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut15_spConv1', padding='SAME'))
# tmpOut15 att Qtemp2
        (self.feed('HG7_tmpOut15_spConv1',
                   'HG7_tmpOut15_U')
         .add(name='HG7_tmpOut15_Qtemp2_add')
         .sigmoid(name='HG7_tmpOut15_Qtemp2'))

# tmpOut15 att i=3 conv  C(3)
        with tf.variable_scope('HG7_tmpOut15_share_param', reuse=True):
            (self.feed('HG7_tmpOut15_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut15_spConv1', padding='SAME'))
# tmpOut15 att Qtemp
        (self.feed('HG7_tmpOut15_spConv1',
                   'HG7_tmpOut15_U')
         .add(name='HG7_tmpOut15_Qtemp3_add')
         .sigmoid(name='HG7_tmpOut15_Qtemp3'))
# tmpOut15 att pfeat
        (self.feed('HG7_tmpOut15_Qtemp3')
         .replicate(256, 3, name='HG7_tmpOut15_pfeat_replicate'))

        (self.feed('HG7_att_pfeat_multiply',
                   'HG7_tmpOut15_pfeat_replicate')
         .multiply2(name='HG7_tmpOut15_pfeat_multiply'))

        (self.feed('HG7_tmpOut15_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut15_s', padding='SAME'))










# attention map for part16
# tmpOut16 att U input: HG7_att_pfeat_multiply
        (self.feed('HG7_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut16_U', padding='SAME'))
# tmpOut16 att i=1 conv  C(1)
        with tf.variable_scope('HG7_tmpOut16_share_param', reuse=False):
            (self.feed('HG7_tmpOut16_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut16_spConv1', padding='SAME'))
# tmpOut16 att Qtemp
        (self.feed('HG7_tmpOut16_spConv1',
                   'HG7_tmpOut16_U')
         .add(name='HG7_tmpOut16_Qtemp1_add')
         .sigmoid(name='HG7_tmpOut16_Qtemp1'))

# tmpOut16 att i=2 conv  C(2)
        with tf.variable_scope('HG7_tmpOut16_share_param', reuse=True):
            (self.feed('HG7_tmpOut16_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut16_spConv1', padding='SAME'))
# tmpOut16 att Qtemp2
        (self.feed('HG7_tmpOut16_spConv1',
                   'HG7_tmpOut16_U')
         .add(name='HG7_tmpOut16_Qtemp2_add')
         .sigmoid(name='HG7_tmpOut16_Qtemp2'))

# tmpOut16 att i=3 conv  C(3)
        with tf.variable_scope('HG7_tmpOut16_share_param', reuse=True):
            (self.feed('HG7_tmpOut16_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut16_spConv1', padding='SAME'))
# tmpOut16 att Qtemp
        (self.feed('HG7_tmpOut16_spConv1',
                   'HG7_tmpOut16_U')
         .add(name='HG7_tmpOut16_Qtemp3_add')
         .sigmoid(name='HG7_tmpOut16_Qtemp3'))
# tmpOut16 att pfeat
        (self.feed('HG7_tmpOut16_Qtemp3')
         .replicate(256, 3, name='HG7_tmpOut16_pfeat_replicate'))

        (self.feed('HG7_att_pfeat_multiply',
                   'HG7_tmpOut16_pfeat_replicate')
         .multiply2(name='HG7_tmpOut16_pfeat_multiply'))

        (self.feed('HG7_tmpOut16_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG7_tmpOut16_s', padding='SAME'))


        (self.feed('HG7_tmpOut1_s',
                   'HG7_tmpOut2_s',
                   'HG7_tmpOut3_s',
                   'HG7_tmpOut4_s',
                   'HG7_tmpOut5_s',
                   'HG7_tmpOut6_s',
                   'HG7_tmpOut7_s',
                   'HG7_tmpOut8_s',
                   'HG7_tmpOut9_s',
                   'HG7_tmpOut10_s',
                   'HG7_tmpOut11_s',
                   'HG7_tmpOut12_s',
                   'HG7_tmpOut13_s',
                   'HG7_tmpOut14_s',
                   'HG7_tmpOut15_s',
                   'HG7_tmpOut16_s')
         .stack(axis = 3,  name='HG7_Heatmap'))        







###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ HG7 postprocess ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

############################# generate input for next HG ##########

# outmap
        (self.feed('HG7_Heatmap')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_outmap', padding='SAME'))
# ll3
        (self.feed('HG7_linearfunc1_relu')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_linearfunc3_conv1', padding='SAME')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_linearfunc3_batch1')
             .relu(name='HG7_linearfunc3_relu'))
# tmointer
        (self.feed('HG7_input',
                   'HG7_outmap',
                   'HG7_linearfunc3_relu')
         .add(name='HG8_input'))

###^^^^^^^^^^^^^^^^^^^^^^^^^^^ generate input for next HG ^^^^^^^^^^^^^^^^^^^^^^^^^^^^###





















































#######################################  HG8  #########################################
                 ####################  Hourglass  #####################
# res1
        (self.feed('HG8_input')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res1_batch1')
             .relu(name='HG8_res1_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG8_res1_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res1_batch2')
             .relu(name='HG8_res1_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG8_res1_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res1_batch3')
             .relu(name='HG8_res1_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res1_conv3'))

        (self.feed('HG8_input')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res1_skip'))

        (self.feed('HG8_res1_conv3',
                   'HG8_res1_skip')
         .add(name='HG8_res1'))

# resPool1
        (self.feed('HG8_res1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_resPool1_batch1')
             .relu(name='HG8_resPool1_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG8_resPool1_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_resPool1_batch2')
             .relu(name='HG8_resPool1_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG8_resPool1_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_resPool1_batch3')
             .relu(name='HG8_resPool1_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_resPool1_conv3'))

        (self.feed('HG8_res1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_resPool1_skip'))

        (self.feed('HG8_res1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_resPool1_batch4')
             .relu(name='HG8_resPool1_relu4')
             .max_pool(2, 2, 2, 2, name='HG8_resPool1_pool')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='HG8_resPool1_conv4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_resPool1_batch5')
             .relu(name='HG8_resPool1_relu5')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='HG8_resPool1_conv5')
             .upsample(64, 64, name='HG8_resPool1_upSample'))


        (self.feed('HG8_resPool1_conv3',
                   'HG8_resPool1_skip',
                   'HG8_resPool1_upSample')
         .add(name='HG8_resPool1'))



# resPool2
        (self.feed('HG8_resPool1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_resPool2_batch1')
             .relu(name='HG8_resPool2_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG8_resPool2_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_resPool2_batch2')
             .relu(name='HG8_resPool2_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG8_resPool2_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_resPool2_batch3')
             .relu(name='HG8_resPool2_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_resPool2_conv3'))

        (self.feed('HG8_resPool1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_resPool2_skip'))

        (self.feed('HG8_resPool1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_resPool2_batch4')
             .relu(name='HG8_resPool2_relu4')
             .max_pool(2, 2, 2, 2, name='HG8_resPool2_pool')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='HG8_resPool2_conv4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_resPool2_batch5')
             .relu(name='HG8_resPool2_relu5')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='HG8_resPool2_conv5')
             .upsample(64, 64, name='HG8_resPool2_upSample'))


        (self.feed('HG8_resPool2_conv3',
                   'HG8_resPool2_skip',
                   'HG8_resPool2_upSample')
         .add(name='HG8_resPool2'))

# pool1
        (self.feed('HG8_input')
             .max_pool(2, 2, 2, 2, name='HG8_pool1'))


# res2
        (self.feed('HG8_pool1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res2_batch1')
             .relu(name='HG8_res2_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG8_res2_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res2_batch2')
             .relu(name='HG8_res2_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG8_res2_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res2_batch3')
             .relu(name='HG8_res2_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res2_conv3'))

        (self.feed('HG8_pool1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res2_skip'))

        (self.feed('HG8_res2_conv3',
                   'HG8_res2_skip')
         .add(name='HG8_res2'))

# res3
        (self.feed('HG8_res2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res3_batch1')
             .relu(name='HG8_res3_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG8_res3_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res3_batch2')
             .relu(name='HG8_res3_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG8_res3_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res3_batch3')
             .relu(name='HG8_res3_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res3_conv3'))

        (self.feed('HG8_res2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res3_skip'))

        (self.feed('HG8_res3_conv3',
                   'HG8_res3_skip')
         .add(name='HG8_res3'))

# resPool3
        (self.feed('HG8_res3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_resPool3_batch1')
             .relu(name='HG8_resPool3_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG8_resPool3_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_resPool3_batch2')
             .relu(name='HG8_resPool3_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG8_resPool3_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_resPool3_batch3')
             .relu(name='HG8_resPool3_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_resPool3_conv3'))

        (self.feed('HG8_res3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_resPool3_skip'))

        (self.feed('HG8_res3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_resPool3_batch4')
             .relu(name='HG8_resPool3_relu4')
             .max_pool(2, 2, 2, 2, name='HG8_resPool3_pool')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='HG8_resPool3_conv4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_resPool3_batch5')
             .relu(name='HG8_resPool3_relu5')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='HG8_resPool3_conv5')
             .upsample(32, 32, name='HG8_resPool3_upSample'))


        (self.feed('HG8_resPool3_conv3',
                   'HG8_resPool3_skip',
                   'HG8_resPool3_upSample')
         .add(name='HG8_resPool3'))




# pool2
        (self.feed('HG8_res2')
             .max_pool(2, 2, 2, 2, name='HG8_pool2'))


# res4
        (self.feed('HG8_pool2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res4_batch1')
             .relu(name='HG8_res4_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG8_res4_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res4_batch2')
             .relu(name='HG8_res4_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG8_res4_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res4_batch3')
             .relu(name='HG8_res4_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res4_conv3'))

        (self.feed('HG8_pool2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res4_skip'))

        (self.feed('HG8_res4_conv3',
                   'HG8_res4_skip')
         .add(name='HG8_res4'))
# id:013 max-pooling
        # (self.feed('HG8_pool3')
        #      .max_pool(2, 2, 2, 2, name='HG8_pool4'))


# res5
        (self.feed('HG8_res4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res5_batch1')
             .relu(name='HG8_res5_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG8_res5_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res5_batch2')
             .relu(name='HG8_res5_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG8_res5_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res5_batch3')
             .relu(name='HG8_res5_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res5_conv3'))

        (self.feed('HG8_res4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res5_skip'))

        (self.feed('HG8_res5_conv3',
                   'HG8_res5_skip')
         .add(name='HG8_res5'))


# pool3
        (self.feed('HG8_res4')
             .max_pool(2, 2, 2, 2, name='HG8_pool3'))


# res6
        (self.feed('HG8_pool3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res6_batch1')
             .relu(name='HG8_res6_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG8_res6_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res6_batch2')
             .relu(name='HG8_res6_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG8_res6_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res6_batch3')
             .relu(name='HG8_res6_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res6_conv3'))

        (self.feed('HG8_pool3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res6_skip'))

        (self.feed('HG8_res6_conv3',
                   'HG8_res6_skip')
         .add(name='HG8_res6'))

# res7
        (self.feed('HG8_res6')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res7_batch1')
             .relu(name='HG8_res7_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG8_res7_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res7_batch2')
             .relu(name='HG8_res7_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG8_res7_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res7_batch3')
             .relu(name='HG8_res7_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res7_conv3'))

        (self.feed('HG8_res6')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res7_skip'))

        (self.feed('HG8_res7_conv3',
                   'HG8_res7_skip')
         .add(name='HG8_res7'))


# pool4
        (self.feed('HG8_res6')
             .max_pool(2, 2, 2, 2, name='HG8_pool4'))

# res8
        (self.feed('HG8_pool4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res8_batch1')
             .relu(name='HG8_res8_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG8_res8_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res8_batch2')
             .relu(name='HG8_res8_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG8_res8_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res8_batch3')
             .relu(name='HG8_res8_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res8_conv3'))

        (self.feed('HG8_pool4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res8_skip'))

        (self.feed('HG8_res8_conv3',
                   'HG8_res8_skip')
         .add(name='HG8_res8'))

# res9
        (self.feed('HG8_res8')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res9_batch1')
             .relu(name='HG8_res9_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG8_res9_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res9_batch2')
             .relu(name='HG8_res9_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG8_res9_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res9_batch3')
             .relu(name='HG8_res9_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res9_conv3'))

        (self.feed('HG8_res8')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res9_skip'))

        (self.feed('HG8_res9_conv3',
                   'HG8_res9_skip')
         .add(name='HG8_res9'))

# res10
        (self.feed('HG8_res9')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res10_batch1')
             .relu(name='HG8_res10_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG8_res10_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res10_batch2')
             .relu(name='HG8_res10_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG8_res10_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res10_batch3')
             .relu(name='HG8_res10_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res10_conv3'))

        (self.feed('HG8_res9')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res10_skip'))

        (self.feed('HG8_res10_conv3',
                   'HG8_res10_skip')
         .add(name='HG8_res10'))


# upsample1
        (self.feed('HG8_res10')
             .upsample(8, 8, name='HG8_upSample1'))

# upsample1 + up1(Hg_res7)
        (self.feed('HG8_upSample1',
                   'HG8_res7')
         .add(name='HG8_add1'))


# res11
        (self.feed('HG8_add1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res11_batch1')
             .relu(name='HG8_res11_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG8_res11_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res11_batch2')
             .relu(name='HG8_res11_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG8_res11_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res11_batch3')
             .relu(name='HG8_res11_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res11_conv3'))

        (self.feed('HG8_add1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res11_skip'))

        (self.feed('HG8_res11_conv3',
                   'HG8_res11_skip')
         .add(name='HG8_res11'))


# upsample2
        (self.feed('HG8_res11')
             .upsample(16, 16, name='HG8_upSample2'))

# upsample2 + up1(Hg_res5)
        (self.feed('HG8_upSample2',
                   'HG8_res5')
         .add(name='HG8_add2'))


# res12
        (self.feed('HG8_add2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res12_batch1')
             .relu(name='HG8_res12_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG8_res12_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res12_batch2')
             .relu(name='HG8_res12_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG8_res12_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res12_batch3')
             .relu(name='HG8_res12_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res12_conv3'))

        (self.feed('HG8_add2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res12_skip'))

        (self.feed('HG8_res12_conv3',
                   'HG8_res12_skip')
         .add(name='HG8_res12'))


# upsample3
        (self.feed('HG8_res12')
             .upsample(32, 32, name='HG8_upSample3'))

# upsample3 + HG8_resPool3
        (self.feed('HG8_upSample3',
                   'HG8_resPool3')
         .add(name='HG8_add3'))


# res13
        (self.feed('HG8_add3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res13_batch1')
             .relu(name='HG8_res13_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG8_res13_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res13_batch2')
             .relu(name='HG8_res13_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG8_res13_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res13_batch3')
             .relu(name='HG8_res13_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res13_conv3'))

        (self.feed('HG8_add3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res13_skip'))

        (self.feed('HG8_res13_conv3',
                   'HG8_res13_skip')
         .add(name='HG8_res13'))


# upsample4
        (self.feed('HG8_res13')
             .upsample(64, 64, name='HG8_upSample4'))

# upsample4 + HG8_resPool2
        (self.feed('HG8_upSample4',
                   'HG8_resPool2')
         .add(name='HG8_add4'))

###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  Hourglass  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

################################# HG8 postprocess #################
# Linear layer to produce first set of predictions
# ll1
        (self.feed('HG8_add4')
             .conv(1, 1, 512, 1, 1, biased=True, relu=False, name='HG8_linearfunc1_conv1', padding='SAME')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_linearfunc1_batch1')
             .relu(name='HG8_linearfunc1_relu'))
# ll2
        (self.feed('HG8_linearfunc1_relu')
             .conv(1, 1, 512, 1, 1, biased=True, relu=False, name='HG8_linearfunc2_conv1', padding='SAME')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_linearfunc2_batch1')
             .relu(name='HG8_linearfunc2_relu'))

# att itersize=3

# att U input: ll2
        (self.feed('HG8_linearfunc2_relu')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG8_att_U', padding='SAME'))
# att i=1 conv  C(1)
        with tf.variable_scope('HG8_att_share_param', reuse=False):
            (self.feed('HG8_att_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_att_spConv1', padding='SAME'))
# att Qtemp
        (self.feed('HG8_att_spConv1',
                   'HG8_att_U')
         .add(name='HG8_att_Qtemp1_add')
         .sigmoid(name='HG8_att_Qtemp1'))

# att i=2 conv  C(2)
        with tf.variable_scope('HG8_att_share_param', reuse=True):
            (self.feed('HG8_att_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_att_spConv1', padding='SAME'))
# att Qtemp2
        (self.feed('HG8_att_spConv1',
                   'HG8_att_U')
         .add(name='HG8_att_Qtemp2_add')
         .sigmoid(name='HG8_att_Qtemp2'))

# att i=3 conv  C(3)
        with tf.variable_scope('HG8_att_share_param', reuse=True):
            (self.feed('HG8_att_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_att_spConv1', padding='SAME'))
# att Qtemp
        (self.feed('HG8_att_spConv1',
                   'HG8_att_U')
         .add(name='HG8_att_Qtemp3_add')
         .sigmoid(name='HG8_att_Qtemp3'))
# att pfeat
        (self.feed('HG8_att_Qtemp3')
         .replicate(512, 3, name='HG8_att_pfeat_replicate'))

        (self.feed('HG8_linearfunc2_relu',
                   'HG8_att_pfeat_replicate')
         .multiply2(name='HG8_att_pfeat_multiply'))

# tmpOut 
        # (self.feed('HG8_att_pfeat_multiply')
        #      .conv(1, 1, n_classes, 1, 1, biased=True, relu=False, name='HG8_att_Heatmap', padding='SAME'))










# # tmpOut att U input: HG8_att_pfeat_multiply
#         (self.feed('HG8_att_pfeat_multiply')
#              .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut_U', padding='SAME'))
# # tmpOut att i=1 conv  C(1)
#         with tf.variable_scope('HG8_tmpOut_share_param', reuse=False):
#             (self.feed('HG8_tmpOut_U')
#                  .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut_spConv1', padding='SAME'))
# # tmpOut att Qtemp
#         (self.feed('HG8_tmpOut_spConv1',
#                    'HG8_tmpOut_U')
#          .add(name='HG8_tmpOut_Qtemp1_add')
#          .sigmoid(name='HG8_tmpOut_Qtemp1'))

# # tmpOut att i=2 conv  C(2)
#         with tf.variable_scope('HG8_tmpOut_share_param', reuse=True):
#             (self.feed('HG8_tmpOut_Qtemp1')
#                  .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut_spConv1', padding='SAME'))
# # tmpOut att Qtemp2
#         (self.feed('HG8_tmpOut_spConv1',
#                    'HG8_tmpOut_U')
#          .add(name='HG8_tmpOut_Qtemp2_add')
#          .sigmoid(name='HG8_tmpOut_Qtemp2'))

# # tmpOut att i=3 conv  C(3)
#         with tf.variable_scope('HG8_tmpOut_share_param', reuse=True):
#             (self.feed('HG8_tmpOut_Qtemp2')
#                  .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut_spConv1', padding='SAME'))
# # tmpOut att Qtemp
#         (self.feed('HG8_tmpOut_spConv1',
#                    'HG8_tmpOut_U')
#          .add(name='HG8_tmpOut_Qtemp3_add')
#          .sigmoid(name='HG8_tmpOut_Qtemp3'))
# # tmpOut att pfeat
#         (self.feed('HG8_tmpOut_Qtemp3')
#          .replicate(512, 3, name='HG8_tmpOut_pfeat_replicate'))

#         (self.feed('HG8_att_pfeat_multiply',
#                    'HG8_tmpOut_pfeat_replicate')
#          .multiply2(name='HG8_tmpOut_pfeat_multiply'))

#         (self.feed('HG8_tmpOut_pfeat_multiply')
#              .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut_s', padding='SAME'))

#         (self.feed('HG8_tmpOut_s')
#          .replicate(16, 3, name='HG8_Heatmap'))        








# attention map for part1
# tmpOut1 att U input: HG8_att_pfeat_multiply
        (self.feed('HG8_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut1_U', padding='SAME'))
# tmpOut1 att i=1 conv  C(1)
        with tf.variable_scope('HG8_tmpOut1_share_param', reuse=False):
            (self.feed('HG8_tmpOut1_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut1_spConv1', padding='SAME'))
# tmpOut1 att Qtemp
        (self.feed('HG8_tmpOut1_spConv1',
                   'HG8_tmpOut1_U')
         .add(name='HG8_tmpOut1_Qtemp1_add')
         .sigmoid(name='HG8_tmpOut1_Qtemp1'))

# tmpOut1 att i=2 conv  C(2)
        with tf.variable_scope('HG8_tmpOut1_share_param', reuse=True):
            (self.feed('HG8_tmpOut1_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut1_spConv1', padding='SAME'))
# tmpOut1 att Qtemp2
        (self.feed('HG8_tmpOut1_spConv1',
                   'HG8_tmpOut1_U')
         .add(name='HG8_tmpOut1_Qtemp2_add')
         .sigmoid(name='HG8_tmpOut1_Qtemp2'))

# tmpOut1 att i=3 conv  C(3)
        with tf.variable_scope('HG8_tmpOut1_share_param', reuse=True):
            (self.feed('HG8_tmpOut1_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut1_spConv1', padding='SAME'))
# tmpOut1 att Qtemp
        (self.feed('HG8_tmpOut1_spConv1',
                   'HG8_tmpOut1_U')
         .add(name='HG8_tmpOut1_Qtemp3_add')
         .sigmoid(name='HG8_tmpOut1_Qtemp3'))
# tmpOut1 att pfeat
        (self.feed('HG8_tmpOut1_Qtemp3')
         .replicate(512, 3, name='HG8_tmpOut1_pfeat_replicate'))

        (self.feed('HG8_att_pfeat_multiply',
                   'HG8_tmpOut1_pfeat_replicate')
         .multiply2(name='HG8_tmpOut1_pfeat_multiply'))

        (self.feed('HG8_tmpOut1_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut1_s', padding='SAME'))




# attention map for part2
# tmpOut2 att U input: HG8_att_pfeat_multiply
        (self.feed('HG8_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut2_U', padding='SAME'))
# tmpOut2 att i=1 conv  C(1)
        with tf.variable_scope('HG8_tmpOut2_share_param', reuse=False):
            (self.feed('HG8_tmpOut2_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut2_spConv1', padding='SAME'))
# tmpOut2 att Qtemp
        (self.feed('HG8_tmpOut2_spConv1',
                   'HG8_tmpOut2_U')
         .add(name='HG8_tmpOut2_Qtemp1_add')
         .sigmoid(name='HG8_tmpOut2_Qtemp1'))

# tmpOut2 att i=2 conv  C(2)
        with tf.variable_scope('HG8_tmpOut2_share_param', reuse=True):
            (self.feed('HG8_tmpOut2_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut2_spConv1', padding='SAME'))
# tmpOut2 att Qtemp2
        (self.feed('HG8_tmpOut2_spConv1',
                   'HG8_tmpOut2_U')
         .add(name='HG8_tmpOut2_Qtemp2_add')
         .sigmoid(name='HG8_tmpOut2_Qtemp2'))

# tmpOut2 att i=3 conv  C(3)
        with tf.variable_scope('HG8_tmpOut2_share_param', reuse=True):
            (self.feed('HG8_tmpOut2_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut2_spConv1', padding='SAME'))
# tmpOut2 att Qtemp
        (self.feed('HG8_tmpOut2_spConv1',
                   'HG8_tmpOut2_U')
         .add(name='HG8_tmpOut2_Qtemp3_add')
         .sigmoid(name='HG8_tmpOut2_Qtemp3'))
# tmpOut2 att pfeat
        (self.feed('HG8_tmpOut2_Qtemp3')
         .replicate(512, 3, name='HG8_tmpOut2_pfeat_replicate'))

        (self.feed('HG8_att_pfeat_multiply',
                   'HG8_tmpOut2_pfeat_replicate')
         .multiply2(name='HG8_tmpOut2_pfeat_multiply'))

        (self.feed('HG8_tmpOut2_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut2_s', padding='SAME'))




# attention map for part3
# tmpOut3 att U input: HG8_att_pfeat_multiply
        (self.feed('HG8_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut3_U', padding='SAME'))
# tmpOut3 att i=1 conv  C(1)
        with tf.variable_scope('HG8_tmpOut3_share_param', reuse=False):
            (self.feed('HG8_tmpOut3_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut3_spConv1', padding='SAME'))
# tmpOut3 att Qtemp
        (self.feed('HG8_tmpOut3_spConv1',
                   'HG8_tmpOut3_U')
         .add(name='HG8_tmpOut3_Qtemp1_add')
         .sigmoid(name='HG8_tmpOut3_Qtemp1'))

# tmpOut3 att i=2 conv  C(2)
        with tf.variable_scope('HG8_tmpOut3_share_param', reuse=True):
            (self.feed('HG8_tmpOut3_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut3_spConv1', padding='SAME'))
# tmpOut3 att Qtemp2
        (self.feed('HG8_tmpOut3_spConv1',
                   'HG8_tmpOut3_U')
         .add(name='HG8_tmpOut3_Qtemp2_add')
         .sigmoid(name='HG8_tmpOut3_Qtemp2'))

# tmpOut3 att i=3 conv  C(3)
        with tf.variable_scope('HG8_tmpOut3_share_param', reuse=True):
            (self.feed('HG8_tmpOut3_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut3_spConv1', padding='SAME'))
# tmpOut3 att Qtemp
        (self.feed('HG8_tmpOut3_spConv1',
                   'HG8_tmpOut3_U')
         .add(name='HG8_tmpOut3_Qtemp3_add')
         .sigmoid(name='HG8_tmpOut3_Qtemp3'))
# tmpOut3 att pfeat
        (self.feed('HG8_tmpOut3_Qtemp3')
         .replicate(512, 3, name='HG8_tmpOut3_pfeat_replicate'))

        (self.feed('HG8_att_pfeat_multiply',
                   'HG8_tmpOut3_pfeat_replicate')
         .multiply2(name='HG8_tmpOut3_pfeat_multiply'))

        (self.feed('HG8_tmpOut3_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut3_s', padding='SAME'))






# attention map for part4
# tmpOut4 att U input: HG8_att_pfeat_multiply
        (self.feed('HG8_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut4_U', padding='SAME'))
# tmpOut4 att i=1 conv  C(1)
        with tf.variable_scope('HG8_tmpOut4_share_param', reuse=False):
            (self.feed('HG8_tmpOut4_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut4_spConv1', padding='SAME'))
# tmpOut4 att Qtemp
        (self.feed('HG8_tmpOut4_spConv1',
                   'HG8_tmpOut4_U')
         .add(name='HG8_tmpOut4_Qtemp1_add')
         .sigmoid(name='HG8_tmpOut4_Qtemp1'))

# tmpOut4 att i=2 conv  C(2)
        with tf.variable_scope('HG8_tmpOut4_share_param', reuse=True):
            (self.feed('HG8_tmpOut4_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut4_spConv1', padding='SAME'))
# tmpOut4 att Qtemp2
        (self.feed('HG8_tmpOut4_spConv1',
                   'HG8_tmpOut4_U')
         .add(name='HG8_tmpOut4_Qtemp2_add')
         .sigmoid(name='HG8_tmpOut4_Qtemp2'))

# tmpOut4 att i=3 conv  C(3)
        with tf.variable_scope('HG8_tmpOut4_share_param', reuse=True):
            (self.feed('HG8_tmpOut4_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut4_spConv1', padding='SAME'))
# tmpOut4 att Qtemp
        (self.feed('HG8_tmpOut4_spConv1',
                   'HG8_tmpOut4_U')
         .add(name='HG8_tmpOut4_Qtemp3_add')
         .sigmoid(name='HG8_tmpOut4_Qtemp3'))
# tmpOut4 att pfeat
        (self.feed('HG8_tmpOut4_Qtemp3')
         .replicate(512, 3, name='HG8_tmpOut4_pfeat_replicate'))

        (self.feed('HG8_att_pfeat_multiply',
                   'HG8_tmpOut4_pfeat_replicate')
         .multiply2(name='HG8_tmpOut4_pfeat_multiply'))

        (self.feed('HG8_tmpOut4_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut4_s', padding='SAME'))









# attention map for part5
# tmpOut5 att U input: HG8_att_pfeat_multiply
        (self.feed('HG8_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut5_U', padding='SAME'))
# tmpOut5 att i=1 conv  C(1)
        with tf.variable_scope('HG8_tmpOut5_share_param', reuse=False):
            (self.feed('HG8_tmpOut5_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut5_spConv1', padding='SAME'))
# tmpOut5 att Qtemp
        (self.feed('HG8_tmpOut5_spConv1',
                   'HG8_tmpOut5_U')
         .add(name='HG8_tmpOut5_Qtemp1_add')
         .sigmoid(name='HG8_tmpOut5_Qtemp1'))

# tmpOut5 att i=2 conv  C(2)
        with tf.variable_scope('HG8_tmpOut5_share_param', reuse=True):
            (self.feed('HG8_tmpOut5_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut5_spConv1', padding='SAME'))
# tmpOut5 att Qtemp2
        (self.feed('HG8_tmpOut5_spConv1',
                   'HG8_tmpOut5_U')
         .add(name='HG8_tmpOut5_Qtemp2_add')
         .sigmoid(name='HG8_tmpOut5_Qtemp2'))

# tmpOut5 att i=3 conv  C(3)
        with tf.variable_scope('HG8_tmpOut5_share_param', reuse=True):
            (self.feed('HG8_tmpOut5_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut5_spConv1', padding='SAME'))
# tmpOut5 att Qtemp
        (self.feed('HG8_tmpOut5_spConv1',
                   'HG8_tmpOut5_U')
         .add(name='HG8_tmpOut5_Qtemp3_add')
         .sigmoid(name='HG8_tmpOut5_Qtemp3'))
# tmpOut5 att pfeat
        (self.feed('HG8_tmpOut5_Qtemp3')
         .replicate(512, 3, name='HG8_tmpOut5_pfeat_replicate'))

        (self.feed('HG8_att_pfeat_multiply',
                   'HG8_tmpOut5_pfeat_replicate')
         .multiply2(name='HG8_tmpOut5_pfeat_multiply'))

        (self.feed('HG8_tmpOut5_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut5_s', padding='SAME'))










# attention map for part6
# tmpOut6 att U input: HG8_att_pfeat_multiply
        (self.feed('HG8_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut6_U', padding='SAME'))
# tmpOut6 att i=1 conv  C(1)
        with tf.variable_scope('HG8_tmpOut6_share_param', reuse=False):
            (self.feed('HG8_tmpOut6_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut6_spConv1', padding='SAME'))
# tmpOut6 att Qtemp
        (self.feed('HG8_tmpOut6_spConv1',
                   'HG8_tmpOut6_U')
         .add(name='HG8_tmpOut6_Qtemp1_add')
         .sigmoid(name='HG8_tmpOut6_Qtemp1'))

# tmpOut6 att i=2 conv  C(2)
        with tf.variable_scope('HG8_tmpOut6_share_param', reuse=True):
            (self.feed('HG8_tmpOut6_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut6_spConv1', padding='SAME'))
# tmpOut6 att Qtemp2
        (self.feed('HG8_tmpOut6_spConv1',
                   'HG8_tmpOut6_U')
         .add(name='HG8_tmpOut6_Qtemp2_add')
         .sigmoid(name='HG8_tmpOut6_Qtemp2'))

# tmpOut6 att i=3 conv  C(3)
        with tf.variable_scope('HG8_tmpOut6_share_param', reuse=True):
            (self.feed('HG8_tmpOut6_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut6_spConv1', padding='SAME'))
# tmpOut6 att Qtemp
        (self.feed('HG8_tmpOut6_spConv1',
                   'HG8_tmpOut6_U')
         .add(name='HG8_tmpOut6_Qtemp3_add')
         .sigmoid(name='HG8_tmpOut6_Qtemp3'))
# tmpOut6 att pfeat
        (self.feed('HG8_tmpOut6_Qtemp3')
         .replicate(512, 3, name='HG8_tmpOut6_pfeat_replicate'))

        (self.feed('HG8_att_pfeat_multiply',
                   'HG8_tmpOut6_pfeat_replicate')
         .multiply2(name='HG8_tmpOut6_pfeat_multiply'))

        (self.feed('HG8_tmpOut6_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut6_s', padding='SAME'))





# attention map for part7
# tmpOut7 att U input: HG8_att_pfeat_multiply
        (self.feed('HG8_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut7_U', padding='SAME'))
# tmpOut7 att i=1 conv  C(1)
        with tf.variable_scope('HG8_tmpOut7_share_param', reuse=False):
            (self.feed('HG8_tmpOut7_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut7_spConv1', padding='SAME'))
# tmpOut7 att Qtemp
        (self.feed('HG8_tmpOut7_spConv1',
                   'HG8_tmpOut7_U')
         .add(name='HG8_tmpOut7_Qtemp1_add')
         .sigmoid(name='HG8_tmpOut7_Qtemp1'))

# tmpOut7 att i=2 conv  C(2)
        with tf.variable_scope('HG8_tmpOut7_share_param', reuse=True):
            (self.feed('HG8_tmpOut7_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut7_spConv1', padding='SAME'))
# tmpOut7 att Qtemp2
        (self.feed('HG8_tmpOut7_spConv1',
                   'HG8_tmpOut7_U')
         .add(name='HG8_tmpOut7_Qtemp2_add')
         .sigmoid(name='HG8_tmpOut7_Qtemp2'))

# tmpOut7 att i=3 conv  C(3)
        with tf.variable_scope('HG8_tmpOut7_share_param', reuse=True):
            (self.feed('HG8_tmpOut7_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut7_spConv1', padding='SAME'))
# tmpOut7 att Qtemp
        (self.feed('HG8_tmpOut7_spConv1',
                   'HG8_tmpOut7_U')
         .add(name='HG8_tmpOut7_Qtemp3_add')
         .sigmoid(name='HG8_tmpOut7_Qtemp3'))
# tmpOut7 att pfeat
        (self.feed('HG8_tmpOut7_Qtemp3')
         .replicate(512, 3, name='HG8_tmpOut7_pfeat_replicate'))

        (self.feed('HG8_att_pfeat_multiply',
                   'HG8_tmpOut7_pfeat_replicate')
         .multiply2(name='HG8_tmpOut7_pfeat_multiply'))

        (self.feed('HG8_tmpOut7_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut7_s', padding='SAME'))








# attention map for part8
# tmpOut8 att U input: HG8_att_pfeat_multiply
        (self.feed('HG8_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut8_U', padding='SAME'))
# tmpOut8 att i=1 conv  C(1)
        with tf.variable_scope('HG8_tmpOut8_share_param', reuse=False):
            (self.feed('HG8_tmpOut8_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut8_spConv1', padding='SAME'))
# tmpOut8 att Qtemp
        (self.feed('HG8_tmpOut8_spConv1',
                   'HG8_tmpOut8_U')
         .add(name='HG8_tmpOut8_Qtemp1_add')
         .sigmoid(name='HG8_tmpOut8_Qtemp1'))

# tmpOut8 att i=2 conv  C(2)
        with tf.variable_scope('HG8_tmpOut8_share_param', reuse=True):
            (self.feed('HG8_tmpOut8_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut8_spConv1', padding='SAME'))
# tmpOut8 att Qtemp2
        (self.feed('HG8_tmpOut8_spConv1',
                   'HG8_tmpOut8_U')
         .add(name='HG8_tmpOut8_Qtemp2_add')
         .sigmoid(name='HG8_tmpOut8_Qtemp2'))

# tmpOut8 att i=3 conv  C(3)
        with tf.variable_scope('HG8_tmpOut8_share_param', reuse=True):
            (self.feed('HG8_tmpOut8_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut8_spConv1', padding='SAME'))
# tmpOut8 att Qtemp
        (self.feed('HG8_tmpOut8_spConv1',
                   'HG8_tmpOut8_U')
         .add(name='HG8_tmpOut8_Qtemp3_add')
         .sigmoid(name='HG8_tmpOut8_Qtemp3'))
# tmpOut8 att pfeat
        (self.feed('HG8_tmpOut8_Qtemp3')
         .replicate(512, 3, name='HG8_tmpOut8_pfeat_replicate'))

        (self.feed('HG8_att_pfeat_multiply',
                   'HG8_tmpOut8_pfeat_replicate')
         .multiply2(name='HG8_tmpOut8_pfeat_multiply'))

        (self.feed('HG8_tmpOut8_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut8_s', padding='SAME'))











# attention map for part9
# tmpOut9 att U input: HG8_att_pfeat_multiply
        (self.feed('HG8_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut9_U', padding='SAME'))
# tmpOut9 att i=1 conv  C(1)
        with tf.variable_scope('HG8_tmpOut9_share_param', reuse=False):
            (self.feed('HG8_tmpOut9_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut9_spConv1', padding='SAME'))
# tmpOut9 att Qtemp
        (self.feed('HG8_tmpOut9_spConv1',
                   'HG8_tmpOut9_U')
         .add(name='HG8_tmpOut9_Qtemp1_add')
         .sigmoid(name='HG8_tmpOut9_Qtemp1'))

# tmpOut9 att i=2 conv  C(2)
        with tf.variable_scope('HG8_tmpOut9_share_param', reuse=True):
            (self.feed('HG8_tmpOut9_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut9_spConv1', padding='SAME'))
# tmpOut9 att Qtemp2
        (self.feed('HG8_tmpOut9_spConv1',
                   'HG8_tmpOut9_U')
         .add(name='HG8_tmpOut9_Qtemp2_add')
         .sigmoid(name='HG8_tmpOut9_Qtemp2'))

# tmpOut9 att i=3 conv  C(3)
        with tf.variable_scope('HG8_tmpOut9_share_param', reuse=True):
            (self.feed('HG8_tmpOut9_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut9_spConv1', padding='SAME'))
# tmpOut9 att Qtemp
        (self.feed('HG8_tmpOut9_spConv1',
                   'HG8_tmpOut9_U')
         .add(name='HG8_tmpOut9_Qtemp3_add')
         .sigmoid(name='HG8_tmpOut9_Qtemp3'))
# tmpOut9 att pfeat
        (self.feed('HG8_tmpOut9_Qtemp3')
         .replicate(512, 3, name='HG8_tmpOut9_pfeat_replicate'))

        (self.feed('HG8_att_pfeat_multiply',
                   'HG8_tmpOut9_pfeat_replicate')
         .multiply2(name='HG8_tmpOut9_pfeat_multiply'))

        (self.feed('HG8_tmpOut9_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut9_s', padding='SAME'))












# attention map for part10
# tmpOut10 att U input: HG8_att_pfeat_multiply
        (self.feed('HG8_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut10_U', padding='SAME'))
# tmpOut10 att i=1 conv  C(1)
        with tf.variable_scope('HG8_tmpOut10_share_param', reuse=False):
            (self.feed('HG8_tmpOut10_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut10_spConv1', padding='SAME'))
# tmpOut10 att Qtemp
        (self.feed('HG8_tmpOut10_spConv1',
                   'HG8_tmpOut10_U')
         .add(name='HG8_tmpOut10_Qtemp1_add')
         .sigmoid(name='HG8_tmpOut10_Qtemp1'))

# tmpOut10 att i=2 conv  C(2)
        with tf.variable_scope('HG8_tmpOut10_share_param', reuse=True):
            (self.feed('HG8_tmpOut10_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut10_spConv1', padding='SAME'))
# tmpOut10 att Qtemp2
        (self.feed('HG8_tmpOut10_spConv1',
                   'HG8_tmpOut10_U')
         .add(name='HG8_tmpOut10_Qtemp2_add')
         .sigmoid(name='HG8_tmpOut10_Qtemp2'))

# tmpOut10 att i=3 conv  C(3)
        with tf.variable_scope('HG8_tmpOut10_share_param', reuse=True):
            (self.feed('HG8_tmpOut10_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut10_spConv1', padding='SAME'))
# tmpOut10 att Qtemp
        (self.feed('HG8_tmpOut10_spConv1',
                   'HG8_tmpOut10_U')
         .add(name='HG8_tmpOut10_Qtemp3_add')
         .sigmoid(name='HG8_tmpOut10_Qtemp3'))
# tmpOut10 att pfeat
        (self.feed('HG8_tmpOut10_Qtemp3')
         .replicate(512, 3, name='HG8_tmpOut10_pfeat_replicate'))

        (self.feed('HG8_att_pfeat_multiply',
                   'HG8_tmpOut10_pfeat_replicate')
         .multiply2(name='HG8_tmpOut10_pfeat_multiply'))

        (self.feed('HG8_tmpOut10_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut10_s', padding='SAME'))










# attention map for part11
# tmpOut11 att U input: HG8_att_pfeat_multiply
        (self.feed('HG8_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut11_U', padding='SAME'))
# tmpOut11 att i=1 conv  C(1)
        with tf.variable_scope('HG8_tmpOut11_share_param', reuse=False):
            (self.feed('HG8_tmpOut11_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut11_spConv1', padding='SAME'))
# tmpOut11 att Qtemp
        (self.feed('HG8_tmpOut11_spConv1',
                   'HG8_tmpOut11_U')
         .add(name='HG8_tmpOut11_Qtemp1_add')
         .sigmoid(name='HG8_tmpOut11_Qtemp1'))

# tmpOut11 att i=2 conv  C(2)
        with tf.variable_scope('HG8_tmpOut11_share_param', reuse=True):
            (self.feed('HG8_tmpOut11_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut11_spConv1', padding='SAME'))
# tmpOut11 att Qtemp2
        (self.feed('HG8_tmpOut11_spConv1',
                   'HG8_tmpOut11_U')
         .add(name='HG8_tmpOut11_Qtemp2_add')
         .sigmoid(name='HG8_tmpOut11_Qtemp2'))

# tmpOut11 att i=3 conv  C(3)
        with tf.variable_scope('HG8_tmpOut11_share_param', reuse=True):
            (self.feed('HG8_tmpOut11_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut11_spConv1', padding='SAME'))
# tmpOut11 att Qtemp
        (self.feed('HG8_tmpOut11_spConv1',
                   'HG8_tmpOut11_U')
         .add(name='HG8_tmpOut11_Qtemp3_add')
         .sigmoid(name='HG8_tmpOut11_Qtemp3'))
# tmpOut11 att pfeat
        (self.feed('HG8_tmpOut11_Qtemp3')
         .replicate(512, 3, name='HG8_tmpOut11_pfeat_replicate'))

        (self.feed('HG8_att_pfeat_multiply',
                   'HG8_tmpOut11_pfeat_replicate')
         .multiply2(name='HG8_tmpOut11_pfeat_multiply'))

        (self.feed('HG8_tmpOut11_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut11_s', padding='SAME'))











# attention map for part12
# tmpOut12 att U input: HG8_att_pfeat_multiply
        (self.feed('HG8_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut12_U', padding='SAME'))
# tmpOut12 att i=1 conv  C(1)
        with tf.variable_scope('HG8_tmpOut12_share_param', reuse=False):
            (self.feed('HG8_tmpOut12_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut12_spConv1', padding='SAME'))
# tmpOut12 att Qtemp
        (self.feed('HG8_tmpOut12_spConv1',
                   'HG8_tmpOut12_U')
         .add(name='HG8_tmpOut12_Qtemp1_add')
         .sigmoid(name='HG8_tmpOut12_Qtemp1'))

# tmpOut12 att i=2 conv  C(2)
        with tf.variable_scope('HG8_tmpOut12_share_param', reuse=True):
            (self.feed('HG8_tmpOut12_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut12_spConv1', padding='SAME'))
# tmpOut12 att Qtemp2
        (self.feed('HG8_tmpOut12_spConv1',
                   'HG8_tmpOut12_U')
         .add(name='HG8_tmpOut12_Qtemp2_add')
         .sigmoid(name='HG8_tmpOut12_Qtemp2'))

# tmpOut12 att i=3 conv  C(3)
        with tf.variable_scope('HG8_tmpOut12_share_param', reuse=True):
            (self.feed('HG8_tmpOut12_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut12_spConv1', padding='SAME'))
# tmpOut12 att Qtemp
        (self.feed('HG8_tmpOut12_spConv1',
                   'HG8_tmpOut12_U')
         .add(name='HG8_tmpOut12_Qtemp3_add')
         .sigmoid(name='HG8_tmpOut12_Qtemp3'))
# tmpOut12 att pfeat
        (self.feed('HG8_tmpOut12_Qtemp3')
         .replicate(512, 3, name='HG8_tmpOut12_pfeat_replicate'))

        (self.feed('HG8_att_pfeat_multiply',
                   'HG8_tmpOut12_pfeat_replicate')
         .multiply2(name='HG8_tmpOut12_pfeat_multiply'))

        (self.feed('HG8_tmpOut12_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut12_s', padding='SAME'))










# attention map for part13
# tmpOut13 att U input: HG8_att_pfeat_multiply
        (self.feed('HG8_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut13_U', padding='SAME'))
# tmpOut13 att i=1 conv  C(1)
        with tf.variable_scope('HG8_tmpOut13_share_param', reuse=False):
            (self.feed('HG8_tmpOut13_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut13_spConv1', padding='SAME'))
# tmpOut13 att Qtemp
        (self.feed('HG8_tmpOut13_spConv1',
                   'HG8_tmpOut13_U')
         .add(name='HG8_tmpOut13_Qtemp1_add')
         .sigmoid(name='HG8_tmpOut13_Qtemp1'))

# tmpOut13 att i=2 conv  C(2)
        with tf.variable_scope('HG8_tmpOut13_share_param', reuse=True):
            (self.feed('HG8_tmpOut13_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut13_spConv1', padding='SAME'))
# tmpOut13 att Qtemp2
        (self.feed('HG8_tmpOut13_spConv1',
                   'HG8_tmpOut13_U')
         .add(name='HG8_tmpOut13_Qtemp2_add')
         .sigmoid(name='HG8_tmpOut13_Qtemp2'))

# tmpOut13 att i=3 conv  C(3)
        with tf.variable_scope('HG8_tmpOut13_share_param', reuse=True):
            (self.feed('HG8_tmpOut13_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut13_spConv1', padding='SAME'))
# tmpOut13 att Qtemp
        (self.feed('HG8_tmpOut13_spConv1',
                   'HG8_tmpOut13_U')
         .add(name='HG8_tmpOut13_Qtemp3_add')
         .sigmoid(name='HG8_tmpOut13_Qtemp3'))
# tmpOut13 att pfeat
        (self.feed('HG8_tmpOut13_Qtemp3')
         .replicate(512, 3, name='HG8_tmpOut13_pfeat_replicate'))

        (self.feed('HG8_att_pfeat_multiply',
                   'HG8_tmpOut13_pfeat_replicate')
         .multiply2(name='HG8_tmpOut13_pfeat_multiply'))

        (self.feed('HG8_tmpOut13_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut13_s', padding='SAME'))










# attention map for part14
# tmpOut14 att U input: HG8_att_pfeat_multiply
        (self.feed('HG8_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut14_U', padding='SAME'))
# tmpOut14 att i=1 conv  C(1)
        with tf.variable_scope('HG8_tmpOut14_share_param', reuse=False):
            (self.feed('HG8_tmpOut14_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut14_spConv1', padding='SAME'))
# tmpOut14 att Qtemp
        (self.feed('HG8_tmpOut14_spConv1',
                   'HG8_tmpOut14_U')
         .add(name='HG8_tmpOut14_Qtemp1_add')
         .sigmoid(name='HG8_tmpOut14_Qtemp1'))

# tmpOut14 att i=2 conv  C(2)
        with tf.variable_scope('HG8_tmpOut14_share_param', reuse=True):
            (self.feed('HG8_tmpOut14_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut14_spConv1', padding='SAME'))
# tmpOut14 att Qtemp2
        (self.feed('HG8_tmpOut14_spConv1',
                   'HG8_tmpOut14_U')
         .add(name='HG8_tmpOut14_Qtemp2_add')
         .sigmoid(name='HG8_tmpOut14_Qtemp2'))

# tmpOut14 att i=3 conv  C(3)
        with tf.variable_scope('HG8_tmpOut14_share_param', reuse=True):
            (self.feed('HG8_tmpOut14_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut14_spConv1', padding='SAME'))
# tmpOut14 att Qtemp
        (self.feed('HG8_tmpOut14_spConv1',
                   'HG8_tmpOut14_U')
         .add(name='HG8_tmpOut14_Qtemp3_add')
         .sigmoid(name='HG8_tmpOut14_Qtemp3'))
# tmpOut14 att pfeat
        (self.feed('HG8_tmpOut14_Qtemp3')
         .replicate(512, 3, name='HG8_tmpOut14_pfeat_replicate'))

        (self.feed('HG8_att_pfeat_multiply',
                   'HG8_tmpOut14_pfeat_replicate')
         .multiply2(name='HG8_tmpOut14_pfeat_multiply'))

        (self.feed('HG8_tmpOut14_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut14_s', padding='SAME'))










# attention map for part15
# tmpOut15 att U input: HG8_att_pfeat_multiply
        (self.feed('HG8_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut15_U', padding='SAME'))
# tmpOut15 att i=1 conv  C(1)
        with tf.variable_scope('HG8_tmpOut15_share_param', reuse=False):
            (self.feed('HG8_tmpOut15_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut15_spConv1', padding='SAME'))
# tmpOut15 att Qtemp
        (self.feed('HG8_tmpOut15_spConv1',
                   'HG8_tmpOut15_U')
         .add(name='HG8_tmpOut15_Qtemp1_add')
         .sigmoid(name='HG8_tmpOut15_Qtemp1'))

# tmpOut15 att i=2 conv  C(2)
        with tf.variable_scope('HG8_tmpOut15_share_param', reuse=True):
            (self.feed('HG8_tmpOut15_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut15_spConv1', padding='SAME'))
# tmpOut15 att Qtemp2
        (self.feed('HG8_tmpOut15_spConv1',
                   'HG8_tmpOut15_U')
         .add(name='HG8_tmpOut15_Qtemp2_add')
         .sigmoid(name='HG8_tmpOut15_Qtemp2'))

# tmpOut15 att i=3 conv  C(3)
        with tf.variable_scope('HG8_tmpOut15_share_param', reuse=True):
            (self.feed('HG8_tmpOut15_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut15_spConv1', padding='SAME'))
# tmpOut15 att Qtemp
        (self.feed('HG8_tmpOut15_spConv1',
                   'HG8_tmpOut15_U')
         .add(name='HG8_tmpOut15_Qtemp3_add')
         .sigmoid(name='HG8_tmpOut15_Qtemp3'))
# tmpOut15 att pfeat
        (self.feed('HG8_tmpOut15_Qtemp3')
         .replicate(512, 3, name='HG8_tmpOut15_pfeat_replicate'))

        (self.feed('HG8_att_pfeat_multiply',
                   'HG8_tmpOut15_pfeat_replicate')
         .multiply2(name='HG8_tmpOut15_pfeat_multiply'))

        (self.feed('HG8_tmpOut15_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut15_s', padding='SAME'))










# attention map for part16
# tmpOut16 att U input: HG8_att_pfeat_multiply
        (self.feed('HG8_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut16_U', padding='SAME'))
# tmpOut16 att i=1 conv  C(1)
        with tf.variable_scope('HG8_tmpOut16_share_param', reuse=False):
            (self.feed('HG8_tmpOut16_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut16_spConv1', padding='SAME'))
# tmpOut16 att Qtemp
        (self.feed('HG8_tmpOut16_spConv1',
                   'HG8_tmpOut16_U')
         .add(name='HG8_tmpOut16_Qtemp1_add')
         .sigmoid(name='HG8_tmpOut16_Qtemp1'))

# tmpOut16 att i=2 conv  C(2)
        with tf.variable_scope('HG8_tmpOut16_share_param', reuse=True):
            (self.feed('HG8_tmpOut16_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut16_spConv1', padding='SAME'))
# tmpOut16 att Qtemp2
        (self.feed('HG8_tmpOut16_spConv1',
                   'HG8_tmpOut16_U')
         .add(name='HG8_tmpOut16_Qtemp2_add')
         .sigmoid(name='HG8_tmpOut16_Qtemp2'))

# tmpOut16 att i=3 conv  C(3)
        with tf.variable_scope('HG8_tmpOut16_share_param', reuse=True):
            (self.feed('HG8_tmpOut16_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut16_spConv1', padding='SAME'))
# tmpOut16 att Qtemp
        (self.feed('HG8_tmpOut16_spConv1',
                   'HG8_tmpOut16_U')
         .add(name='HG8_tmpOut16_Qtemp3_add')
         .sigmoid(name='HG8_tmpOut16_Qtemp3'))
# tmpOut16 att pfeat
        (self.feed('HG8_tmpOut16_Qtemp3')
         .replicate(512, 3, name='HG8_tmpOut16_pfeat_replicate'))

        (self.feed('HG8_att_pfeat_multiply',
                   'HG8_tmpOut16_pfeat_replicate')
         .multiply2(name='HG8_tmpOut16_pfeat_multiply'))

        (self.feed('HG8_tmpOut16_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='HG8_tmpOut16_s', padding='SAME'))


        (self.feed('HG8_tmpOut1_s',
                   'HG8_tmpOut2_s',
                   'HG8_tmpOut3_s',
                   'HG8_tmpOut4_s',
                   'HG8_tmpOut5_s',
                   'HG8_tmpOut6_s',
                   'HG8_tmpOut7_s',
                   'HG8_tmpOut8_s',
                   'HG8_tmpOut9_s',
                   'HG8_tmpOut10_s',
                   'HG8_tmpOut11_s',
                   'HG8_tmpOut12_s',
                   'HG8_tmpOut13_s',
                   'HG8_tmpOut14_s',
                   'HG8_tmpOut15_s',
                   'HG8_tmpOut16_s')
         .stack( axis = 3, name='HG8_Heatmap'))        




###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ HG8 postprocess ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

# ############################# generate input for next HG ##########

# # outmap
#         (self.feed('HG8_Heatmap')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_outmap', padding='SAME'))
# # ll3
#         (self.feed('HG8_linearfunc1_relu')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_linearfunc3_conv1', padding='SAME')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_linearfunc3_batch1')
#              .relu(name='HG8_linearfunc3_relu'))
# # tmointer
#         (self.feed('HG8_input',
#                    'HG8_outmap',
#                    'HG8_linearfunc3_relu')
#          .add(name='HG8_input'))

# ###^^^^^^^^^^^^^^^^^^^^^^^^^^^ generate input for next HG ^^^^^^^^^^^^^^^^^^^^^^^^^^^^###


















































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































# #######################################  Hourglass2  #####################
# # res1
#         (self.feed('HG2_input')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res1_batch1')
#              .relu(name='HG2_res1_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG2_res1_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res1_batch2')
#              .relu(name='HG2_res1_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG2_res1_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res1_batch3')
#              .relu(name='HG2_res1_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res1_conv3'))

#         (self.feed('HG2_input')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res1_skip'))

#         (self.feed('HG2_res1_conv3',
#                    'HG2_res1_skip')
#          .add(name='HG2_res1'))

# #   pool1
#         (self.feed('HG2_input')
#              .max_pool(2, 2, 2, 2, name='HG2_pool1'))


# # res2
#         (self.feed('HG2_pool1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res2_batch1')
#              .relu(name='HG2_res2_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG2_res2_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res2_batch2')
#              .relu(name='HG2_res2_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG2_res2_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res2_batch3')
#              .relu(name='HG2_res2_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res2_conv3'))

#         (self.feed('HG2_pool1')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res2_skip'))

#         (self.feed('HG2_res2_conv3',
#                    'HG2_res2_skip')
#          .add(name='HG2_res2'))
# # id:009 max-pooling
#         # (self.feed('HG2_pool1')
#         #      .max_pool(2, 2, 2, 2, name='HG2_pool2'))


# # res3
#         (self.feed('HG2_res2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res3_batch1')
#              .relu(name='HG2_res3_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG2_res3_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res3_batch2')
#              .relu(name='HG2_res3_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG2_res3_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res3_batch3')
#              .relu(name='HG2_res3_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res3_conv3'))

#         (self.feed('HG2_res2')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res3_skip'))

#         (self.feed('HG2_res3_conv3',
#                    'HG2_res3_skip')
#          .add(name='HG2_res3'))


# # pool2
#         (self.feed('HG2_res2')
#              .max_pool(2, 2, 2, 2, name='HG2_pool2'))


# # res4
#         (self.feed('HG2_pool2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res4_batch1')
#              .relu(name='HG2_res4_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG2_res4_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res4_batch2')
#              .relu(name='HG2_res4_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG2_res4_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res4_batch3')
#              .relu(name='HG2_res4_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res4_conv3'))

#         (self.feed('HG2_pool2')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res4_skip'))

#         (self.feed('HG2_res4_conv3',
#                    'HG2_res4_skip')
#          .add(name='HG2_res4'))
# # id:013 max-pooling
#         # (self.feed('HG2_pool3')
#         #      .max_pool(2, 2, 2, 2, name='HG2_pool4'))


# # res5
#         (self.feed('HG2_res4')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res5_batch1')
#              .relu(name='HG2_res5_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG2_res5_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res5_batch2')
#              .relu(name='HG2_res5_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG2_res5_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res5_batch3')
#              .relu(name='HG2_res5_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res5_conv3'))

#         (self.feed('HG2_res4')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res5_skip'))

#         (self.feed('HG2_res5_conv3',
#                    'HG2_res5_skip')
#          .add(name='HG2_res5'))


# # pool3
#         (self.feed('HG2_res4')
#              .max_pool(2, 2, 2, 2, name='HG2_pool3'))


# # res6
#         (self.feed('HG2_pool3')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res6_batch1')
#              .relu(name='HG2_res6_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG2_res6_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res6_batch2')
#              .relu(name='HG2_res6_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG2_res6_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res6_batch3')
#              .relu(name='HG2_res6_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res6_conv3'))

#         (self.feed('HG2_pool3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res6_skip'))

#         (self.feed('HG2_res6_conv3',
#                    'HG2_res6_skip')
#          .add(name='HG2_res6'))

# # res7
#         (self.feed('HG2_res6')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res7_batch1')
#              .relu(name='HG2_res7_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG2_res7_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res7_batch2')
#              .relu(name='HG2_res7_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG2_res7_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res7_batch3')
#              .relu(name='HG2_res7_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res7_conv3'))

#         (self.feed('HG2_res6')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res7_skip'))

#         (self.feed('HG2_res7_conv3',
#                    'HG2_res7_skip')
#          .add(name='HG2_res7'))


# # pool4
#         (self.feed('HG2_res6')
#              .max_pool(2, 2, 2, 2, name='HG2_pool4'))

# # res8
#         (self.feed('HG2_pool4')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res8_batch1')
#              .relu(name='HG2_res8_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG2_res8_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res8_batch2')
#              .relu(name='HG2_res8_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG2_res8_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res8_batch3')
#              .relu(name='HG2_res8_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res8_conv3'))

#         (self.feed('HG2_pool4')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res8_skip'))

#         (self.feed('HG2_res8_conv3',
#                    'HG2_res8_skip')
#          .add(name='HG2_res8'))

# # res9
#         (self.feed('HG2_res8')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res9_batch1')
#              .relu(name='HG2_res9_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG2_res9_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res9_batch2')
#              .relu(name='HG2_res9_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG2_res9_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res9_batch3')
#              .relu(name='HG2_res9_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res9_conv3'))

#         (self.feed('HG2_res8')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res9_skip'))

#         (self.feed('HG2_res9_conv3',
#                    'HG2_res9_skip')
#          .add(name='HG2_res9'))

# # res10
#         (self.feed('HG2_res9')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res10_batch1')
#              .relu(name='HG2_res10_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG2_res10_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res10_batch2')
#              .relu(name='HG2_res10_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG2_res10_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res10_batch3')
#              .relu(name='HG2_res10_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res10_conv3'))

#         (self.feed('HG2_res9')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res10_skip'))

#         (self.feed('HG2_res10_conv3',
#                    'HG2_res10_skip')
#          .add(name='HG2_res10'))


# # upsample1
#         (self.feed('HG2_res10')
#              .upsample(8, 8, name='HG2_upSample1'))

# # upsample1 + up1(Hg_res7)
#         (self.feed('HG2_upSample1',
#                    'HG2_res7')
#          .add(name='HG2_add1'))


# # res11
#         (self.feed('HG2_add1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res11_batch1')
#              .relu(name='HG2_res11_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG2_res11_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res11_batch2')
#              .relu(name='HG2_res11_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG2_res11_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res11_batch3')
#              .relu(name='HG2_res11_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res11_conv3'))

#         (self.feed('HG2_add1')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res11_skip'))

#         (self.feed('HG2_res11_conv3',
#                    'HG2_res11_skip')
#          .add(name='HG2_res11'))


# # upsample2
#         (self.feed('HG2_res11')
#              .upsample(16, 16, name='HG2_upSample2'))

# # upsample2 + up1(Hg_res5)
#         (self.feed('HG2_upSample2',
#                    'HG2_res5')
#          .add(name='HG2_add2'))


# # res12
#         (self.feed('HG2_add2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res12_batch1')
#              .relu(name='HG2_res12_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG2_res12_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res12_batch2')
#              .relu(name='HG2_res12_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG2_res12_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res12_batch3')
#              .relu(name='HG2_res12_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res12_conv3'))

#         (self.feed('HG2_add2')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res12_skip'))

#         (self.feed('HG2_res12_conv3',
#                    'HG2_res12_skip')
#          .add(name='HG2_res12'))


# # upsample3
#         (self.feed('HG2_res12')
#              .upsample(32, 32, name='HG2_upSample3'))

# # upsample3 + up1(Hg_res3)
#         (self.feed('HG2_upSample3',
#                    'HG2_res3')
#          .add(name='HG2_add3'))


# # res13
#         (self.feed('HG2_add3')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res13_batch1')
#              .relu(name='HG2_res13_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG2_res13_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res13_batch2')
#              .relu(name='HG2_res13_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG2_res13_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res13_batch3')
#              .relu(name='HG2_res13_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res13_conv3'))

#         (self.feed('HG2_add3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res13_skip'))

#         (self.feed('HG2_res13_conv3',
#                    'HG2_res13_skip')
#          .add(name='HG2_res13'))


# # upsample4
#         (self.feed('HG2_res13')
#              .upsample(64, 64, name='HG2_upSample4'))

# # upsample4 + up1(Hg_res1)
#         (self.feed('HG2_upSample4',
#                    'HG2_res1')
#          .add(name='HG2_add4'))

# ###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  Hourglass2  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

# ################################# Hourglass2 postprocess #################

# # id:025  Res14
#         (self.feed('HG2_add4')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res14_batch1')
#              .relu(name='HG2_res14_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG2_res14_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res14_batch2')
#              .relu(name='HG2_res14_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG2_res14_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res14_batch3')
#              .relu(name='HG2_res14_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res14_conv3')

#          )

#         (self.feed('HG2_add4')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res14_skip'))

#         (self.feed('HG2_res14_conv3',
#                    'HG2_res14_skip')
#          .add(name='HG2_res14'))

# # Linear layer to produce first set of predictions
# # ll
#         (self.feed('HG2_res14')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_linearfunc_conv1', padding='SAME')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_linearfunc_batch1')
#              .relu(name='HG2_linearfunc_relu'))

# #************************************* Predicted heatmaps tmpOut ****************************#
# # tmpOut
#         (self.feed('HG2_linearfunc_relu')
#              .conv(1, 1, n_classes, 1, 1, biased=True, relu=False, name='HG2_Heatmap', padding='SAME'))

# ###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Hourglass2 postprocess ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

# ############################# generate input for next Hourglass ##########

# # ll_
#         (self.feed('HG2_linearfunc_relu')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_ll_', padding='SAME'))
# # tmpOut_
#         (self.feed('HG2_Heatmap')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_tmpOut_', padding='SAME'))
# # inter
#         (self.feed('HG2_input',
#                    'HG2_ll_',
#                    'HG2_tmpOut_')
#          .add(name='HG3_input'))

# ###^^^^^^^^^^^^^^^^^^^^^^^^^^^ generate input for next Hourglass ^^^^^^^^^^^^^^^^^^^^^^^^^^^^###


# #######################################  Hourglass3  #####################
# # res1
#         (self.feed('HG3_input')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res1_batch1')
#              .relu(name='HG3_res1_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG3_res1_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res1_batch2')
#              .relu(name='HG3_res1_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG3_res1_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res1_batch3')
#              .relu(name='HG3_res1_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res1_conv3'))

#         (self.feed('HG3_input')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res1_skip'))

#         (self.feed('HG3_res1_conv3',
#                    'HG3_res1_skip')
#          .add(name='HG3_res1'))

# #   pool1
#         (self.feed('HG3_input')
#              .max_pool(2, 2, 2, 2, name='HG3_pool1'))


# # res2
#         (self.feed('HG3_pool1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res2_batch1')
#              .relu(name='HG3_res2_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG3_res2_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res2_batch2')
#              .relu(name='HG3_res2_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG3_res2_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res2_batch3')
#              .relu(name='HG3_res2_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res2_conv3'))

#         (self.feed('HG3_pool1')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res2_skip'))

#         (self.feed('HG3_res2_conv3',
#                    'HG3_res2_skip')
#          .add(name='HG3_res2'))
# # id:009 max-pooling
#         # (self.feed('HG3_pool1')
#         #      .max_pool(2, 2, 2, 2, name='HG3_pool2'))


# # res3
#         (self.feed('HG3_res2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res3_batch1')
#              .relu(name='HG3_res3_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG3_res3_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res3_batch2')
#              .relu(name='HG3_res3_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG3_res3_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res3_batch3')
#              .relu(name='HG3_res3_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res3_conv3'))

#         (self.feed('HG3_res2')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res3_skip'))

#         (self.feed('HG3_res3_conv3',
#                    'HG3_res3_skip')
#          .add(name='HG3_res3'))


# # pool2
#         (self.feed('HG3_res2')
#              .max_pool(2, 2, 2, 2, name='HG3_pool2'))


# # res4
#         (self.feed('HG3_pool2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res4_batch1')
#              .relu(name='HG3_res4_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG3_res4_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res4_batch2')
#              .relu(name='HG3_res4_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG3_res4_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res4_batch3')
#              .relu(name='HG3_res4_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res4_conv3'))

#         (self.feed('HG3_pool2')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res4_skip'))

#         (self.feed('HG3_res4_conv3',
#                    'HG3_res4_skip')
#          .add(name='HG3_res4'))
# # id:013 max-pooling
#         # (self.feed('HG3_pool3')
#         #      .max_pool(2, 2, 2, 2, name='HG3_pool4'))


# # res5
#         (self.feed('HG3_res4')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res5_batch1')
#              .relu(name='HG3_res5_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG3_res5_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res5_batch2')
#              .relu(name='HG3_res5_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG3_res5_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res5_batch3')
#              .relu(name='HG3_res5_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res5_conv3'))

#         (self.feed('HG3_res4')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res5_skip'))

#         (self.feed('HG3_res5_conv3',
#                    'HG3_res5_skip')
#          .add(name='HG3_res5'))


# # pool3
#         (self.feed('HG3_res4')
#              .max_pool(2, 2, 2, 2, name='HG3_pool3'))


# # res6
#         (self.feed('HG3_pool3')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res6_batch1')
#              .relu(name='HG3_res6_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG3_res6_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res6_batch2')
#              .relu(name='HG3_res6_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG3_res6_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res6_batch3')
#              .relu(name='HG3_res6_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res6_conv3'))

#         (self.feed('HG3_pool3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res6_skip'))

#         (self.feed('HG3_res6_conv3',
#                    'HG3_res6_skip')
#          .add(name='HG3_res6'))

# # res7
#         (self.feed('HG3_res6')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res7_batch1')
#              .relu(name='HG3_res7_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG3_res7_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res7_batch2')
#              .relu(name='HG3_res7_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG3_res7_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res7_batch3')
#              .relu(name='HG3_res7_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res7_conv3'))

#         (self.feed('HG3_res6')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res7_skip'))

#         (self.feed('HG3_res7_conv3',
#                    'HG3_res7_skip')
#          .add(name='HG3_res7'))


# # pool4
#         (self.feed('HG3_res6')
#              .max_pool(2, 2, 2, 2, name='HG3_pool4'))

# # res8
#         (self.feed('HG3_pool4')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res8_batch1')
#              .relu(name='HG3_res8_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG3_res8_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res8_batch2')
#              .relu(name='HG3_res8_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG3_res8_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res8_batch3')
#              .relu(name='HG3_res8_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res8_conv3'))

#         (self.feed('HG3_pool4')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res8_skip'))

#         (self.feed('HG3_res8_conv3',
#                    'HG3_res8_skip')
#          .add(name='HG3_res8'))

# # res9
#         (self.feed('HG3_res8')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res9_batch1')
#              .relu(name='HG3_res9_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG3_res9_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res9_batch2')
#              .relu(name='HG3_res9_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG3_res9_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res9_batch3')
#              .relu(name='HG3_res9_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res9_conv3'))

#         (self.feed('HG3_res8')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res9_skip'))

#         (self.feed('HG3_res9_conv3',
#                    'HG3_res9_skip')
#          .add(name='HG3_res9'))

# # res10
#         (self.feed('HG3_res9')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res10_batch1')
#              .relu(name='HG3_res10_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG3_res10_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res10_batch2')
#              .relu(name='HG3_res10_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG3_res10_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res10_batch3')
#              .relu(name='HG3_res10_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res10_conv3'))

#         (self.feed('HG3_res9')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res10_skip'))

#         (self.feed('HG3_res10_conv3',
#                    'HG3_res10_skip')
#          .add(name='HG3_res10'))


# # upsample1
#         (self.feed('HG3_res10')
#              .upsample(8, 8, name='HG3_upSample1'))

# # upsample1 + up1(Hg_res7)
#         (self.feed('HG3_upSample1',
#                    'HG3_res7')
#          .add(name='HG3_add1'))


# # res11
#         (self.feed('HG3_add1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res11_batch1')
#              .relu(name='HG3_res11_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG3_res11_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res11_batch2')
#              .relu(name='HG3_res11_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG3_res11_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res11_batch3')
#              .relu(name='HG3_res11_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res11_conv3'))

#         (self.feed('HG3_add1')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res11_skip'))

#         (self.feed('HG3_res11_conv3',
#                    'HG3_res11_skip')
#          .add(name='HG3_res11'))


# # upsample2
#         (self.feed('HG3_res11')
#              .upsample(16, 16, name='HG3_upSample2'))

# # upsample2 + up1(Hg_res5)
#         (self.feed('HG3_upSample2',
#                    'HG3_res5')
#          .add(name='HG3_add2'))


# # res12
#         (self.feed('HG3_add2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res12_batch1')
#              .relu(name='HG3_res12_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG3_res12_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res12_batch2')
#              .relu(name='HG3_res12_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG3_res12_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res12_batch3')
#              .relu(name='HG3_res12_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res12_conv3'))

#         (self.feed('HG3_add2')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res12_skip'))

#         (self.feed('HG3_res12_conv3',
#                    'HG3_res12_skip')
#          .add(name='HG3_res12'))


# # upsample3
#         (self.feed('HG3_res12')
#              .upsample(32, 32, name='HG3_upSample3'))

# # upsample3 + up1(Hg_res3)
#         (self.feed('HG3_upSample3',
#                    'HG3_res3')
#          .add(name='HG3_add3'))


# # res13
#         (self.feed('HG3_add3')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res13_batch1')
#              .relu(name='HG3_res13_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG3_res13_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res13_batch2')
#              .relu(name='HG3_res13_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG3_res13_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res13_batch3')
#              .relu(name='HG3_res13_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res13_conv3'))

#         (self.feed('HG3_add3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res13_skip'))

#         (self.feed('HG3_res13_conv3',
#                    'HG3_res13_skip')
#          .add(name='HG3_res13'))


# # upsample4
#         (self.feed('HG3_res13')
#              .upsample(64, 64, name='HG3_upSample4'))

# # upsample4 + up1(Hg_res1)
#         (self.feed('HG3_upSample4',
#                    'HG3_res1')
#          .add(name='HG3_add4'))

# ###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  Hourglass3  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

# ################################# Hourglass3 postprocess #################

# # id:025  Res14
#         (self.feed('HG3_add4')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res14_batch1')
#              .relu(name='HG3_res14_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG3_res14_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res14_batch2')
#              .relu(name='HG3_res14_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG3_res14_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res14_batch3')
#              .relu(name='HG3_res14_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res14_conv3')

#          )

#         (self.feed('HG3_add4')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res14_skip'))

#         (self.feed('HG3_res14_conv3',
#                    'HG3_res14_skip')
#          .add(name='HG3_res14'))

# # Linear layer to produce first set of predictions
# # ll
#         (self.feed('HG3_res14')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_linearfunc_conv1', padding='SAME')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_linearfunc_batch1')
#              .relu(name='HG3_linearfunc_relu'))

# #************************************* Predicted heatmaps tmpOut ****************************#
# # tmpOut
#         (self.feed('HG3_linearfunc_relu')
#              .conv(1, 1, n_classes, 1, 1, biased=True, relu=False, name='HG3_Heatmap', padding='SAME'))

# ###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Hourglass3 postprocess ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

# ############################# generate input for next Hourglass ##########

# # ll_
#         (self.feed('HG3_linearfunc_relu')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_ll_', padding='SAME'))
# # tmpOut_
#         (self.feed('HG3_Heatmap')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_tmpOut_', padding='SAME'))
# # inter
#         (self.feed('HG3_input',
#                    'HG3_ll_',
#                    'HG3_tmpOut_')
#          .add(name='HG4_input'))

# ###^^^^^^^^^^^^^^^^^^^^^^^^^^^ generate input for next Hourglass ^^^^^^^^^^^^^^^^^^^^^^^^^^^^###


# #######################################  Hourglass4  #####################
# # res1
#         (self.feed('HG4_input')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res1_batch1')
#              .relu(name='HG4_res1_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG4_res1_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res1_batch2')
#              .relu(name='HG4_res1_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG4_res1_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res1_batch3')
#              .relu(name='HG4_res1_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res1_conv3'))

#         (self.feed('HG4_input')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res1_skip'))

#         (self.feed('HG4_res1_conv3',
#                    'HG4_res1_skip')
#          .add(name='HG4_res1'))

# #   pool1
#         (self.feed('HG4_input')
#              .max_pool(2, 2, 2, 2, name='HG4_pool1'))


# # res2
#         (self.feed('HG4_pool1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res2_batch1')
#              .relu(name='HG4_res2_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG4_res2_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res2_batch2')
#              .relu(name='HG4_res2_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG4_res2_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res2_batch3')
#              .relu(name='HG4_res2_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res2_conv3'))

#         (self.feed('HG4_pool1')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res2_skip'))

#         (self.feed('HG4_res2_conv3',
#                    'HG4_res2_skip')
#          .add(name='HG4_res2'))
# # id:009 max-pooling
#         # (self.feed('HG4_pool1')
#         #      .max_pool(2, 2, 2, 2, name='HG4_pool2'))


# # res3
#         (self.feed('HG4_res2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res3_batch1')
#              .relu(name='HG4_res3_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG4_res3_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res3_batch2')
#              .relu(name='HG4_res3_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG4_res3_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res3_batch3')
#              .relu(name='HG4_res3_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res3_conv3'))

#         (self.feed('HG4_res2')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res3_skip'))

#         (self.feed('HG4_res3_conv3',
#                    'HG4_res3_skip')
#          .add(name='HG4_res3'))


# # pool2
#         (self.feed('HG4_res2')
#              .max_pool(2, 2, 2, 2, name='HG4_pool2'))


# # res4
#         (self.feed('HG4_pool2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res4_batch1')
#              .relu(name='HG4_res4_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG4_res4_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res4_batch2')
#              .relu(name='HG4_res4_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG4_res4_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res4_batch3')
#              .relu(name='HG4_res4_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res4_conv3'))

#         (self.feed('HG4_pool2')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res4_skip'))

#         (self.feed('HG4_res4_conv3',
#                    'HG4_res4_skip')
#          .add(name='HG4_res4'))
# # id:013 max-pooling
#         # (self.feed('HG4_pool3')
#         #      .max_pool(2, 2, 2, 2, name='HG4_pool4'))


# # res5
#         (self.feed('HG4_res4')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res5_batch1')
#              .relu(name='HG4_res5_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG4_res5_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res5_batch2')
#              .relu(name='HG4_res5_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG4_res5_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res5_batch3')
#              .relu(name='HG4_res5_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res5_conv3'))

#         (self.feed('HG4_res4')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res5_skip'))

#         (self.feed('HG4_res5_conv3',
#                    'HG4_res5_skip')
#          .add(name='HG4_res5'))


# # pool3
#         (self.feed('HG4_res4')
#              .max_pool(2, 2, 2, 2, name='HG4_pool3'))


# # res6
#         (self.feed('HG4_pool3')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res6_batch1')
#              .relu(name='HG4_res6_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG4_res6_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res6_batch2')
#              .relu(name='HG4_res6_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG4_res6_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res6_batch3')
#              .relu(name='HG4_res6_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res6_conv3'))

#         (self.feed('HG4_pool3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res6_skip'))

#         (self.feed('HG4_res6_conv3',
#                    'HG4_res6_skip')
#          .add(name='HG4_res6'))

# # res7
#         (self.feed('HG4_res6')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res7_batch1')
#              .relu(name='HG4_res7_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG4_res7_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res7_batch2')
#              .relu(name='HG4_res7_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG4_res7_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res7_batch3')
#              .relu(name='HG4_res7_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res7_conv3'))

#         (self.feed('HG4_res6')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res7_skip'))

#         (self.feed('HG4_res7_conv3',
#                    'HG4_res7_skip')
#          .add(name='HG4_res7'))


# # pool4
#         (self.feed('HG4_res6')
#              .max_pool(2, 2, 2, 2, name='HG4_pool4'))

# # res8
#         (self.feed('HG4_pool4')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res8_batch1')
#              .relu(name='HG4_res8_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG4_res8_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res8_batch2')
#              .relu(name='HG4_res8_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG4_res8_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res8_batch3')
#              .relu(name='HG4_res8_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res8_conv3'))

#         (self.feed('HG4_pool4')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res8_skip'))

#         (self.feed('HG4_res8_conv3',
#                    'HG4_res8_skip')
#          .add(name='HG4_res8'))

# # res9
#         (self.feed('HG4_res8')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res9_batch1')
#              .relu(name='HG4_res9_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG4_res9_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res9_batch2')
#              .relu(name='HG4_res9_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG4_res9_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res9_batch3')
#              .relu(name='HG4_res9_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res9_conv3'))

#         (self.feed('HG4_res8')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res9_skip'))

#         (self.feed('HG4_res9_conv3',
#                    'HG4_res9_skip')
#          .add(name='HG4_res9'))

# # res10
#         (self.feed('HG4_res9')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res10_batch1')
#              .relu(name='HG4_res10_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG4_res10_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res10_batch2')
#              .relu(name='HG4_res10_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG4_res10_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res10_batch3')
#              .relu(name='HG4_res10_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res10_conv3'))

#         (self.feed('HG4_res9')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res10_skip'))

#         (self.feed('HG4_res10_conv3',
#                    'HG4_res10_skip')
#          .add(name='HG4_res10'))


# # upsample1
#         (self.feed('HG4_res10')
#              .upsample(8, 8, name='HG4_upSample1'))

# # upsample1 + up1(Hg_res7)
#         (self.feed('HG4_upSample1',
#                    'HG4_res7')
#          .add(name='HG4_add1'))


# # res11
#         (self.feed('HG4_add1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res11_batch1')
#              .relu(name='HG4_res11_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG4_res11_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res11_batch2')
#              .relu(name='HG4_res11_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG4_res11_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res11_batch3')
#              .relu(name='HG4_res11_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res11_conv3'))

#         (self.feed('HG4_add1')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res11_skip'))

#         (self.feed('HG4_res11_conv3',
#                    'HG4_res11_skip')
#          .add(name='HG4_res11'))


# # upsample2
#         (self.feed('HG4_res11')
#              .upsample(16, 16, name='HG4_upSample2'))

# # upsample2 + up1(Hg_res5)
#         (self.feed('HG4_upSample2',
#                    'HG4_res5')
#          .add(name='HG4_add2'))


# # res12
#         (self.feed('HG4_add2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res12_batch1')
#              .relu(name='HG4_res12_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG4_res12_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res12_batch2')
#              .relu(name='HG4_res12_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG4_res12_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res12_batch3')
#              .relu(name='HG4_res12_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res12_conv3'))

#         (self.feed('HG4_add2')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res12_skip'))

#         (self.feed('HG4_res12_conv3',
#                    'HG4_res12_skip')
#          .add(name='HG4_res12'))


# # upsample3
#         (self.feed('HG4_res12')
#              .upsample(32, 32, name='HG4_upSample3'))

# # upsample3 + up1(Hg_res3)
#         (self.feed('HG4_upSample3',
#                    'HG4_res3')
#          .add(name='HG4_add3'))


# # res13
#         (self.feed('HG4_add3')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res13_batch1')
#              .relu(name='HG4_res13_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG4_res13_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res13_batch2')
#              .relu(name='HG4_res13_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG4_res13_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res13_batch3')
#              .relu(name='HG4_res13_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res13_conv3'))

#         (self.feed('HG4_add3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res13_skip'))

#         (self.feed('HG4_res13_conv3',
#                    'HG4_res13_skip')
#          .add(name='HG4_res13'))


# # upsample4
#         (self.feed('HG4_res13')
#              .upsample(64, 64, name='HG4_upSample4'))

# # upsample4 + up1(Hg_res1)
#         (self.feed('HG4_upSample4',
#                    'HG4_res1')
#          .add(name='HG4_add4'))

# ###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  Hourglass4  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

# ################################# Hourglass4 postprocess #################

# # id:025  Res14
#         (self.feed('HG4_add4')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res14_batch1')
#              .relu(name='HG4_res14_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG4_res14_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res14_batch2')
#              .relu(name='HG4_res14_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG4_res14_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res14_batch3')
#              .relu(name='HG4_res14_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res14_conv3')

#          )

#         (self.feed('HG4_add4')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res14_skip'))

#         (self.feed('HG4_res14_conv3',
#                    'HG4_res14_skip')
#          .add(name='HG4_res14'))

# # Linear layer to produce first set of predictions
# # ll
#         (self.feed('HG4_res14')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_linearfunc_conv1', padding='SAME')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_linearfunc_batch1')
#              .relu(name='HG4_linearfunc_relu'))

# #************************************* Predicted heatmaps tmpOut ****************************#
# # tmpOut
#         (self.feed('HG4_linearfunc_relu')
#              .conv(1, 1, n_classes, 1, 1, biased=True, relu=False, name='HG4_Heatmap', padding='SAME'))

# ###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Hourglass4 postprocess ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

# ############################# generate input for next Hourglass ##########

# # ll_
#         (self.feed('HG4_linearfunc_relu')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_ll_', padding='SAME'))
# # tmpOut_
#         (self.feed('HG4_Heatmap')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_tmpOut_', padding='SAME'))
# # inter
#         (self.feed('HG4_input',
#                    'HG4_ll_',
#                    'HG4_tmpOut_')
#          .add(name='HG5_input'))

# ###^^^^^^^^^^^^^^^^^^^^^^^^^^^ generate input for next Hourglass ^^^^^^^^^^^^^^^^^^^^^^^^^^^^###




# #######################################  Hourglass5  #####################
# # res1
#         (self.feed('HG5_input')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res1_batch1')
#              .relu(name='HG5_res1_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG5_res1_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res1_batch2')
#              .relu(name='HG5_res1_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG5_res1_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res1_batch3')
#              .relu(name='HG5_res1_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res1_conv3'))

#         (self.feed('HG5_input')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res1_skip'))

#         (self.feed('HG5_res1_conv3',
#                    'HG5_res1_skip')
#          .add(name='HG5_res1'))

# #   pool1
#         (self.feed('HG5_input')
#              .max_pool(2, 2, 2, 2, name='HG5_pool1'))


# # res2
#         (self.feed('HG5_pool1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res2_batch1')
#              .relu(name='HG5_res2_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG5_res2_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res2_batch2')
#              .relu(name='HG5_res2_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG5_res2_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res2_batch3')
#              .relu(name='HG5_res2_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res2_conv3'))

#         (self.feed('HG5_pool1')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res2_skip'))

#         (self.feed('HG5_res2_conv3',
#                    'HG5_res2_skip')
#          .add(name='HG5_res2'))
# # id:009 max-pooling
#         # (self.feed('HG5_pool1')
#         #      .max_pool(2, 2, 2, 2, name='HG5_pool2'))


# # res3
#         (self.feed('HG5_res2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res3_batch1')
#              .relu(name='HG5_res3_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG5_res3_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res3_batch2')
#              .relu(name='HG5_res3_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG5_res3_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res3_batch3')
#              .relu(name='HG5_res3_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res3_conv3'))

#         (self.feed('HG5_res2')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res3_skip'))

#         (self.feed('HG5_res3_conv3',
#                    'HG5_res3_skip')
#          .add(name='HG5_res3'))


# # pool2
#         (self.feed('HG5_res2')
#              .max_pool(2, 2, 2, 2, name='HG5_pool2'))


# # res4
#         (self.feed('HG5_pool2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res4_batch1')
#              .relu(name='HG5_res4_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG5_res4_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res4_batch2')
#              .relu(name='HG5_res4_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG5_res4_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res4_batch3')
#              .relu(name='HG5_res4_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res4_conv3'))

#         (self.feed('HG5_pool2')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res4_skip'))

#         (self.feed('HG5_res4_conv3',
#                    'HG5_res4_skip')
#          .add(name='HG5_res4'))
# # id:013 max-pooling
#         # (self.feed('HG5_pool3')
#         #      .max_pool(2, 2, 2, 2, name='HG5_pool4'))


# # res5
#         (self.feed('HG5_res4')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res5_batch1')
#              .relu(name='HG5_res5_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG5_res5_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res5_batch2')
#              .relu(name='HG5_res5_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG5_res5_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res5_batch3')
#              .relu(name='HG5_res5_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res5_conv3'))

#         (self.feed('HG5_res4')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res5_skip'))

#         (self.feed('HG5_res5_conv3',
#                    'HG5_res5_skip')
#          .add(name='HG5_res5'))


# # pool3
#         (self.feed('HG5_res4')
#              .max_pool(2, 2, 2, 2, name='HG5_pool3'))


# # res6
#         (self.feed('HG5_pool3')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res6_batch1')
#              .relu(name='HG5_res6_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG5_res6_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res6_batch2')
#              .relu(name='HG5_res6_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG5_res6_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res6_batch3')
#              .relu(name='HG5_res6_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res6_conv3'))

#         (self.feed('HG5_pool3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res6_skip'))

#         (self.feed('HG5_res6_conv3',
#                    'HG5_res6_skip')
#          .add(name='HG5_res6'))

# # res7
#         (self.feed('HG5_res6')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res7_batch1')
#              .relu(name='HG5_res7_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG5_res7_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res7_batch2')
#              .relu(name='HG5_res7_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG5_res7_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res7_batch3')
#              .relu(name='HG5_res7_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res7_conv3'))

#         (self.feed('HG5_res6')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res7_skip'))

#         (self.feed('HG5_res7_conv3',
#                    'HG5_res7_skip')
#          .add(name='HG5_res7'))


# # pool4
#         (self.feed('HG5_res6')
#              .max_pool(2, 2, 2, 2, name='HG5_pool4'))

# # res8
#         (self.feed('HG5_pool4')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res8_batch1')
#              .relu(name='HG5_res8_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG5_res8_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res8_batch2')
#              .relu(name='HG5_res8_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG5_res8_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res8_batch3')
#              .relu(name='HG5_res8_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res8_conv3'))

#         (self.feed('HG5_pool4')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res8_skip'))

#         (self.feed('HG5_res8_conv3',
#                    'HG5_res8_skip')
#          .add(name='HG5_res8'))

# # res9
#         (self.feed('HG5_res8')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res9_batch1')
#              .relu(name='HG5_res9_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG5_res9_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res9_batch2')
#              .relu(name='HG5_res9_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG5_res9_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res9_batch3')
#              .relu(name='HG5_res9_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res9_conv3'))

#         (self.feed('HG5_res8')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res9_skip'))

#         (self.feed('HG5_res9_conv3',
#                    'HG5_res9_skip')
#          .add(name='HG5_res9'))

# # res10
#         (self.feed('HG5_res9')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res10_batch1')
#              .relu(name='HG5_res10_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG5_res10_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res10_batch2')
#              .relu(name='HG5_res10_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG5_res10_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res10_batch3')
#              .relu(name='HG5_res10_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res10_conv3'))

#         (self.feed('HG5_res9')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res10_skip'))

#         (self.feed('HG5_res10_conv3',
#                    'HG5_res10_skip')
#          .add(name='HG5_res10'))


# # upsample1
#         (self.feed('HG5_res10')
#              .upsample(8, 8, name='HG5_upSample1'))

# # upsample1 + up1(Hg_res7)
#         (self.feed('HG5_upSample1',
#                    'HG5_res7')
#          .add(name='HG5_add1'))


# # res11
#         (self.feed('HG5_add1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res11_batch1')
#              .relu(name='HG5_res11_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG5_res11_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res11_batch2')
#              .relu(name='HG5_res11_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG5_res11_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res11_batch3')
#              .relu(name='HG5_res11_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res11_conv3'))

#         (self.feed('HG5_add1')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res11_skip'))

#         (self.feed('HG5_res11_conv3',
#                    'HG5_res11_skip')
#          .add(name='HG5_res11'))


# # upsample2
#         (self.feed('HG5_res11')
#              .upsample(16, 16, name='HG5_upSample2'))

# # upsample2 + up1(Hg_res5)
#         (self.feed('HG5_upSample2',
#                    'HG5_res5')
#          .add(name='HG5_add2'))


# # res12
#         (self.feed('HG5_add2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res12_batch1')
#              .relu(name='HG5_res12_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG5_res12_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res12_batch2')
#              .relu(name='HG5_res12_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG5_res12_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res12_batch3')
#              .relu(name='HG5_res12_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res12_conv3'))

#         (self.feed('HG5_add2')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res12_skip'))

#         (self.feed('HG5_res12_conv3',
#                    'HG5_res12_skip')
#          .add(name='HG5_res12'))


# # upsample3
#         (self.feed('HG5_res12')
#              .upsample(32, 32, name='HG5_upSample3'))

# # upsample3 + up1(Hg_res3)
#         (self.feed('HG5_upSample3',
#                    'HG5_res3')
#          .add(name='HG5_add3'))


# # res13
#         (self.feed('HG5_add3')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res13_batch1')
#              .relu(name='HG5_res13_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG5_res13_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res13_batch2')
#              .relu(name='HG5_res13_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG5_res13_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res13_batch3')
#              .relu(name='HG5_res13_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res13_conv3'))

#         (self.feed('HG5_add3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res13_skip'))

#         (self.feed('HG5_res13_conv3',
#                    'HG5_res13_skip')
#          .add(name='HG5_res13'))


# # upsample4
#         (self.feed('HG5_res13')
#              .upsample(64, 64, name='HG5_upSample4'))

# # upsample4 + up1(Hg_res1)
#         (self.feed('HG5_upSample4',
#                    'HG5_res1')
#          .add(name='HG5_add4'))

# ###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  Hourglass5  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

# ################################# Hourglass5 postprocess #################

# # id:025  Res14
#         (self.feed('HG5_add4')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res14_batch1')
#              .relu(name='HG5_res14_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG5_res14_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res14_batch2')
#              .relu(name='HG5_res14_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG5_res14_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res14_batch3')
#              .relu(name='HG5_res14_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res14_conv3')

#          )

#         (self.feed('HG5_add4')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res14_skip'))

#         (self.feed('HG5_res14_conv3',
#                    'HG5_res14_skip')
#          .add(name='HG5_res14'))

# # Linear layer to produce first set of predictions
# # ll
#         (self.feed('HG5_res14')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_linearfunc_conv1', padding='SAME')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_linearfunc_batch1')
#              .relu(name='HG5_linearfunc_relu'))

# #************************************* Predicted heatmaps tmpOut ****************************#
# # tmpOut
#         (self.feed('HG5_linearfunc_relu')
#              .conv(1, 1, n_classes, 1, 1, biased=True, relu=False, name='HG5_Heatmap', padding='SAME'))

# ###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Hourglass5 postprocess ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

# ############################# generate input for next Hourglass ##########

# # ll_
#         (self.feed('HG5_linearfunc_relu')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_ll_', padding='SAME'))
# # tmpOut_
#         (self.feed('HG5_Heatmap')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_tmpOut_', padding='SAME'))
# # inter
#         (self.feed('HG5_input',
#                    'HG5_ll_',
#                    'HG5_tmpOut_')
#          .add(name='HG6_input'))

# ###^^^^^^^^^^^^^^^^^^^^^^^^^^^ generate input for next Hourglass ^^^^^^^^^^^^^^^^^^^^^^^^^^^^###





# #######################################  Hourglass6  #####################
# # res1
#         (self.feed('HG6_input')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res1_batch1')
#              .relu(name='HG6_res1_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG6_res1_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res1_batch2')
#              .relu(name='HG6_res1_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG6_res1_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res1_batch3')
#              .relu(name='HG6_res1_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res1_conv3'))

#         (self.feed('HG6_input')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res1_skip'))

#         (self.feed('HG6_res1_conv3',
#                    'HG6_res1_skip')
#          .add(name='HG6_res1'))

# #   pool1
#         (self.feed('HG6_input')
#              .max_pool(2, 2, 2, 2, name='HG6_pool1'))


# # res2
#         (self.feed('HG6_pool1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res2_batch1')
#              .relu(name='HG6_res2_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG6_res2_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res2_batch2')
#              .relu(name='HG6_res2_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG6_res2_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res2_batch3')
#              .relu(name='HG6_res2_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res2_conv3'))

#         (self.feed('HG6_pool1')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res2_skip'))

#         (self.feed('HG6_res2_conv3',
#                    'HG6_res2_skip')
#          .add(name='HG6_res2'))
# # id:009 max-pooling
#         # (self.feed('HG6_pool1')
#         #      .max_pool(2, 2, 2, 2, name='HG6_pool2'))


# # res3
#         (self.feed('HG6_res2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res3_batch1')
#              .relu(name='HG6_res3_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG6_res3_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res3_batch2')
#              .relu(name='HG6_res3_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG6_res3_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res3_batch3')
#              .relu(name='HG6_res3_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res3_conv3'))

#         (self.feed('HG6_res2')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res3_skip'))

#         (self.feed('HG6_res3_conv3',
#                    'HG6_res3_skip')
#          .add(name='HG6_res3'))


# # pool2
#         (self.feed('HG6_res2')
#              .max_pool(2, 2, 2, 2, name='HG6_pool2'))


# # res4
#         (self.feed('HG6_pool2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res4_batch1')
#              .relu(name='HG6_res4_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG6_res4_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res4_batch2')
#              .relu(name='HG6_res4_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG6_res4_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res4_batch3')
#              .relu(name='HG6_res4_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res4_conv3'))

#         (self.feed('HG6_pool2')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res4_skip'))

#         (self.feed('HG6_res4_conv3',
#                    'HG6_res4_skip')
#          .add(name='HG6_res4'))
# # id:013 max-pooling
#         # (self.feed('HG6_pool3')
#         #      .max_pool(2, 2, 2, 2, name='HG6_pool4'))


# # res5
#         (self.feed('HG6_res4')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res5_batch1')
#              .relu(name='HG6_res5_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG6_res5_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res5_batch2')
#              .relu(name='HG6_res5_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG6_res5_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res5_batch3')
#              .relu(name='HG6_res5_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res5_conv3'))

#         (self.feed('HG6_res4')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res5_skip'))

#         (self.feed('HG6_res5_conv3',
#                    'HG6_res5_skip')
#          .add(name='HG6_res5'))


# # pool3
#         (self.feed('HG6_res4')
#              .max_pool(2, 2, 2, 2, name='HG6_pool3'))


# # res6
#         (self.feed('HG6_pool3')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res6_batch1')
#              .relu(name='HG6_res6_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG6_res6_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res6_batch2')
#              .relu(name='HG6_res6_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG6_res6_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res6_batch3')
#              .relu(name='HG6_res6_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res6_conv3'))

#         (self.feed('HG6_pool3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res6_skip'))

#         (self.feed('HG6_res6_conv3',
#                    'HG6_res6_skip')
#          .add(name='HG6_res6'))

# # res7
#         (self.feed('HG6_res6')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res7_batch1')
#              .relu(name='HG6_res7_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG6_res7_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res7_batch2')
#              .relu(name='HG6_res7_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG6_res7_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res7_batch3')
#              .relu(name='HG6_res7_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res7_conv3'))

#         (self.feed('HG6_res6')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res7_skip'))

#         (self.feed('HG6_res7_conv3',
#                    'HG6_res7_skip')
#          .add(name='HG6_res7'))


# # pool4
#         (self.feed('HG6_res6')
#              .max_pool(2, 2, 2, 2, name='HG6_pool4'))

# # res8
#         (self.feed('HG6_pool4')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res8_batch1')
#              .relu(name='HG6_res8_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG6_res8_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res8_batch2')
#              .relu(name='HG6_res8_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG6_res8_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res8_batch3')
#              .relu(name='HG6_res8_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res8_conv3'))

#         (self.feed('HG6_pool4')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res8_skip'))

#         (self.feed('HG6_res8_conv3',
#                    'HG6_res8_skip')
#          .add(name='HG6_res8'))

# # res9
#         (self.feed('HG6_res8')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res9_batch1')
#              .relu(name='HG6_res9_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG6_res9_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res9_batch2')
#              .relu(name='HG6_res9_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG6_res9_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res9_batch3')
#              .relu(name='HG6_res9_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res9_conv3'))

#         (self.feed('HG6_res8')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res9_skip'))

#         (self.feed('HG6_res9_conv3',
#                    'HG6_res9_skip')
#          .add(name='HG6_res9'))

# # res10
#         (self.feed('HG6_res9')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res10_batch1')
#              .relu(name='HG6_res10_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG6_res10_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res10_batch2')
#              .relu(name='HG6_res10_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG6_res10_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res10_batch3')
#              .relu(name='HG6_res10_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res10_conv3'))

#         (self.feed('HG6_res9')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res10_skip'))

#         (self.feed('HG6_res10_conv3',
#                    'HG6_res10_skip')
#          .add(name='HG6_res10'))


# # upsample1
#         (self.feed('HG6_res10')
#              .upsample(8, 8, name='HG6_upSample1'))

# # upsample1 + up1(Hg_res7)
#         (self.feed('HG6_upSample1',
#                    'HG6_res7')
#          .add(name='HG6_add1'))


# # res11
#         (self.feed('HG6_add1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res11_batch1')
#              .relu(name='HG6_res11_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG6_res11_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res11_batch2')
#              .relu(name='HG6_res11_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG6_res11_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res11_batch3')
#              .relu(name='HG6_res11_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res11_conv3'))

#         (self.feed('HG6_add1')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res11_skip'))

#         (self.feed('HG6_res11_conv3',
#                    'HG6_res11_skip')
#          .add(name='HG6_res11'))


# # upsample2
#         (self.feed('HG6_res11')
#              .upsample(16, 16, name='HG6_upSample2'))

# # upsample2 + up1(Hg_res5)
#         (self.feed('HG6_upSample2',
#                    'HG6_res5')
#          .add(name='HG6_add2'))


# # res12
#         (self.feed('HG6_add2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res12_batch1')
#              .relu(name='HG6_res12_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG6_res12_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res12_batch2')
#              .relu(name='HG6_res12_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG6_res12_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res12_batch3')
#              .relu(name='HG6_res12_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res12_conv3'))

#         (self.feed('HG6_add2')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res12_skip'))

#         (self.feed('HG6_res12_conv3',
#                    'HG6_res12_skip')
#          .add(name='HG6_res12'))


# # upsample3
#         (self.feed('HG6_res12')
#              .upsample(32, 32, name='HG6_upSample3'))

# # upsample3 + up1(Hg_res3)
#         (self.feed('HG6_upSample3',
#                    'HG6_res3')
#          .add(name='HG6_add3'))


# # res13
#         (self.feed('HG6_add3')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res13_batch1')
#              .relu(name='HG6_res13_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG6_res13_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res13_batch2')
#              .relu(name='HG6_res13_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG6_res13_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res13_batch3')
#              .relu(name='HG6_res13_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res13_conv3'))

#         (self.feed('HG6_add3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res13_skip'))

#         (self.feed('HG6_res13_conv3',
#                    'HG6_res13_skip')
#          .add(name='HG6_res13'))


# # upsample4
#         (self.feed('HG6_res13')
#              .upsample(64, 64, name='HG6_upSample4'))

# # upsample4 + up1(Hg_res1)
#         (self.feed('HG6_upSample4',
#                    'HG6_res1')
#          .add(name='HG6_add4'))

# ###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  Hourglass6  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

# ################################# Hourglass6 postprocess #################

# # id:025  Res14
#         (self.feed('HG6_add4')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res14_batch1')
#              .relu(name='HG6_res14_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG6_res14_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res14_batch2')
#              .relu(name='HG6_res14_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG6_res14_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res14_batch3')
#              .relu(name='HG6_res14_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res14_conv3')

#          )

#         (self.feed('HG6_add4')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res14_skip'))

#         (self.feed('HG6_res14_conv3',
#                    'HG6_res14_skip')
#          .add(name='HG6_res14'))

# # Linear layer to produce first set of predictions
# # ll
#         (self.feed('HG6_res14')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_linearfunc_conv1', padding='SAME')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_linearfunc_batch1')
#              .relu(name='HG6_linearfunc_relu'))

# #************************************* Predicted heatmaps tmpOut ****************************#
# # tmpOut
#         (self.feed('HG6_linearfunc_relu')
#              .conv(1, 1, n_classes, 1, 1, biased=True, relu=False, name='HG6_Heatmap', padding='SAME'))

# ###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Hourglass6 postprocess ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

# ############################# generate input for next Hourglass ##########

# # ll_
#         (self.feed('HG6_linearfunc_relu')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_ll_', padding='SAME'))
# # tmpOut_
#         (self.feed('HG6_Heatmap')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_tmpOut_', padding='SAME'))
# # inter
#         (self.feed('HG6_input',
#                    'HG6_ll_',
#                    'HG6_tmpOut_')
#          .add(name='HG7_input'))

# ###^^^^^^^^^^^^^^^^^^^^^^^^^^^ generate input for next Hourglass ^^^^^^^^^^^^^^^^^^^^^^^^^^^^###






# #######################################  Hourglass7  #####################
# # res1
#         (self.feed('HG7_input')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res1_batch1')
#              .relu(name='HG7_res1_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG7_res1_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res1_batch2')
#              .relu(name='HG7_res1_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG7_res1_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res1_batch3')
#              .relu(name='HG7_res1_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res1_conv3'))

#         (self.feed('HG7_input')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res1_skip'))

#         (self.feed('HG7_res1_conv3',
#                    'HG7_res1_skip')
#          .add(name='HG7_res1'))

# #   pool1
#         (self.feed('HG7_input')
#              .max_pool(2, 2, 2, 2, name='HG7_pool1'))


# # res2
#         (self.feed('HG7_pool1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res2_batch1')
#              .relu(name='HG7_res2_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG7_res2_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res2_batch2')
#              .relu(name='HG7_res2_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG7_res2_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res2_batch3')
#              .relu(name='HG7_res2_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res2_conv3'))

#         (self.feed('HG7_pool1')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res2_skip'))

#         (self.feed('HG7_res2_conv3',
#                    'HG7_res2_skip')
#          .add(name='HG7_res2'))
# # id:009 max-pooling
#         # (self.feed('HG7_pool1')
#         #      .max_pool(2, 2, 2, 2, name='HG7_pool2'))


# # res3
#         (self.feed('HG7_res2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res3_batch1')
#              .relu(name='HG7_res3_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG7_res3_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res3_batch2')
#              .relu(name='HG7_res3_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG7_res3_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res3_batch3')
#              .relu(name='HG7_res3_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res3_conv3'))

#         (self.feed('HG7_res2')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res3_skip'))

#         (self.feed('HG7_res3_conv3',
#                    'HG7_res3_skip')
#          .add(name='HG7_res3'))


# # pool2
#         (self.feed('HG7_res2')
#              .max_pool(2, 2, 2, 2, name='HG7_pool2'))


# # res4
#         (self.feed('HG7_pool2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res4_batch1')
#              .relu(name='HG7_res4_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG7_res4_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res4_batch2')
#              .relu(name='HG7_res4_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG7_res4_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res4_batch3')
#              .relu(name='HG7_res4_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res4_conv3'))

#         (self.feed('HG7_pool2')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res4_skip'))

#         (self.feed('HG7_res4_conv3',
#                    'HG7_res4_skip')
#          .add(name='HG7_res4'))
# # id:013 max-pooling
#         # (self.feed('HG7_pool3')
#         #      .max_pool(2, 2, 2, 2, name='HG7_pool4'))


# # res5
#         (self.feed('HG7_res4')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res5_batch1')
#              .relu(name='HG7_res5_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG7_res5_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res5_batch2')
#              .relu(name='HG7_res5_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG7_res5_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res5_batch3')
#              .relu(name='HG7_res5_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res5_conv3'))

#         (self.feed('HG7_res4')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res5_skip'))

#         (self.feed('HG7_res5_conv3',
#                    'HG7_res5_skip')
#          .add(name='HG7_res5'))


# # pool3
#         (self.feed('HG7_res4')
#              .max_pool(2, 2, 2, 2, name='HG7_pool3'))


# # res6
#         (self.feed('HG7_pool3')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res6_batch1')
#              .relu(name='HG7_res6_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG7_res6_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res6_batch2')
#              .relu(name='HG7_res6_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG7_res6_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res6_batch3')
#              .relu(name='HG7_res6_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res6_conv3'))

#         (self.feed('HG7_pool3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res6_skip'))

#         (self.feed('HG7_res6_conv3',
#                    'HG7_res6_skip')
#          .add(name='HG7_res6'))

# # res7
#         (self.feed('HG7_res6')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res7_batch1')
#              .relu(name='HG7_res7_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG7_res7_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res7_batch2')
#              .relu(name='HG7_res7_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG7_res7_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res7_batch3')
#              .relu(name='HG7_res7_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res7_conv3'))

#         (self.feed('HG7_res6')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res7_skip'))

#         (self.feed('HG7_res7_conv3',
#                    'HG7_res7_skip')
#          .add(name='HG7_res7'))


# # pool4
#         (self.feed('HG7_res6')
#              .max_pool(2, 2, 2, 2, name='HG7_pool4'))

# # res8
#         (self.feed('HG7_pool4')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res8_batch1')
#              .relu(name='HG7_res8_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG7_res8_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res8_batch2')
#              .relu(name='HG7_res8_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG7_res8_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res8_batch3')
#              .relu(name='HG7_res8_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res8_conv3'))

#         (self.feed('HG7_pool4')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res8_skip'))

#         (self.feed('HG7_res8_conv3',
#                    'HG7_res8_skip')
#          .add(name='HG7_res8'))

# # res9
#         (self.feed('HG7_res8')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res9_batch1')
#              .relu(name='HG7_res9_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG7_res9_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res9_batch2')
#              .relu(name='HG7_res9_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG7_res9_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res9_batch3')
#              .relu(name='HG7_res9_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res9_conv3'))

#         (self.feed('HG7_res8')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res9_skip'))

#         (self.feed('HG7_res9_conv3',
#                    'HG7_res9_skip')
#          .add(name='HG7_res9'))

# # res10
#         (self.feed('HG7_res9')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res10_batch1')
#              .relu(name='HG7_res10_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG7_res10_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res10_batch2')
#              .relu(name='HG7_res10_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG7_res10_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res10_batch3')
#              .relu(name='HG7_res10_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res10_conv3'))

#         (self.feed('HG7_res9')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res10_skip'))

#         (self.feed('HG7_res10_conv3',
#                    'HG7_res10_skip')
#          .add(name='HG7_res10'))


# # upsample1
#         (self.feed('HG7_res10')
#              .upsample(8, 8, name='HG7_upSample1'))

# # upsample1 + up1(Hg_res7)
#         (self.feed('HG7_upSample1',
#                    'HG7_res7')
#          .add(name='HG7_add1'))


# # res11
#         (self.feed('HG7_add1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res11_batch1')
#              .relu(name='HG7_res11_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG7_res11_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res11_batch2')
#              .relu(name='HG7_res11_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG7_res11_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res11_batch3')
#              .relu(name='HG7_res11_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res11_conv3'))

#         (self.feed('HG7_add1')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res11_skip'))

#         (self.feed('HG7_res11_conv3',
#                    'HG7_res11_skip')
#          .add(name='HG7_res11'))


# # upsample2
#         (self.feed('HG7_res11')
#              .upsample(16, 16, name='HG7_upSample2'))

# # upsample2 + up1(Hg_res5)
#         (self.feed('HG7_upSample2',
#                    'HG7_res5')
#          .add(name='HG7_add2'))


# # res12
#         (self.feed('HG7_add2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res12_batch1')
#              .relu(name='HG7_res12_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG7_res12_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res12_batch2')
#              .relu(name='HG7_res12_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG7_res12_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res12_batch3')
#              .relu(name='HG7_res12_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res12_conv3'))

#         (self.feed('HG7_add2')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res12_skip'))

#         (self.feed('HG7_res12_conv3',
#                    'HG7_res12_skip')
#          .add(name='HG7_res12'))


# # upsample3
#         (self.feed('HG7_res12')
#              .upsample(32, 32, name='HG7_upSample3'))

# # upsample3 + up1(Hg_res3)
#         (self.feed('HG7_upSample3',
#                    'HG7_res3')
#          .add(name='HG7_add3'))


# # res13
#         (self.feed('HG7_add3')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res13_batch1')
#              .relu(name='HG7_res13_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG7_res13_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res13_batch2')
#              .relu(name='HG7_res13_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG7_res13_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res13_batch3')
#              .relu(name='HG7_res13_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res13_conv3'))

#         (self.feed('HG7_add3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res13_skip'))

#         (self.feed('HG7_res13_conv3',
#                    'HG7_res13_skip')
#          .add(name='HG7_res13'))


# # upsample4
#         (self.feed('HG7_res13')
#              .upsample(64, 64, name='HG7_upSample4'))

# # upsample4 + up1(Hg_res1)
#         (self.feed('HG7_upSample4',
#                    'HG7_res1')
#          .add(name='HG7_add4'))

# ###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  Hourglass7  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

# ################################# Hourglass7 postprocess #################

# # id:025  Res14
#         (self.feed('HG7_add4')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res14_batch1')
#              .relu(name='HG7_res14_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG7_res14_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res14_batch2')
#              .relu(name='HG7_res14_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG7_res14_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res14_batch3')
#              .relu(name='HG7_res14_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res14_conv3')

#          )

#         (self.feed('HG7_add4')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res14_skip'))

#         (self.feed('HG7_res14_conv3',
#                    'HG7_res14_skip')
#          .add(name='HG7_res14'))

# # Linear layer to produce first set of predictions
# # ll
#         (self.feed('HG7_res14')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_linearfunc_conv1', padding='SAME')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_linearfunc_batch1')
#              .relu(name='HG7_linearfunc_relu'))

# #************************************* Predicted heatmaps tmpOut ****************************#
# # tmpOut
#         (self.feed('HG7_linearfunc_relu')
#              .conv(1, 1, n_classes, 1, 1, biased=True, relu=False, name='HG7_Heatmap', padding='SAME'))

# ###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Hourglass7 postprocess ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

# ############################# generate input for next Hourglass ##########

# # ll_
#         (self.feed('HG7_linearfunc_relu')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_ll_', padding='SAME'))
# # tmpOut_
#         (self.feed('HG7_Heatmap')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_tmpOut_', padding='SAME'))
# # inter
#         (self.feed('HG7_input',
#                    'HG7_ll_',
#                    'HG7_tmpOut_')
#          .add(name='HG8_input'))

# ###^^^^^^^^^^^^^^^^^^^^^^^^^^^ generate input for next Hourglass ^^^^^^^^^^^^^^^^^^^^^^^^^^^^###
























# #######################################  Hourglass8  #####################
# # res1
#         (self.feed('HG8_input')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res1_batch1')
#              .relu(name='HG8_res1_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG8_res1_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res1_batch2')
#              .relu(name='HG8_res1_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG8_res1_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res1_batch3')
#              .relu(name='HG8_res1_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res1_conv3'))

#         (self.feed('HG8_input')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res1_skip'))

#         (self.feed('HG8_res1_conv3',
#                    'HG8_res1_skip')
#          .add(name='HG8_res1'))

# #   pool1
#         (self.feed('HG8_input')
#              .max_pool(2, 2, 2, 2, name='HG8_pool1'))


# # res2
#         (self.feed('HG8_pool1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res2_batch1')
#              .relu(name='HG8_res2_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG8_res2_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res2_batch2')
#              .relu(name='HG8_res2_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG8_res2_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res2_batch3')
#              .relu(name='HG8_res2_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res2_conv3'))

#         (self.feed('HG8_pool1')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res2_skip'))

#         (self.feed('HG8_res2_conv3',
#                    'HG8_res2_skip')
#          .add(name='HG8_res2'))
# # id:009 max-pooling
#         # (self.feed('HG8_pool1')
#         #      .max_pool(2, 2, 2, 2, name='HG8_pool2'))


# # res3
#         (self.feed('HG8_res2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res3_batch1')
#              .relu(name='HG8_res3_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG8_res3_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res3_batch2')
#              .relu(name='HG8_res3_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG8_res3_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res3_batch3')
#              .relu(name='HG8_res3_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res3_conv3'))

#         (self.feed('HG8_res2')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res3_skip'))

#         (self.feed('HG8_res3_conv3',
#                    'HG8_res3_skip')
#          .add(name='HG8_res3'))


# # pool2
#         (self.feed('HG8_res2')
#              .max_pool(2, 2, 2, 2, name='HG8_pool2'))


# # res4
#         (self.feed('HG8_pool2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res4_batch1')
#              .relu(name='HG8_res4_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG8_res4_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res4_batch2')
#              .relu(name='HG8_res4_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG8_res4_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res4_batch3')
#              .relu(name='HG8_res4_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res4_conv3'))

#         (self.feed('HG8_pool2')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res4_skip'))

#         (self.feed('HG8_res4_conv3',
#                    'HG8_res4_skip')
#          .add(name='HG8_res4'))
# # id:013 max-pooling
#         # (self.feed('HG8_pool3')
#         #      .max_pool(2, 2, 2, 2, name='HG8_pool4'))


# # res5
#         (self.feed('HG8_res4')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res5_batch1')
#              .relu(name='HG8_res5_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG8_res5_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res5_batch2')
#              .relu(name='HG8_res5_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG8_res5_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res5_batch3')
#              .relu(name='HG8_res5_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res5_conv3'))

#         (self.feed('HG8_res4')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res5_skip'))

#         (self.feed('HG8_res5_conv3',
#                    'HG8_res5_skip')
#          .add(name='HG8_res5'))


# # pool3
#         (self.feed('HG8_res4')
#              .max_pool(2, 2, 2, 2, name='HG8_pool3'))


# # res6
#         (self.feed('HG8_pool3')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res6_batch1')
#              .relu(name='HG8_res6_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG8_res6_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res6_batch2')
#              .relu(name='HG8_res6_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG8_res6_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res6_batch3')
#              .relu(name='HG8_res6_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res6_conv3'))

#         (self.feed('HG8_pool3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res6_skip'))

#         (self.feed('HG8_res6_conv3',
#                    'HG8_res6_skip')
#          .add(name='HG8_res6'))

# # res7
#         (self.feed('HG8_res6')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res7_batch1')
#              .relu(name='HG8_res7_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG8_res7_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res7_batch2')
#              .relu(name='HG8_res7_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG8_res7_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res7_batch3')
#              .relu(name='HG8_res7_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res7_conv3'))

#         (self.feed('HG8_res6')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res7_skip'))

#         (self.feed('HG8_res7_conv3',
#                    'HG8_res7_skip')
#          .add(name='HG8_res7'))


# # pool4
#         (self.feed('HG8_res6')
#              .max_pool(2, 2, 2, 2, name='HG8_pool4'))

# # res8
#         (self.feed('HG8_pool4')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res8_batch1')
#              .relu(name='HG8_res8_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG8_res8_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res8_batch2')
#              .relu(name='HG8_res8_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG8_res8_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res8_batch3')
#              .relu(name='HG8_res8_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res8_conv3'))

#         (self.feed('HG8_pool4')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res8_skip'))

#         (self.feed('HG8_res8_conv3',
#                    'HG8_res8_skip')
#          .add(name='HG8_res8'))

# # res9
#         (self.feed('HG8_res8')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res9_batch1')
#              .relu(name='HG8_res9_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG8_res9_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res9_batch2')
#              .relu(name='HG8_res9_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG8_res9_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res9_batch3')
#              .relu(name='HG8_res9_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res9_conv3'))

#         (self.feed('HG8_res8')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res9_skip'))

#         (self.feed('HG8_res9_conv3',
#                    'HG8_res9_skip')
#          .add(name='HG8_res9'))

# # res10
#         (self.feed('HG8_res9')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res10_batch1')
#              .relu(name='HG8_res10_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG8_res10_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res10_batch2')
#              .relu(name='HG8_res10_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG8_res10_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res10_batch3')
#              .relu(name='HG8_res10_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res10_conv3'))

#         (self.feed('HG8_res9')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res10_skip'))

#         (self.feed('HG8_res10_conv3',
#                    'HG8_res10_skip')
#          .add(name='HG8_res10'))


# # upsample1
#         (self.feed('HG8_res10')
#              .upsample(8, 8, name='HG8_upSample1'))

# # upsample1 + up1(Hg_res7)
#         (self.feed('HG8_upSample1',
#                    'HG8_res7')
#          .add(name='HG8_add1'))


# # res11
#         (self.feed('HG8_add1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res11_batch1')
#              .relu(name='HG8_res11_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG8_res11_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res11_batch2')
#              .relu(name='HG8_res11_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG8_res11_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res11_batch3')
#              .relu(name='HG8_res11_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res11_conv3'))

#         (self.feed('HG8_add1')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res11_skip'))

#         (self.feed('HG8_res11_conv3',
#                    'HG8_res11_skip')
#          .add(name='HG8_res11'))


# # upsample2
#         (self.feed('HG8_res11')
#              .upsample(16, 16, name='HG8_upSample2'))

# # upsample2 + up1(Hg_res5)
#         (self.feed('HG8_upSample2',
#                    'HG8_res5')
#          .add(name='HG8_add2'))


# # res12
#         (self.feed('HG8_add2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res12_batch1')
#              .relu(name='HG8_res12_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG8_res12_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res12_batch2')
#              .relu(name='HG8_res12_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG8_res12_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res12_batch3')
#              .relu(name='HG8_res12_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res12_conv3'))

#         (self.feed('HG8_add2')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res12_skip'))

#         (self.feed('HG8_res12_conv3',
#                    'HG8_res12_skip')
#          .add(name='HG8_res12'))


# # upsample3
#         (self.feed('HG8_res12')
#              .upsample(32, 32, name='HG8_upSample3'))

# # upsample3 + up1(Hg_res3)
#         (self.feed('HG8_upSample3',
#                    'HG8_res3')
#          .add(name='HG8_add3'))


# # res13
#         (self.feed('HG8_add3')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res13_batch1')
#              .relu(name='HG8_res13_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG8_res13_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res13_batch2')
#              .relu(name='HG8_res13_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG8_res13_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res13_batch3')
#              .relu(name='HG8_res13_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res13_conv3'))

#         (self.feed('HG8_add3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res13_skip'))

#         (self.feed('HG8_res13_conv3',
#                    'HG8_res13_skip')
#          .add(name='HG8_res13'))


# # upsample4
#         (self.feed('HG8_res13')
#              .upsample(64, 64, name='HG8_upSample4'))

# # upsample4 + up1(Hg_res1)
#         (self.feed('HG8_upSample4',
#                    'HG8_res1')
#          .add(name='HG8_add4'))

# ###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  Hourglass8  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

# ################################# Hourglass8 postprocess #################

# # id:025  Res14
#         (self.feed('HG8_add4')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res14_batch1')
#              .relu(name='HG8_res14_relu1')
#              .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG8_res14_conv1')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res14_batch2')
#              .relu(name='HG8_res14_relu2')
#              #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
#              .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG8_res14_conv2')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res14_batch3')
#              .relu(name='HG8_res14_relu3')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res14_conv3')

#          )

#         (self.feed('HG8_add4')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res14_skip'))

#         (self.feed('HG8_res14_conv3',
#                    'HG8_res14_skip')
#          .add(name='HG8_res14'))

# # Linear layer to produce first set of predictions
# # ll
#         (self.feed('HG8_res14')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_linearfunc_conv1', padding='SAME')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_linearfunc_batch1')
#              .relu(name='HG8_linearfunc_relu'))

# #************************************* Predicted heatmaps tmpOut ****************************#
# # tmpOut
#         (self.feed('HG8_linearfunc_relu')
#              .conv(1, 1, n_classes, 1, 1, biased=True, relu=False, name='HG8_Heatmap', padding='SAME'))

# ###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Hourglass8 postprocess ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###
