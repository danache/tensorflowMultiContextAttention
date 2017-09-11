# Converted to TensorFlow .caffemodel
# with the DeepLab-ResNet configuration.
# The batch normalisation layer is provided by
# the slim library
# (https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim).

from kaffe.tensorflow import Network
import tensorflow as tf
import numpy as np


class MultiContextAttentionModel(Network):

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































#######################################  Stack1  #########################################
                 ####################  Hourglass  #####################
# res1
        (self.feed('Res6')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack1_res1_batch1')
             .relu(name='Stack1_res1_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack1_res1_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack1_res1_batch2')
             .relu(name='Stack1_res1_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack1_res1_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack1_res1_batch3')
             .relu(name='Stack1_res1_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack1_res1_conv3'))

        (self.feed('Res6')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack1_res1_skip'))

        (self.feed('Stack1_res1_conv3',
                   'Stack1_res1_skip')
         .add(name='Stack1_res1'))

# resPool1
        (self.feed('Stack1_res1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack1_resPool1_batch1')
             .relu(name='Stack1_resPool1_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack1_resPool1_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack1_resPool1_batch2')
             .relu(name='Stack1_resPool1_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack1_resPool1_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack1_resPool1_batch3')
             .relu(name='Stack1_resPool1_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack1_resPool1_conv3'))

        (self.feed('Stack1_res1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack1_resPool1_skip'))

        (self.feed('Stack1_res1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack1_resPool1_batch4')
             .relu(name='Stack1_resPool1_relu4')
             .max_pool(2, 2, 2, 2, name='Stack1_resPool1_pool')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='Stack1_resPool1_conv4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack1_resPool1_batch5')
             .relu(name='Stack1_resPool1_relu5')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='Stack1_resPool1_conv5')
             .upsample(64, 64, name='Stack1_resPool1_upSample'))


        (self.feed('Stack1_resPool1_conv3',
                   'Stack1_resPool1_skip',
                   'Stack1_resPool1_upSample')
         .add(name='Stack1_resPool1'))



# resPool2
        (self.feed('Stack1_resPool1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack1_resPool2_batch1')
             .relu(name='Stack1_resPool2_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack1_resPool2_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack1_resPool2_batch2')
             .relu(name='Stack1_resPool2_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack1_resPool2_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack1_resPool2_batch3')
             .relu(name='Stack1_resPool2_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack1_resPool2_conv3'))

        (self.feed('Stack1_resPool1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack1_resPool2_skip'))

        (self.feed('Stack1_resPool1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack1_resPool2_batch4')
             .relu(name='Stack1_resPool2_relu4')
             .max_pool(2, 2, 2, 2, name='Stack1_resPool2_pool')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='Stack1_resPool2_conv4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack1_resPool2_batch5')
             .relu(name='Stack1_resPool2_relu5')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='Stack1_resPool2_conv5')
             .upsample(64, 64, name='Stack1_resPool2_upSample'))


        (self.feed('Stack1_resPool2_conv3',
                   'Stack1_resPool2_skip',
                   'Stack1_resPool2_upSample')
         .add(name='Stack1_resPool2'))

# pool1
        (self.feed('Res6')
             .max_pool(2, 2, 2, 2, name='Stack1_pool1'))


# res2
        (self.feed('Stack1_pool1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack1_res2_batch1')
             .relu(name='Stack1_res2_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack1_res2_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack1_res2_batch2')
             .relu(name='Stack1_res2_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack1_res2_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack1_res2_batch3')
             .relu(name='Stack1_res2_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack1_res2_conv3'))

        (self.feed('Stack1_pool1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack1_res2_skip'))

        (self.feed('Stack1_res2_conv3',
                   'Stack1_res2_skip')
         .add(name='Stack1_res2'))

# res3
        (self.feed('Stack1_res2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack1_res3_batch1')
             .relu(name='Stack1_res3_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack1_res3_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack1_res3_batch2')
             .relu(name='Stack1_res3_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack1_res3_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack1_res3_batch3')
             .relu(name='Stack1_res3_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack1_res3_conv3'))

        (self.feed('Stack1_res2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack1_res3_skip'))

        (self.feed('Stack1_res3_conv3',
                   'Stack1_res3_skip')
         .add(name='Stack1_res3'))

# resPool3
        (self.feed('Stack1_res3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack1_resPool3_batch1')
             .relu(name='Stack1_resPool3_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack1_resPool3_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack1_resPool3_batch2')
             .relu(name='Stack1_resPool3_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack1_resPool3_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack1_resPool3_batch3')
             .relu(name='Stack1_resPool3_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack1_resPool3_conv3'))

        (self.feed('Stack1_res3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack1_resPool3_skip'))

        (self.feed('Stack1_res3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack1_resPool3_batch4')
             .relu(name='Stack1_resPool3_relu4')
             .max_pool(2, 2, 2, 2, name='Stack1_resPool3_pool')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='Stack1_resPool3_conv4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack1_resPool3_batch5')
             .relu(name='Stack1_resPool3_relu5')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='Stack1_resPool3_conv5')
             .upsample(32, 32, name='Stack1_resPool3_upSample'))


        (self.feed('Stack1_resPool3_conv3',
                   'Stack1_resPool3_skip',
                   'Stack1_resPool3_upSample')
         .add(name='Stack1_resPool3'))




# pool2
        (self.feed('Stack1_res2')
             .max_pool(2, 2, 2, 2, name='Stack1_pool2'))


# res4
        (self.feed('Stack1_pool2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack1_res4_batch1')
             .relu(name='Stack1_res4_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack1_res4_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack1_res4_batch2')
             .relu(name='Stack1_res4_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack1_res4_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack1_res4_batch3')
             .relu(name='Stack1_res4_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack1_res4_conv3'))

        (self.feed('Stack1_pool2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack1_res4_skip'))

        (self.feed('Stack1_res4_conv3',
                   'Stack1_res4_skip')
         .add(name='Stack1_res4'))
# id:013 max-pooling
        # (self.feed('Stack1_pool3')
        #      .max_pool(2, 2, 2, 2, name='Stack1_pool4'))


# res5
        (self.feed('Stack1_res4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack1_res5_batch1')
             .relu(name='Stack1_res5_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack1_res5_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack1_res5_batch2')
             .relu(name='Stack1_res5_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack1_res5_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack1_res5_batch3')
             .relu(name='Stack1_res5_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack1_res5_conv3'))

        (self.feed('Stack1_res4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack1_res5_skip'))

        (self.feed('Stack1_res5_conv3',
                   'Stack1_res5_skip')
         .add(name='Stack1_res5'))


# pool3
        (self.feed('Stack1_res4')
             .max_pool(2, 2, 2, 2, name='Stack1_pool3'))


# res6
        (self.feed('Stack1_pool3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack1_res6_batch1')
             .relu(name='Stack1_res6_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack1_res6_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack1_res6_batch2')
             .relu(name='Stack1_res6_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack1_res6_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack1_res6_batch3')
             .relu(name='Stack1_res6_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack1_res6_conv3'))

        (self.feed('Stack1_pool3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack1_res6_skip'))

        (self.feed('Stack1_res6_conv3',
                   'Stack1_res6_skip')
         .add(name='Stack1_res6'))

# res7
        (self.feed('Stack1_res6')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack1_res7_batch1')
             .relu(name='Stack1_res7_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack1_res7_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack1_res7_batch2')
             .relu(name='Stack1_res7_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack1_res7_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack1_res7_batch3')
             .relu(name='Stack1_res7_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack1_res7_conv3'))

        (self.feed('Stack1_res6')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack1_res7_skip'))

        (self.feed('Stack1_res7_conv3',
                   'Stack1_res7_skip')
         .add(name='Stack1_res7'))


# pool4
        (self.feed('Stack1_res6')
             .max_pool(2, 2, 2, 2, name='Stack1_pool4'))

# res8
        (self.feed('Stack1_pool4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack1_res8_batch1')
             .relu(name='Stack1_res8_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack1_res8_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack1_res8_batch2')
             .relu(name='Stack1_res8_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack1_res8_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack1_res8_batch3')
             .relu(name='Stack1_res8_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack1_res8_conv3'))

        (self.feed('Stack1_pool4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack1_res8_skip'))

        (self.feed('Stack1_res8_conv3',
                   'Stack1_res8_skip')
         .add(name='Stack1_res8'))

# res9
        (self.feed('Stack1_res8')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack1_res9_batch1')
             .relu(name='Stack1_res9_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack1_res9_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack1_res9_batch2')
             .relu(name='Stack1_res9_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack1_res9_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack1_res9_batch3')
             .relu(name='Stack1_res9_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack1_res9_conv3'))

        (self.feed('Stack1_res8')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack1_res9_skip'))

        (self.feed('Stack1_res9_conv3',
                   'Stack1_res9_skip')
         .add(name='Stack1_res9'))

# res10
        (self.feed('Stack1_res9')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack1_res10_batch1')
             .relu(name='Stack1_res10_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack1_res10_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack1_res10_batch2')
             .relu(name='Stack1_res10_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack1_res10_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack1_res10_batch3')
             .relu(name='Stack1_res10_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack1_res10_conv3'))

        (self.feed('Stack1_res9')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack1_res10_skip'))

        (self.feed('Stack1_res10_conv3',
                   'Stack1_res10_skip')
         .add(name='Stack1_res10'))


# upsample1
        (self.feed('Stack1_res10')
             .upsample(8, 8, name='Stack1_upSample1'))

# upsample1 + up1(Hg_res7)
        (self.feed('Stack1_upSample1',
                   'Stack1_res7')
         .add(name='Stack1_add1'))


# res11
        (self.feed('Stack1_add1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack1_res11_batch1')
             .relu(name='Stack1_res11_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack1_res11_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack1_res11_batch2')
             .relu(name='Stack1_res11_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack1_res11_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack1_res11_batch3')
             .relu(name='Stack1_res11_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack1_res11_conv3'))

        (self.feed('Stack1_add1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack1_res11_skip'))

        (self.feed('Stack1_res11_conv3',
                   'Stack1_res11_skip')
         .add(name='Stack1_res11'))


# upsample2
        (self.feed('Stack1_res11')
             .upsample(16, 16, name='Stack1_upSample2'))

# upsample2 + up1(Hg_res5)
        (self.feed('Stack1_upSample2',
                   'Stack1_res5')
         .add(name='Stack1_add2'))


# res12
        (self.feed('Stack1_add2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack1_res12_batch1')
             .relu(name='Stack1_res12_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack1_res12_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack1_res12_batch2')
             .relu(name='Stack1_res12_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack1_res12_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack1_res12_batch3')
             .relu(name='Stack1_res12_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack1_res12_conv3'))

        (self.feed('Stack1_add2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack1_res12_skip'))

        (self.feed('Stack1_res12_conv3',
                   'Stack1_res12_skip')
         .add(name='Stack1_res12'))


# upsample3
        (self.feed('Stack1_res12')
             .upsample(32, 32, name='Stack1_upSample3'))

# upsample3 + Stack1_resPool3
        (self.feed('Stack1_upSample3',
                   'Stack1_resPool3')
         .add(name='Stack1_add3'))


# res13
        (self.feed('Stack1_add3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack1_res13_batch1')
             .relu(name='Stack1_res13_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack1_res13_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack1_res13_batch2')
             .relu(name='Stack1_res13_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack1_res13_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack1_res13_batch3')
             .relu(name='Stack1_res13_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack1_res13_conv3'))

        (self.feed('Stack1_add3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack1_res13_skip'))

        (self.feed('Stack1_res13_conv3',
                   'Stack1_res13_skip')
         .add(name='Stack1_res13'))


# upsample4
        (self.feed('Stack1_res13')
             .upsample(64, 64, name='Stack1_upSample4'))

# upsample4 + Stack1_resPool2
        (self.feed('Stack1_upSample4',
                   'Stack1_resPool2')
         .add(name='Stack1_add4'))

###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  Hourglass  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

################################# Stack1 postprocess #################
# Linear layer to produce first set of predictions
# ll1
        (self.feed('Stack1_add4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack1_linearfunc1_conv1', padding='SAME')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack1_linearfunc1_batch1')
             .relu(name='Stack1_linearfunc1_relu'))
# ll2
        (self.feed('Stack1_linearfunc1_relu')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack1_linearfunc2_conv1', padding='SAME')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack1_linearfunc2_batch1')
             .relu(name='Stack1_linearfunc2_relu'))

# att itersize=3
# att U input: ll2
        (self.feed('Stack1_linearfunc2_relu')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='Stack1_U', padding='SAME'))

        # self.Stack1_U = (self.feed('Stack1_linearfunc2_relu')
        #      .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='Stack1_U', padding='SAME'))
# att i=1 conv  C(1)
        # with tf.variable_scope('Stack1_share_param', reuse=False):
        #     (self.feed('Stack1_U')
        #          .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='Stack1_spConv1', padding='SAME'))

        with tf.variable_scope('Stack1_share_param', reuse=False):
            (self.feed('Stack1_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='Stack1_spConv1', padding='SAME'))
# att Qtemp
        (self.feed('Stack1_spConv1',
                   'Stack1_U')
         .add(name='Stack1_Qtemp1_add')
         .sigmoid(name='Stack1_Qtemp1'))
        # (self.add([self.Stack1_spConv1, self.Stack1_U], name = 'Stack1_Qtemp1_add')
        #     .sigmoid(name='Stack1_Qtemp1'))

# att i=2 conv  C(2)
        with tf.variable_scope('Stack1_share_param', reuse=True):
            (self.feed('Stack1_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False,name='Stack1_spConv1',  padding='SAME'))
# att Qtemp2
        (self.feed('Stack1_spConv1',
                   'Stack1_U')
         .add(name='Stack1_Qtemp2_add')
         .sigmoid(name='Stack1_Qtemp2'))
        # self.add([Stack1_spConv2, Stack1_U], name = 'Stack1_Qtemp2_add')
        # (self.feed('Stack1_Qtemp2_add')
        #     .sigmoid(name='Stack1_Qtemp2'))

# att i=3 conv  C(3)
        with tf.variable_scope('Stack1_share_param', reuse=True):
            (self.feed('Stack1_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='Stack1_spConv1' , padding='SAME'))
# att Qtemp
        (self.feed('Stack1_spConv1',
                   'Stack1_U')
         .add(name='Stack1_Qtemp3_add')
         .sigmoid(name='Stack1_Qtemp3'))
        # self.add([Stack1_spConv3, Stack1_U], name = 'Stack1_Qtemp3_add')
        # (self.feed('Stack1_Qtemp3_add')
        #     .sigmoid(name='Stack1_Qtemp3'))

# att pfeat
        (self.feed('Stack1_Qtemp3')
         .replicate(256, 3, name='Stack1_pfeat_replicate'))### dim =1?

        (self.feed('Stack1_linearfunc2_relu')
            .printLayer(name='printLayer1'))

        (self.feed('Stack1_pfeat_replicate')
            .printLayer(name='printLayer2'))

        (self.feed('Stack1_Qtemp3')
            .printLayer(name='printLayer3'))


        (self.feed('Stack1_linearfunc2_relu',
                   'Stack1_pfeat_replicate')
         .multiply2(name='Stack1_pfeat_multiply'))

# tmpOut
        (self.feed('Stack1_pfeat_multiply')
             .conv(1, 1, n_classes, 1, 1, biased=True, relu=False, name='Stack1_Heatmap', padding='SAME'))

###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Stack1 postprocess ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

############################# generate input for next Stack ##########

# outmap
        (self.feed('Stack1_Heatmap')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack1_outmap', padding='SAME'))
# ll3
        (self.feed('Stack1_linearfunc1_relu')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack1_linearfunc3_conv1', padding='SAME')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack1_linearfunc3_batch1')
             .relu(name='Stack1_linearfunc3_relu'))
# tmointer
        (self.feed('Res6',
                   'Stack1_outmap',
                   'Stack1_linearfunc3_relu')
         .add(name='Stack2_input'))

###^^^^^^^^^^^^^^^^^^^^^^^^^^^ generate input for next Stack ^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

























#######################################  Stack2  #########################################
                 ####################  Hourglass  #####################
# res1
        (self.feed('Stack2_input')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack2_res1_batch1')
             .relu(name='Stack2_res1_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack2_res1_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack2_res1_batch2')
             .relu(name='Stack2_res1_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack2_res1_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack2_res1_batch3')
             .relu(name='Stack2_res1_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack2_res1_conv3'))

        (self.feed('Stack2_input')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack2_res1_skip'))

        (self.feed('Stack2_res1_conv3',
                   'Stack2_res1_skip')
         .add(name='Stack2_res1'))

# resPool1
        (self.feed('Stack2_res1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack2_resPool1_batch1')
             .relu(name='Stack2_resPool1_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack2_resPool1_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack2_resPool1_batch2')
             .relu(name='Stack2_resPool1_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack2_resPool1_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack2_resPool1_batch3')
             .relu(name='Stack2_resPool1_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack2_resPool1_conv3'))

        (self.feed('Stack2_res1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack2_resPool1_skip'))

        (self.feed('Stack2_res1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack2_resPool1_batch4')
             .relu(name='Stack2_resPool1_relu4')
             .max_pool(2, 2, 2, 2, name='Stack2_resPool1_pool')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='Stack2_resPool1_conv4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack2_resPool1_batch5')
             .relu(name='Stack2_resPool1_relu5')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='Stack2_resPool1_conv5')
             .upsample(64, 64, name='Stack2_resPool1_upSample'))


        (self.feed('Stack2_resPool1_conv3',
                   'Stack2_resPool1_skip',
                   'Stack2_resPool1_upSample')
         .add(name='Stack2_resPool1'))



# resPool2
        (self.feed('Stack2_resPool1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack2_resPool2_batch1')
             .relu(name='Stack2_resPool2_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack2_resPool2_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack2_resPool2_batch2')
             .relu(name='Stack2_resPool2_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack2_resPool2_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack2_resPool2_batch3')
             .relu(name='Stack2_resPool2_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack2_resPool2_conv3'))

        (self.feed('Stack2_resPool1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack2_resPool2_skip'))

        (self.feed('Stack2_resPool1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack2_resPool2_batch4')
             .relu(name='Stack2_resPool2_relu4')
             .max_pool(2, 2, 2, 2, name='Stack2_resPool2_pool')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='Stack2_resPool2_conv4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack2_resPool2_batch5')
             .relu(name='Stack2_resPool2_relu5')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='Stack2_resPool2_conv5')
             .upsample(64, 64, name='Stack2_resPool2_upSample'))


        (self.feed('Stack2_resPool2_conv3',
                   'Stack2_resPool2_skip',
                   'Stack2_resPool2_upSample')
         .add(name='Stack2_resPool2'))

# pool1
        (self.feed('Stack2_input')
             .max_pool(2, 2, 2, 2, name='Stack2_pool1'))


# res2
        (self.feed('Stack2_pool1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack2_res2_batch1')
             .relu(name='Stack2_res2_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack2_res2_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack2_res2_batch2')
             .relu(name='Stack2_res2_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack2_res2_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack2_res2_batch3')
             .relu(name='Stack2_res2_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack2_res2_conv3'))

        (self.feed('Stack2_pool1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack2_res2_skip'))

        (self.feed('Stack2_res2_conv3',
                   'Stack2_res2_skip')
         .add(name='Stack2_res2'))

# res3
        (self.feed('Stack2_res2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack2_res3_batch1')
             .relu(name='Stack2_res3_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack2_res3_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack2_res3_batch2')
             .relu(name='Stack2_res3_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack2_res3_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack2_res3_batch3')
             .relu(name='Stack2_res3_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack2_res3_conv3'))

        (self.feed('Stack2_res2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack2_res3_skip'))

        (self.feed('Stack2_res3_conv3',
                   'Stack2_res3_skip')
         .add(name='Stack2_res3'))

# resPool3
        (self.feed('Stack2_res3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack2_resPool3_batch1')
             .relu(name='Stack2_resPool3_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack2_resPool3_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack2_resPool3_batch2')
             .relu(name='Stack2_resPool3_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack2_resPool3_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack2_resPool3_batch3')
             .relu(name='Stack2_resPool3_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack2_resPool3_conv3'))

        (self.feed('Stack2_res3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack2_resPool3_skip'))

        (self.feed('Stack2_res3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack2_resPool3_batch4')
             .relu(name='Stack2_resPool3_relu4')
             .max_pool(2, 2, 2, 2, name='Stack2_resPool3_pool')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='Stack2_resPool3_conv4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack2_resPool3_batch5')
             .relu(name='Stack2_resPool3_relu5')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='Stack2_resPool3_conv5')
             .upsample(32, 32, name='Stack2_resPool3_upSample'))


        (self.feed('Stack2_resPool3_conv3',
                   'Stack2_resPool3_skip',
                   'Stack2_resPool3_upSample')
         .add(name='Stack2_resPool3'))




# pool2
        (self.feed('Stack2_res2')
             .max_pool(2, 2, 2, 2, name='Stack2_pool2'))


# res4
        (self.feed('Stack2_pool2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack2_res4_batch1')
             .relu(name='Stack2_res4_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack2_res4_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack2_res4_batch2')
             .relu(name='Stack2_res4_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack2_res4_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack2_res4_batch3')
             .relu(name='Stack2_res4_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack2_res4_conv3'))

        (self.feed('Stack2_pool2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack2_res4_skip'))

        (self.feed('Stack2_res4_conv3',
                   'Stack2_res4_skip')
         .add(name='Stack2_res4'))
# id:013 max-pooling
        # (self.feed('Stack2_pool3')
        #      .max_pool(2, 2, 2, 2, name='Stack2_pool4'))


# res5
        (self.feed('Stack2_res4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack2_res5_batch1')
             .relu(name='Stack2_res5_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack2_res5_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack2_res5_batch2')
             .relu(name='Stack2_res5_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack2_res5_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack2_res5_batch3')
             .relu(name='Stack2_res5_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack2_res5_conv3'))

        (self.feed('Stack2_res4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack2_res5_skip'))

        (self.feed('Stack2_res5_conv3',
                   'Stack2_res5_skip')
         .add(name='Stack2_res5'))


# pool3
        (self.feed('Stack2_res4')
             .max_pool(2, 2, 2, 2, name='Stack2_pool3'))


# res6
        (self.feed('Stack2_pool3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack2_res6_batch1')
             .relu(name='Stack2_res6_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack2_res6_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack2_res6_batch2')
             .relu(name='Stack2_res6_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack2_res6_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack2_res6_batch3')
             .relu(name='Stack2_res6_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack2_res6_conv3'))

        (self.feed('Stack2_pool3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack2_res6_skip'))

        (self.feed('Stack2_res6_conv3',
                   'Stack2_res6_skip')
         .add(name='Stack2_res6'))

# res7
        (self.feed('Stack2_res6')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack2_res7_batch1')
             .relu(name='Stack2_res7_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack2_res7_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack2_res7_batch2')
             .relu(name='Stack2_res7_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack2_res7_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack2_res7_batch3')
             .relu(name='Stack2_res7_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack2_res7_conv3'))

        (self.feed('Stack2_res6')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack2_res7_skip'))

        (self.feed('Stack2_res7_conv3',
                   'Stack2_res7_skip')
         .add(name='Stack2_res7'))


# pool4
        (self.feed('Stack2_res6')
             .max_pool(2, 2, 2, 2, name='Stack2_pool4'))

# res8
        (self.feed('Stack2_pool4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack2_res8_batch1')
             .relu(name='Stack2_res8_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack2_res8_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack2_res8_batch2')
             .relu(name='Stack2_res8_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack2_res8_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack2_res8_batch3')
             .relu(name='Stack2_res8_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack2_res8_conv3'))

        (self.feed('Stack2_pool4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack2_res8_skip'))

        (self.feed('Stack2_res8_conv3',
                   'Stack2_res8_skip')
         .add(name='Stack2_res8'))

# res9
        (self.feed('Stack2_res8')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack2_res9_batch1')
             .relu(name='Stack2_res9_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack2_res9_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack2_res9_batch2')
             .relu(name='Stack2_res9_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack2_res9_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack2_res9_batch3')
             .relu(name='Stack2_res9_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack2_res9_conv3'))

        (self.feed('Stack2_res8')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack2_res9_skip'))

        (self.feed('Stack2_res9_conv3',
                   'Stack2_res9_skip')
         .add(name='Stack2_res9'))

# res10
        (self.feed('Stack2_res9')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack2_res10_batch1')
             .relu(name='Stack2_res10_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack2_res10_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack2_res10_batch2')
             .relu(name='Stack2_res10_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack2_res10_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack2_res10_batch3')
             .relu(name='Stack2_res10_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack2_res10_conv3'))

        (self.feed('Stack2_res9')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack2_res10_skip'))

        (self.feed('Stack2_res10_conv3',
                   'Stack2_res10_skip')
         .add(name='Stack2_res10'))


# upsample1
        (self.feed('Stack2_res10')
             .upsample(8, 8, name='Stack2_upSample1'))

# upsample1 + up1(Hg_res7)
        (self.feed('Stack2_upSample1',
                   'Stack2_res7')
         .add(name='Stack2_add1'))


# res11
        (self.feed('Stack2_add1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack2_res11_batch1')
             .relu(name='Stack2_res11_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack2_res11_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack2_res11_batch2')
             .relu(name='Stack2_res11_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack2_res11_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack2_res11_batch3')
             .relu(name='Stack2_res11_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack2_res11_conv3'))

        (self.feed('Stack2_add1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack2_res11_skip'))

        (self.feed('Stack2_res11_conv3',
                   'Stack2_res11_skip')
         .add(name='Stack2_res11'))


# upsample2
        (self.feed('Stack2_res11')
             .upsample(16, 16, name='Stack2_upSample2'))

# upsample2 + up1(Hg_res5)
        (self.feed('Stack2_upSample2',
                   'Stack2_res5')
         .add(name='Stack2_add2'))


# res12
        (self.feed('Stack2_add2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack2_res12_batch1')
             .relu(name='Stack2_res12_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack2_res12_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack2_res12_batch2')
             .relu(name='Stack2_res12_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack2_res12_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack2_res12_batch3')
             .relu(name='Stack2_res12_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack2_res12_conv3'))

        (self.feed('Stack2_add2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack2_res12_skip'))

        (self.feed('Stack2_res12_conv3',
                   'Stack2_res12_skip')
         .add(name='Stack2_res12'))


# upsample3
        (self.feed('Stack2_res12')
             .upsample(32, 32, name='Stack2_upSample3'))

# upsample3 + Stack2_resPool3
        (self.feed('Stack2_upSample3',
                   'Stack2_resPool3')
         .add(name='Stack2_add3'))


# res13
        (self.feed('Stack2_add3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack2_res13_batch1')
             .relu(name='Stack2_res13_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack2_res13_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack2_res13_batch2')
             .relu(name='Stack2_res13_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack2_res13_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack2_res13_batch3')
             .relu(name='Stack2_res13_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack2_res13_conv3'))

        (self.feed('Stack2_add3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack2_res13_skip'))

        (self.feed('Stack2_res13_conv3',
                   'Stack2_res13_skip')
         .add(name='Stack2_res13'))


# upsample4
        (self.feed('Stack2_res13')
             .upsample(64, 64, name='Stack2_upSample4'))

# upsample4 + Stack2_resPool2
        (self.feed('Stack2_upSample4',
                   'Stack2_resPool2')
         .add(name='Stack2_add4'))

###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  Hourglass  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

################################# Stack2 postprocess #################
# Linear layer to produce first set of predictions
# ll1
        (self.feed('Stack2_add4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack2_linearfunc1_conv1', padding='SAME')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack2_linearfunc1_batch1')
             .relu(name='Stack2_linearfunc1_relu'))
# ll2
        (self.feed('Stack2_linearfunc1_relu')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack2_linearfunc2_conv1', padding='SAME')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack2_linearfunc2_batch1')
             .relu(name='Stack2_linearfunc2_relu'))

# att itersize=3
# att U input: ll2
        (self.feed('Stack2_linearfunc2_relu')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='Stack2_U', padding='SAME'))
# att i=1 conv  C(1)
        with tf.variable_scope('Stack2_share_param', reuse=False):
            (self.feed('Stack2_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='Stack2_spConv1', padding='SAME'))
# att Qtemp
        (self.feed('Stack2_spConv1',
                   'Stack2_U')
         .add(name='Stack2_Qtemp1_add')
         .sigmoid(name='Stack2_Qtemp1'))

# att i=2 conv  C(2)
        with tf.variable_scope('Stack2_share_param', reuse=True):
            (self.feed('Stack2_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='Stack2_spConv1', padding='SAME'))
# att Qtemp2
        (self.feed('Stack2_spConv1',
                   'Stack2_U')
         .add(name='Stack2_Qtemp2_add')
         .sigmoid(name='Stack2_Qtemp2'))

# att i=3 conv  C(3)
        with tf.variable_scope('Stack2_share_param', reuse=True):
            (self.feed('Stack2_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='Stack2_spConv1', padding='SAME'))
# att Qtemp
        (self.feed('Stack2_spConv1',
                   'Stack2_U')
         .add(name='Stack2_Qtemp3_add')
         .sigmoid(name='Stack2_Qtemp3'))
# att pfeat
        (self.feed('Stack2_Qtemp3')
         .replicate(256, 3, name='Stack2_pfeat_replicate'))

        (self.feed('Stack2_linearfunc2_relu',
                   'Stack2_pfeat_replicate')
         .multiply2(name='Stack2_pfeat_multiply'))

# tmpOut
        (self.feed('Stack2_pfeat_multiply')
             .conv(1, 1, n_classes, 1, 1, biased=True, relu=False, name='Stack2_Heatmap', padding='SAME'))

###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Stack2 postprocess ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

############################# generate input for next Stack ##########

# outmap
        (self.feed('Stack2_Heatmap')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack2_outmap', padding='SAME'))
# ll3
        (self.feed('Stack2_linearfunc1_relu')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack2_linearfunc3_conv1', padding='SAME')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack2_linearfunc3_batch1')
             .relu(name='Stack2_linearfunc3_relu'))
# tmointer
        (self.feed('Stack2_input',
                   'Stack2_outmap',
                   'Stack2_linearfunc3_relu')
         .add(name='Stack3_input'))

###^^^^^^^^^^^^^^^^^^^^^^^^^^^ generate input for next Stack ^^^^^^^^^^^^^^^^^^^^^^^^^^^^###



































































































































































































































































#######################################  Stack3  #########################################
                 ####################  Hourglass  #####################
# res1
        (self.feed('Stack3_input')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack3_res1_batch1')
             .relu(name='Stack3_res1_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack3_res1_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack3_res1_batch2')
             .relu(name='Stack3_res1_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack3_res1_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack3_res1_batch3')
             .relu(name='Stack3_res1_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack3_res1_conv3'))

        (self.feed('Stack3_input')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack3_res1_skip'))

        (self.feed('Stack3_res1_conv3',
                   'Stack3_res1_skip')
         .add(name='Stack3_res1'))

# resPool1
        (self.feed('Stack3_res1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack3_resPool1_batch1')
             .relu(name='Stack3_resPool1_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack3_resPool1_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack3_resPool1_batch2')
             .relu(name='Stack3_resPool1_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack3_resPool1_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack3_resPool1_batch3')
             .relu(name='Stack3_resPool1_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack3_resPool1_conv3'))

        (self.feed('Stack3_res1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack3_resPool1_skip'))

        (self.feed('Stack3_res1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack3_resPool1_batch4')
             .relu(name='Stack3_resPool1_relu4')
             .max_pool(2, 2, 2, 2, name='Stack3_resPool1_pool')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='Stack3_resPool1_conv4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack3_resPool1_batch5')
             .relu(name='Stack3_resPool1_relu5')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='Stack3_resPool1_conv5')
             .upsample(64, 64, name='Stack3_resPool1_upSample'))


        (self.feed('Stack3_resPool1_conv3',
                   'Stack3_resPool1_skip',
                   'Stack3_resPool1_upSample')
         .add(name='Stack3_resPool1'))



# resPool2
        (self.feed('Stack3_resPool1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack3_resPool2_batch1')
             .relu(name='Stack3_resPool2_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack3_resPool2_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack3_resPool2_batch2')
             .relu(name='Stack3_resPool2_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack3_resPool2_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack3_resPool2_batch3')
             .relu(name='Stack3_resPool2_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack3_resPool2_conv3'))

        (self.feed('Stack3_resPool1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack3_resPool2_skip'))

        (self.feed('Stack3_resPool1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack3_resPool2_batch4')
             .relu(name='Stack3_resPool2_relu4')
             .max_pool(2, 2, 2, 2, name='Stack3_resPool2_pool')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='Stack3_resPool2_conv4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack3_resPool2_batch5')
             .relu(name='Stack3_resPool2_relu5')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='Stack3_resPool2_conv5')
             .upsample(64, 64, name='Stack3_resPool2_upSample'))


        (self.feed('Stack3_resPool2_conv3',
                   'Stack3_resPool2_skip',
                   'Stack3_resPool2_upSample')
         .add(name='Stack3_resPool2'))

# pool1
        (self.feed('Stack3_input')
             .max_pool(2, 2, 2, 2, name='Stack3_pool1'))


# res2
        (self.feed('Stack3_pool1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack3_res2_batch1')
             .relu(name='Stack3_res2_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack3_res2_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack3_res2_batch2')
             .relu(name='Stack3_res2_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack3_res2_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack3_res2_batch3')
             .relu(name='Stack3_res2_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack3_res2_conv3'))

        (self.feed('Stack3_pool1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack3_res2_skip'))

        (self.feed('Stack3_res2_conv3',
                   'Stack3_res2_skip')
         .add(name='Stack3_res2'))

# res3
        (self.feed('Stack3_res2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack3_res3_batch1')
             .relu(name='Stack3_res3_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack3_res3_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack3_res3_batch2')
             .relu(name='Stack3_res3_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack3_res3_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack3_res3_batch3')
             .relu(name='Stack3_res3_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack3_res3_conv3'))

        (self.feed('Stack3_res2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack3_res3_skip'))

        (self.feed('Stack3_res3_conv3',
                   'Stack3_res3_skip')
         .add(name='Stack3_res3'))

# resPool3
        (self.feed('Stack3_res3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack3_resPool3_batch1')
             .relu(name='Stack3_resPool3_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack3_resPool3_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack3_resPool3_batch2')
             .relu(name='Stack3_resPool3_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack3_resPool3_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack3_resPool3_batch3')
             .relu(name='Stack3_resPool3_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack3_resPool3_conv3'))

        (self.feed('Stack3_res3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack3_resPool3_skip'))

        (self.feed('Stack3_res3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack3_resPool3_batch4')
             .relu(name='Stack3_resPool3_relu4')
             .max_pool(2, 2, 2, 2, name='Stack3_resPool3_pool')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='Stack3_resPool3_conv4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack3_resPool3_batch5')
             .relu(name='Stack3_resPool3_relu5')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='Stack3_resPool3_conv5')
             .upsample(32, 32, name='Stack3_resPool3_upSample'))


        (self.feed('Stack3_resPool3_conv3',
                   'Stack3_resPool3_skip',
                   'Stack3_resPool3_upSample')
         .add(name='Stack3_resPool3'))




# pool2
        (self.feed('Stack3_res2')
             .max_pool(2, 2, 2, 2, name='Stack3_pool2'))


# res4
        (self.feed('Stack3_pool2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack3_res4_batch1')
             .relu(name='Stack3_res4_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack3_res4_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack3_res4_batch2')
             .relu(name='Stack3_res4_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack3_res4_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack3_res4_batch3')
             .relu(name='Stack3_res4_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack3_res4_conv3'))

        (self.feed('Stack3_pool2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack3_res4_skip'))

        (self.feed('Stack3_res4_conv3',
                   'Stack3_res4_skip')
         .add(name='Stack3_res4'))
# id:013 max-pooling
        # (self.feed('Stack3_pool3')
        #      .max_pool(2, 2, 2, 2, name='Stack3_pool4'))


# res5
        (self.feed('Stack3_res4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack3_res5_batch1')
             .relu(name='Stack3_res5_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack3_res5_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack3_res5_batch2')
             .relu(name='Stack3_res5_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack3_res5_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack3_res5_batch3')
             .relu(name='Stack3_res5_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack3_res5_conv3'))

        (self.feed('Stack3_res4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack3_res5_skip'))

        (self.feed('Stack3_res5_conv3',
                   'Stack3_res5_skip')
         .add(name='Stack3_res5'))


# pool3
        (self.feed('Stack3_res4')
             .max_pool(2, 2, 2, 2, name='Stack3_pool3'))


# res6
        (self.feed('Stack3_pool3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack3_res6_batch1')
             .relu(name='Stack3_res6_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack3_res6_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack3_res6_batch2')
             .relu(name='Stack3_res6_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack3_res6_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack3_res6_batch3')
             .relu(name='Stack3_res6_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack3_res6_conv3'))

        (self.feed('Stack3_pool3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack3_res6_skip'))

        (self.feed('Stack3_res6_conv3',
                   'Stack3_res6_skip')
         .add(name='Stack3_res6'))

# res7
        (self.feed('Stack3_res6')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack3_res7_batch1')
             .relu(name='Stack3_res7_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack3_res7_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack3_res7_batch2')
             .relu(name='Stack3_res7_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack3_res7_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack3_res7_batch3')
             .relu(name='Stack3_res7_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack3_res7_conv3'))

        (self.feed('Stack3_res6')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack3_res7_skip'))

        (self.feed('Stack3_res7_conv3',
                   'Stack3_res7_skip')
         .add(name='Stack3_res7'))


# pool4
        (self.feed('Stack3_res6')
             .max_pool(2, 2, 2, 2, name='Stack3_pool4'))

# res8
        (self.feed('Stack3_pool4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack3_res8_batch1')
             .relu(name='Stack3_res8_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack3_res8_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack3_res8_batch2')
             .relu(name='Stack3_res8_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack3_res8_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack3_res8_batch3')
             .relu(name='Stack3_res8_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack3_res8_conv3'))

        (self.feed('Stack3_pool4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack3_res8_skip'))

        (self.feed('Stack3_res8_conv3',
                   'Stack3_res8_skip')
         .add(name='Stack3_res8'))

# res9
        (self.feed('Stack3_res8')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack3_res9_batch1')
             .relu(name='Stack3_res9_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack3_res9_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack3_res9_batch2')
             .relu(name='Stack3_res9_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack3_res9_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack3_res9_batch3')
             .relu(name='Stack3_res9_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack3_res9_conv3'))

        (self.feed('Stack3_res8')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack3_res9_skip'))

        (self.feed('Stack3_res9_conv3',
                   'Stack3_res9_skip')
         .add(name='Stack3_res9'))

# res10
        (self.feed('Stack3_res9')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack3_res10_batch1')
             .relu(name='Stack3_res10_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack3_res10_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack3_res10_batch2')
             .relu(name='Stack3_res10_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack3_res10_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack3_res10_batch3')
             .relu(name='Stack3_res10_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack3_res10_conv3'))

        (self.feed('Stack3_res9')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack3_res10_skip'))

        (self.feed('Stack3_res10_conv3',
                   'Stack3_res10_skip')
         .add(name='Stack3_res10'))


# upsample1
        (self.feed('Stack3_res10')
             .upsample(8, 8, name='Stack3_upSample1'))

# upsample1 + up1(Hg_res7)
        (self.feed('Stack3_upSample1',
                   'Stack3_res7')
         .add(name='Stack3_add1'))


# res11
        (self.feed('Stack3_add1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack3_res11_batch1')
             .relu(name='Stack3_res11_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack3_res11_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack3_res11_batch2')
             .relu(name='Stack3_res11_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack3_res11_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack3_res11_batch3')
             .relu(name='Stack3_res11_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack3_res11_conv3'))

        (self.feed('Stack3_add1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack3_res11_skip'))

        (self.feed('Stack3_res11_conv3',
                   'Stack3_res11_skip')
         .add(name='Stack3_res11'))


# upsample2
        (self.feed('Stack3_res11')
             .upsample(16, 16, name='Stack3_upSample2'))

# upsample2 + up1(Hg_res5)
        (self.feed('Stack3_upSample2',
                   'Stack3_res5')
         .add(name='Stack3_add2'))


# res12
        (self.feed('Stack3_add2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack3_res12_batch1')
             .relu(name='Stack3_res12_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack3_res12_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack3_res12_batch2')
             .relu(name='Stack3_res12_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack3_res12_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack3_res12_batch3')
             .relu(name='Stack3_res12_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack3_res12_conv3'))

        (self.feed('Stack3_add2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack3_res12_skip'))

        (self.feed('Stack3_res12_conv3',
                   'Stack3_res12_skip')
         .add(name='Stack3_res12'))


# upsample3
        (self.feed('Stack3_res12')
             .upsample(32, 32, name='Stack3_upSample3'))

# upsample3 + Stack3_resPool3
        (self.feed('Stack3_upSample3',
                   'Stack3_resPool3')
         .add(name='Stack3_add3'))


# res13
        (self.feed('Stack3_add3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack3_res13_batch1')
             .relu(name='Stack3_res13_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack3_res13_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack3_res13_batch2')
             .relu(name='Stack3_res13_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack3_res13_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack3_res13_batch3')
             .relu(name='Stack3_res13_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack3_res13_conv3'))

        (self.feed('Stack3_add3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack3_res13_skip'))

        (self.feed('Stack3_res13_conv3',
                   'Stack3_res13_skip')
         .add(name='Stack3_res13'))


# upsample4
        (self.feed('Stack3_res13')
             .upsample(64, 64, name='Stack3_upSample4'))

# upsample4 + Stack3_resPool2
        (self.feed('Stack3_upSample4',
                   'Stack3_resPool2')
         .add(name='Stack3_add4'))

###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  Hourglass  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

################################# Stack3 postprocess #################
# Linear layer to produce first set of predictions
# ll1
        (self.feed('Stack3_add4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack3_linearfunc1_conv1', padding='SAME')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack3_linearfunc1_batch1')
             .relu(name='Stack3_linearfunc1_relu'))
# ll2
        (self.feed('Stack3_linearfunc1_relu')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack3_linearfunc2_conv1', padding='SAME')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack3_linearfunc2_batch1')
             .relu(name='Stack3_linearfunc2_relu'))

# att itersize=3
# att U input: ll2
        (self.feed('Stack3_linearfunc2_relu')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='Stack3_U', padding='SAME'))
# att i=1 conv  C(1)
        with tf.variable_scope('Stack3_share_param', reuse=False):
            (self.feed('Stack3_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='Stack3_spConv1', padding='SAME'))
# att Qtemp
        (self.feed('Stack3_spConv1',
                   'Stack3_U')
         .add(name='Stack3_Qtemp1_add')
         .sigmoid(name='Stack3_Qtemp1'))

# att i=2 conv  C(2)
        with tf.variable_scope('Stack3_share_param', reuse=True):
            (self.feed('Stack3_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='Stack3_spConv1', padding='SAME'))
# att Qtemp2
        (self.feed('Stack3_spConv1',
                   'Stack3_U')
         .add(name='Stack3_Qtemp2_add')
         .sigmoid(name='Stack3_Qtemp2'))

# att i=3 conv  C(3)
        with tf.variable_scope('Stack3_share_param', reuse=True):
            (self.feed('Stack3_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='Stack3_spConv1', padding='SAME'))
# att Qtemp
        (self.feed('Stack3_spConv1',
                   'Stack3_U')
         .add(name='Stack3_Qtemp3_add')
         .sigmoid(name='Stack3_Qtemp3'))
# att pfeat
        (self.feed('Stack3_Qtemp3')
         .replicate(256, 3, name='Stack3_pfeat_replicate'))

        (self.feed('Stack3_linearfunc2_relu',
                   'Stack3_pfeat_replicate')
         .multiply2(name='Stack3_pfeat_multiply'))

# tmpOut
        (self.feed('Stack3_pfeat_multiply')
             .conv(1, 1, n_classes, 1, 1, biased=True, relu=False, name='Stack3_Heatmap', padding='SAME'))

###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Stack3 postprocess ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

############################# generate input for next Stack ##########

# outmap
        (self.feed('Stack3_Heatmap')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack3_outmap', padding='SAME'))
# ll3
        (self.feed('Stack3_linearfunc1_relu')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack3_linearfunc3_conv1', padding='SAME')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack3_linearfunc3_batch1')
             .relu(name='Stack3_linearfunc3_relu'))
# tmointer
        (self.feed('Stack3_input',
                   'Stack3_outmap',
                   'Stack3_linearfunc3_relu')
         .add(name='Stack4_input'))

###^^^^^^^^^^^^^^^^^^^^^^^^^^^ generate input for next Stack ^^^^^^^^^^^^^^^^^^^^^^^^^^^^###



























































#######################################  Stack4  #########################################
                 ####################  Hourglass  #####################
# res1
        (self.feed('Stack4_input')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack4_res1_batch1')
             .relu(name='Stack4_res1_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack4_res1_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack4_res1_batch2')
             .relu(name='Stack4_res1_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack4_res1_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack4_res1_batch3')
             .relu(name='Stack4_res1_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack4_res1_conv3'))

        (self.feed('Stack4_input')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack4_res1_skip'))

        (self.feed('Stack4_res1_conv3',
                   'Stack4_res1_skip')
         .add(name='Stack4_res1'))

# resPool1
        (self.feed('Stack4_res1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack4_resPool1_batch1')
             .relu(name='Stack4_resPool1_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack4_resPool1_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack4_resPool1_batch2')
             .relu(name='Stack4_resPool1_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack4_resPool1_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack4_resPool1_batch3')
             .relu(name='Stack4_resPool1_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack4_resPool1_conv3'))

        (self.feed('Stack4_res1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack4_resPool1_skip'))

        (self.feed('Stack4_res1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack4_resPool1_batch4')
             .relu(name='Stack4_resPool1_relu4')
             .max_pool(2, 2, 2, 2, name='Stack4_resPool1_pool')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='Stack4_resPool1_conv4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack4_resPool1_batch5')
             .relu(name='Stack4_resPool1_relu5')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='Stack4_resPool1_conv5')
             .upsample(64, 64, name='Stack4_resPool1_upSample'))


        (self.feed('Stack4_resPool1_conv3',
                   'Stack4_resPool1_skip',
                   'Stack4_resPool1_upSample')
         .add(name='Stack4_resPool1'))



# resPool2
        (self.feed('Stack4_resPool1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack4_resPool2_batch1')
             .relu(name='Stack4_resPool2_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack4_resPool2_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack4_resPool2_batch2')
             .relu(name='Stack4_resPool2_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack4_resPool2_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack4_resPool2_batch3')
             .relu(name='Stack4_resPool2_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack4_resPool2_conv3'))

        (self.feed('Stack4_resPool1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack4_resPool2_skip'))

        (self.feed('Stack4_resPool1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack4_resPool2_batch4')
             .relu(name='Stack4_resPool2_relu4')
             .max_pool(2, 2, 2, 2, name='Stack4_resPool2_pool')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='Stack4_resPool2_conv4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack4_resPool2_batch5')
             .relu(name='Stack4_resPool2_relu5')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='Stack4_resPool2_conv5')
             .upsample(64, 64, name='Stack4_resPool2_upSample'))


        (self.feed('Stack4_resPool2_conv3',
                   'Stack4_resPool2_skip',
                   'Stack4_resPool2_upSample')
         .add(name='Stack4_resPool2'))

# pool1
        (self.feed('Stack4_input')
             .max_pool(2, 2, 2, 2, name='Stack4_pool1'))


# res2
        (self.feed('Stack4_pool1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack4_res2_batch1')
             .relu(name='Stack4_res2_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack4_res2_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack4_res2_batch2')
             .relu(name='Stack4_res2_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack4_res2_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack4_res2_batch3')
             .relu(name='Stack4_res2_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack4_res2_conv3'))

        (self.feed('Stack4_pool1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack4_res2_skip'))

        (self.feed('Stack4_res2_conv3',
                   'Stack4_res2_skip')
         .add(name='Stack4_res2'))

# res3
        (self.feed('Stack4_res2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack4_res3_batch1')
             .relu(name='Stack4_res3_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack4_res3_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack4_res3_batch2')
             .relu(name='Stack4_res3_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack4_res3_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack4_res3_batch3')
             .relu(name='Stack4_res3_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack4_res3_conv3'))

        (self.feed('Stack4_res2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack4_res3_skip'))

        (self.feed('Stack4_res3_conv3',
                   'Stack4_res3_skip')
         .add(name='Stack4_res3'))

# resPool3
        (self.feed('Stack4_res3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack4_resPool3_batch1')
             .relu(name='Stack4_resPool3_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack4_resPool3_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack4_resPool3_batch2')
             .relu(name='Stack4_resPool3_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack4_resPool3_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack4_resPool3_batch3')
             .relu(name='Stack4_resPool3_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack4_resPool3_conv3'))

        (self.feed('Stack4_res3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack4_resPool3_skip'))

        (self.feed('Stack4_res3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack4_resPool3_batch4')
             .relu(name='Stack4_resPool3_relu4')
             .max_pool(2, 2, 2, 2, name='Stack4_resPool3_pool')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='Stack4_resPool3_conv4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack4_resPool3_batch5')
             .relu(name='Stack4_resPool3_relu5')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='Stack4_resPool3_conv5')
             .upsample(32, 32, name='Stack4_resPool3_upSample'))


        (self.feed('Stack4_resPool3_conv3',
                   'Stack4_resPool3_skip',
                   'Stack4_resPool3_upSample')
         .add(name='Stack4_resPool3'))




# pool2
        (self.feed('Stack4_res2')
             .max_pool(2, 2, 2, 2, name='Stack4_pool2'))


# res4
        (self.feed('Stack4_pool2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack4_res4_batch1')
             .relu(name='Stack4_res4_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack4_res4_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack4_res4_batch2')
             .relu(name='Stack4_res4_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack4_res4_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack4_res4_batch3')
             .relu(name='Stack4_res4_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack4_res4_conv3'))

        (self.feed('Stack4_pool2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack4_res4_skip'))

        (self.feed('Stack4_res4_conv3',
                   'Stack4_res4_skip')
         .add(name='Stack4_res4'))
# id:013 max-pooling
        # (self.feed('Stack4_pool3')
        #      .max_pool(2, 2, 2, 2, name='Stack4_pool4'))


# res5
        (self.feed('Stack4_res4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack4_res5_batch1')
             .relu(name='Stack4_res5_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack4_res5_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack4_res5_batch2')
             .relu(name='Stack4_res5_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack4_res5_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack4_res5_batch3')
             .relu(name='Stack4_res5_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack4_res5_conv3'))

        (self.feed('Stack4_res4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack4_res5_skip'))

        (self.feed('Stack4_res5_conv3',
                   'Stack4_res5_skip')
         .add(name='Stack4_res5'))


# pool3
        (self.feed('Stack4_res4')
             .max_pool(2, 2, 2, 2, name='Stack4_pool3'))


# res6
        (self.feed('Stack4_pool3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack4_res6_batch1')
             .relu(name='Stack4_res6_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack4_res6_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack4_res6_batch2')
             .relu(name='Stack4_res6_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack4_res6_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack4_res6_batch3')
             .relu(name='Stack4_res6_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack4_res6_conv3'))

        (self.feed('Stack4_pool3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack4_res6_skip'))

        (self.feed('Stack4_res6_conv3',
                   'Stack4_res6_skip')
         .add(name='Stack4_res6'))

# res7
        (self.feed('Stack4_res6')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack4_res7_batch1')
             .relu(name='Stack4_res7_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack4_res7_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack4_res7_batch2')
             .relu(name='Stack4_res7_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack4_res7_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack4_res7_batch3')
             .relu(name='Stack4_res7_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack4_res7_conv3'))

        (self.feed('Stack4_res6')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack4_res7_skip'))

        (self.feed('Stack4_res7_conv3',
                   'Stack4_res7_skip')
         .add(name='Stack4_res7'))


# pool4
        (self.feed('Stack4_res6')
             .max_pool(2, 2, 2, 2, name='Stack4_pool4'))

# res8
        (self.feed('Stack4_pool4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack4_res8_batch1')
             .relu(name='Stack4_res8_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack4_res8_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack4_res8_batch2')
             .relu(name='Stack4_res8_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack4_res8_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack4_res8_batch3')
             .relu(name='Stack4_res8_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack4_res8_conv3'))

        (self.feed('Stack4_pool4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack4_res8_skip'))

        (self.feed('Stack4_res8_conv3',
                   'Stack4_res8_skip')
         .add(name='Stack4_res8'))

# res9
        (self.feed('Stack4_res8')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack4_res9_batch1')
             .relu(name='Stack4_res9_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack4_res9_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack4_res9_batch2')
             .relu(name='Stack4_res9_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack4_res9_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack4_res9_batch3')
             .relu(name='Stack4_res9_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack4_res9_conv3'))

        (self.feed('Stack4_res8')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack4_res9_skip'))

        (self.feed('Stack4_res9_conv3',
                   'Stack4_res9_skip')
         .add(name='Stack4_res9'))

# res10
        (self.feed('Stack4_res9')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack4_res10_batch1')
             .relu(name='Stack4_res10_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack4_res10_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack4_res10_batch2')
             .relu(name='Stack4_res10_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack4_res10_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack4_res10_batch3')
             .relu(name='Stack4_res10_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack4_res10_conv3'))

        (self.feed('Stack4_res9')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack4_res10_skip'))

        (self.feed('Stack4_res10_conv3',
                   'Stack4_res10_skip')
         .add(name='Stack4_res10'))


# upsample1
        (self.feed('Stack4_res10')
             .upsample(8, 8, name='Stack4_upSample1'))

# upsample1 + up1(Hg_res7)
        (self.feed('Stack4_upSample1',
                   'Stack4_res7')
         .add(name='Stack4_add1'))


# res11
        (self.feed('Stack4_add1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack4_res11_batch1')
             .relu(name='Stack4_res11_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack4_res11_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack4_res11_batch2')
             .relu(name='Stack4_res11_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack4_res11_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack4_res11_batch3')
             .relu(name='Stack4_res11_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack4_res11_conv3'))

        (self.feed('Stack4_add1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack4_res11_skip'))

        (self.feed('Stack4_res11_conv3',
                   'Stack4_res11_skip')
         .add(name='Stack4_res11'))


# upsample2
        (self.feed('Stack4_res11')
             .upsample(16, 16, name='Stack4_upSample2'))

# upsample2 + up1(Hg_res5)
        (self.feed('Stack4_upSample2',
                   'Stack4_res5')
         .add(name='Stack4_add2'))


# res12
        (self.feed('Stack4_add2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack4_res12_batch1')
             .relu(name='Stack4_res12_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack4_res12_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack4_res12_batch2')
             .relu(name='Stack4_res12_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack4_res12_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack4_res12_batch3')
             .relu(name='Stack4_res12_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack4_res12_conv3'))

        (self.feed('Stack4_add2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack4_res12_skip'))

        (self.feed('Stack4_res12_conv3',
                   'Stack4_res12_skip')
         .add(name='Stack4_res12'))


# upsample3
        (self.feed('Stack4_res12')
             .upsample(32, 32, name='Stack4_upSample3'))

# upsample3 + Stack4_resPool3
        (self.feed('Stack4_upSample3',
                   'Stack4_resPool3')
         .add(name='Stack4_add3'))


# res13
        (self.feed('Stack4_add3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack4_res13_batch1')
             .relu(name='Stack4_res13_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack4_res13_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack4_res13_batch2')
             .relu(name='Stack4_res13_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack4_res13_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack4_res13_batch3')
             .relu(name='Stack4_res13_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack4_res13_conv3'))

        (self.feed('Stack4_add3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack4_res13_skip'))

        (self.feed('Stack4_res13_conv3',
                   'Stack4_res13_skip')
         .add(name='Stack4_res13'))


# upsample4
        (self.feed('Stack4_res13')
             .upsample(64, 64, name='Stack4_upSample4'))

# upsample4 + Stack4_resPool2
        (self.feed('Stack4_upSample4',
                   'Stack4_resPool2')
         .add(name='Stack4_add4'))

###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  Hourglass  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

################################# Stack4 postprocess #################
# Linear layer to produce first set of predictions
# ll1
        (self.feed('Stack4_add4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack4_linearfunc1_conv1', padding='SAME')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack4_linearfunc1_batch1')
             .relu(name='Stack4_linearfunc1_relu'))
# ll2
        (self.feed('Stack4_linearfunc1_relu')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack4_linearfunc2_conv1', padding='SAME')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack4_linearfunc2_batch1')
             .relu(name='Stack4_linearfunc2_relu'))

# att itersize=3
# att U input: ll2
        (self.feed('Stack4_linearfunc2_relu')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='Stack4_U', padding='SAME'))
# att i=1 conv  C(1)
        with tf.variable_scope('Stack4_share_param', reuse=False):
            (self.feed('Stack4_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='Stack4_spConv1', padding='SAME'))
# att Qtemp
        (self.feed('Stack4_spConv1',
                   'Stack4_U')
         .add(name='Stack4_Qtemp1_add')
         .sigmoid(name='Stack4_Qtemp1'))

# att i=2 conv  C(2)
        with tf.variable_scope('Stack4_share_param', reuse=True):
            (self.feed('Stack4_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='Stack4_spConv1', padding='SAME'))
# att Qtemp2
        (self.feed('Stack4_spConv1',
                   'Stack4_U')
         .add(name='Stack4_Qtemp2_add')
         .sigmoid(name='Stack4_Qtemp2'))

# att i=3 conv  C(3)
        with tf.variable_scope('Stack4_share_param', reuse=True):
            (self.feed('Stack4_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='Stack4_spConv1', padding='SAME'))
# att Qtemp
        (self.feed('Stack4_spConv1',
                   'Stack4_U')
         .add(name='Stack4_Qtemp3_add')
         .sigmoid(name='Stack4_Qtemp3'))
# att pfeat
        (self.feed('Stack4_Qtemp3')
         .replicate(256, 3, name='Stack4_pfeat_replicate'))

        (self.feed('Stack4_linearfunc2_relu',
                   'Stack4_pfeat_replicate')
         .multiply2(name='Stack4_pfeat_multiply'))

# tmpOut
        (self.feed('Stack4_pfeat_multiply')
             .conv(1, 1, n_classes, 1, 1, biased=True, relu=False, name='Stack4_Heatmap', padding='SAME'))

###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Stack4 postprocess ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

############################# generate input for next Stack ##########

# outmap
        (self.feed('Stack4_Heatmap')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack4_outmap', padding='SAME'))
# ll3
        (self.feed('Stack4_linearfunc1_relu')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack4_linearfunc3_conv1', padding='SAME')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack4_linearfunc3_batch1')
             .relu(name='Stack4_linearfunc3_relu'))
# tmointer
        (self.feed('Stack4_input',
                   'Stack4_outmap',
                   'Stack4_linearfunc3_relu')
         .add(name='Stack5_input'))

###^^^^^^^^^^^^^^^^^^^^^^^^^^^ generate input for next Stack ^^^^^^^^^^^^^^^^^^^^^^^^^^^^###


























































































#######################################  Stack5  #########################################
                 ####################  Hourglass  #####################
# res1
        (self.feed('Stack5_input')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack5_res1_batch1')
             .relu(name='Stack5_res1_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack5_res1_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack5_res1_batch2')
             .relu(name='Stack5_res1_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack5_res1_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack5_res1_batch3')
             .relu(name='Stack5_res1_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack5_res1_conv3'))

        (self.feed('Stack5_input')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack5_res1_skip'))

        (self.feed('Stack5_res1_conv3',
                   'Stack5_res1_skip')
         .add(name='Stack5_res1'))

# resPool1
        (self.feed('Stack5_res1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack5_resPool1_batch1')
             .relu(name='Stack5_resPool1_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack5_resPool1_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack5_resPool1_batch2')
             .relu(name='Stack5_resPool1_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack5_resPool1_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack5_resPool1_batch3')
             .relu(name='Stack5_resPool1_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack5_resPool1_conv3'))

        (self.feed('Stack5_res1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack5_resPool1_skip'))

        (self.feed('Stack5_res1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack5_resPool1_batch4')
             .relu(name='Stack5_resPool1_relu4')
             .max_pool(2, 2, 2, 2, name='Stack5_resPool1_pool')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='Stack5_resPool1_conv4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack5_resPool1_batch5')
             .relu(name='Stack5_resPool1_relu5')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='Stack5_resPool1_conv5')
             .upsample(64, 64, name='Stack5_resPool1_upSample'))


        (self.feed('Stack5_resPool1_conv3',
                   'Stack5_resPool1_skip',
                   'Stack5_resPool1_upSample')
         .add(name='Stack5_resPool1'))



# resPool2
        (self.feed('Stack5_resPool1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack5_resPool2_batch1')
             .relu(name='Stack5_resPool2_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack5_resPool2_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack5_resPool2_batch2')
             .relu(name='Stack5_resPool2_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack5_resPool2_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack5_resPool2_batch3')
             .relu(name='Stack5_resPool2_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack5_resPool2_conv3'))

        (self.feed('Stack5_resPool1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack5_resPool2_skip'))

        (self.feed('Stack5_resPool1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack5_resPool2_batch4')
             .relu(name='Stack5_resPool2_relu4')
             .max_pool(2, 2, 2, 2, name='Stack5_resPool2_pool')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='Stack5_resPool2_conv4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack5_resPool2_batch5')
             .relu(name='Stack5_resPool2_relu5')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='Stack5_resPool2_conv5')
             .upsample(64, 64, name='Stack5_resPool2_upSample'))


        (self.feed('Stack5_resPool2_conv3',
                   'Stack5_resPool2_skip',
                   'Stack5_resPool2_upSample')
         .add(name='Stack5_resPool2'))

# pool1
        (self.feed('Stack5_input')
             .max_pool(2, 2, 2, 2, name='Stack5_pool1'))


# res2
        (self.feed('Stack5_pool1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack5_res2_batch1')
             .relu(name='Stack5_res2_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack5_res2_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack5_res2_batch2')
             .relu(name='Stack5_res2_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack5_res2_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack5_res2_batch3')
             .relu(name='Stack5_res2_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack5_res2_conv3'))

        (self.feed('Stack5_pool1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack5_res2_skip'))

        (self.feed('Stack5_res2_conv3',
                   'Stack5_res2_skip')
         .add(name='Stack5_res2'))

# res3
        (self.feed('Stack5_res2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack5_res3_batch1')
             .relu(name='Stack5_res3_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack5_res3_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack5_res3_batch2')
             .relu(name='Stack5_res3_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack5_res3_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack5_res3_batch3')
             .relu(name='Stack5_res3_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack5_res3_conv3'))

        (self.feed('Stack5_res2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack5_res3_skip'))

        (self.feed('Stack5_res3_conv3',
                   'Stack5_res3_skip')
         .add(name='Stack5_res3'))

# resPool3
        (self.feed('Stack5_res3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack5_resPool3_batch1')
             .relu(name='Stack5_resPool3_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack5_resPool3_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack5_resPool3_batch2')
             .relu(name='Stack5_resPool3_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack5_resPool3_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack5_resPool3_batch3')
             .relu(name='Stack5_resPool3_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack5_resPool3_conv3'))

        (self.feed('Stack5_res3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack5_resPool3_skip'))

        (self.feed('Stack5_res3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack5_resPool3_batch4')
             .relu(name='Stack5_resPool3_relu4')
             .max_pool(2, 2, 2, 2, name='Stack5_resPool3_pool')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='Stack5_resPool3_conv4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack5_resPool3_batch5')
             .relu(name='Stack5_resPool3_relu5')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='Stack5_resPool3_conv5')
             .upsample(32, 32, name='Stack5_resPool3_upSample'))


        (self.feed('Stack5_resPool3_conv3',
                   'Stack5_resPool3_skip',
                   'Stack5_resPool3_upSample')
         .add(name='Stack5_resPool3'))




# pool2
        (self.feed('Stack5_res2')
             .max_pool(2, 2, 2, 2, name='Stack5_pool2'))


# res4
        (self.feed('Stack5_pool2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack5_res4_batch1')
             .relu(name='Stack5_res4_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack5_res4_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack5_res4_batch2')
             .relu(name='Stack5_res4_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack5_res4_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack5_res4_batch3')
             .relu(name='Stack5_res4_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack5_res4_conv3'))

        (self.feed('Stack5_pool2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack5_res4_skip'))

        (self.feed('Stack5_res4_conv3',
                   'Stack5_res4_skip')
         .add(name='Stack5_res4'))
# id:013 max-pooling
        # (self.feed('Stack5_pool3')
        #      .max_pool(2, 2, 2, 2, name='Stack5_pool4'))


# res5
        (self.feed('Stack5_res4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack5_res5_batch1')
             .relu(name='Stack5_res5_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack5_res5_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack5_res5_batch2')
             .relu(name='Stack5_res5_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack5_res5_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack5_res5_batch3')
             .relu(name='Stack5_res5_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack5_res5_conv3'))

        (self.feed('Stack5_res4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack5_res5_skip'))

        (self.feed('Stack5_res5_conv3',
                   'Stack5_res5_skip')
         .add(name='Stack5_res5'))


# pool3
        (self.feed('Stack5_res4')
             .max_pool(2, 2, 2, 2, name='Stack5_pool3'))


# res6
        (self.feed('Stack5_pool3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack5_res6_batch1')
             .relu(name='Stack5_res6_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack5_res6_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack5_res6_batch2')
             .relu(name='Stack5_res6_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack5_res6_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack5_res6_batch3')
             .relu(name='Stack5_res6_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack5_res6_conv3'))

        (self.feed('Stack5_pool3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack5_res6_skip'))

        (self.feed('Stack5_res6_conv3',
                   'Stack5_res6_skip')
         .add(name='Stack5_res6'))

# res7
        (self.feed('Stack5_res6')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack5_res7_batch1')
             .relu(name='Stack5_res7_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack5_res7_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack5_res7_batch2')
             .relu(name='Stack5_res7_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack5_res7_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack5_res7_batch3')
             .relu(name='Stack5_res7_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack5_res7_conv3'))

        (self.feed('Stack5_res6')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack5_res7_skip'))

        (self.feed('Stack5_res7_conv3',
                   'Stack5_res7_skip')
         .add(name='Stack5_res7'))


# pool4
        (self.feed('Stack5_res6')
             .max_pool(2, 2, 2, 2, name='Stack5_pool4'))

# res8
        (self.feed('Stack5_pool4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack5_res8_batch1')
             .relu(name='Stack5_res8_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack5_res8_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack5_res8_batch2')
             .relu(name='Stack5_res8_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack5_res8_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack5_res8_batch3')
             .relu(name='Stack5_res8_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack5_res8_conv3'))

        (self.feed('Stack5_pool4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack5_res8_skip'))

        (self.feed('Stack5_res8_conv3',
                   'Stack5_res8_skip')
         .add(name='Stack5_res8'))

# res9
        (self.feed('Stack5_res8')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack5_res9_batch1')
             .relu(name='Stack5_res9_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack5_res9_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack5_res9_batch2')
             .relu(name='Stack5_res9_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack5_res9_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack5_res9_batch3')
             .relu(name='Stack5_res9_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack5_res9_conv3'))

        (self.feed('Stack5_res8')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack5_res9_skip'))

        (self.feed('Stack5_res9_conv3',
                   'Stack5_res9_skip')
         .add(name='Stack5_res9'))

# res10
        (self.feed('Stack5_res9')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack5_res10_batch1')
             .relu(name='Stack5_res10_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack5_res10_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack5_res10_batch2')
             .relu(name='Stack5_res10_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack5_res10_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack5_res10_batch3')
             .relu(name='Stack5_res10_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack5_res10_conv3'))

        (self.feed('Stack5_res9')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack5_res10_skip'))

        (self.feed('Stack5_res10_conv3',
                   'Stack5_res10_skip')
         .add(name='Stack5_res10'))


# upsample1
        (self.feed('Stack5_res10')
             .upsample(8, 8, name='Stack5_upSample1'))

# upsample1 + up1(Hg_res7)
        (self.feed('Stack5_upSample1',
                   'Stack5_res7')
         .add(name='Stack5_add1'))


# res11
        (self.feed('Stack5_add1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack5_res11_batch1')
             .relu(name='Stack5_res11_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack5_res11_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack5_res11_batch2')
             .relu(name='Stack5_res11_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack5_res11_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack5_res11_batch3')
             .relu(name='Stack5_res11_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack5_res11_conv3'))

        (self.feed('Stack5_add1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack5_res11_skip'))

        (self.feed('Stack5_res11_conv3',
                   'Stack5_res11_skip')
         .add(name='Stack5_res11'))


# upsample2
        (self.feed('Stack5_res11')
             .upsample(16, 16, name='Stack5_upSample2'))

# upsample2 + up1(Hg_res5)
        (self.feed('Stack5_upSample2',
                   'Stack5_res5')
         .add(name='Stack5_add2'))


# res12
        (self.feed('Stack5_add2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack5_res12_batch1')
             .relu(name='Stack5_res12_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack5_res12_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack5_res12_batch2')
             .relu(name='Stack5_res12_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack5_res12_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack5_res12_batch3')
             .relu(name='Stack5_res12_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack5_res12_conv3'))

        (self.feed('Stack5_add2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack5_res12_skip'))

        (self.feed('Stack5_res12_conv3',
                   'Stack5_res12_skip')
         .add(name='Stack5_res12'))


# upsample3
        (self.feed('Stack5_res12')
             .upsample(32, 32, name='Stack5_upSample3'))

# upsample3 + Stack5_resPool3
        (self.feed('Stack5_upSample3',
                   'Stack5_resPool3')
         .add(name='Stack5_add3'))


# res13
        (self.feed('Stack5_add3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack5_res13_batch1')
             .relu(name='Stack5_res13_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack5_res13_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack5_res13_batch2')
             .relu(name='Stack5_res13_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack5_res13_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack5_res13_batch3')
             .relu(name='Stack5_res13_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack5_res13_conv3'))

        (self.feed('Stack5_add3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack5_res13_skip'))

        (self.feed('Stack5_res13_conv3',
                   'Stack5_res13_skip')
         .add(name='Stack5_res13'))


# upsample4
        (self.feed('Stack5_res13')
             .upsample(64, 64, name='Stack5_upSample4'))

# upsample4 + Stack5_resPool2
        (self.feed('Stack5_upSample4',
                   'Stack5_resPool2')
         .add(name='Stack5_add4'))

###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  Hourglass  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

################################# Stack5 postprocess #################
# Linear layer to produce first set of predictions
# ll1
        (self.feed('Stack5_add4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack5_linearfunc1_conv1', padding='SAME')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack5_linearfunc1_batch1')
             .relu(name='Stack5_linearfunc1_relu'))
# ll2
        (self.feed('Stack5_linearfunc1_relu')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack5_linearfunc2_conv1', padding='SAME')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack5_linearfunc2_batch1')
             .relu(name='Stack5_linearfunc2_relu'))

# att itersize=3

# att U input: ll2
        (self.feed('Stack5_linearfunc2_relu')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='Stack5_att_U', padding='SAME'))
# att i=1 conv  C(1)
        with tf.variable_scope('Stack5_att_share_param', reuse=False):
            (self.feed('Stack5_att_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='Stack5_att_spConv1', padding='SAME'))
# att Qtemp
        (self.feed('Stack5_att_spConv1',
                   'Stack5_att_U')
         .add(name='Stack5_att_Qtemp1_add')
         .sigmoid(name='Stack5_att_Qtemp1'))

# att i=2 conv  C(2)
        with tf.variable_scope('Stack5_att_share_param', reuse=True):
            (self.feed('Stack5_att_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='Stack5_att_spConv1', padding='SAME'))
# att Qtemp2
        (self.feed('Stack5_att_spConv1',
                   'Stack5_att_U')
         .add(name='Stack5_att_Qtemp2_add')
         .sigmoid(name='Stack5_att_Qtemp2'))

# att i=3 conv  C(3)
        with tf.variable_scope('Stack5_att_share_param', reuse=True):
            (self.feed('Stack5_att_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='Stack5_att_spConv1', padding='SAME'))
# att Qtemp
        (self.feed('Stack5_att_spConv1',
                   'Stack5_att_U')
         .add(name='Stack5_att_Qtemp3_add')
         .sigmoid(name='Stack5_att_Qtemp3'))
# att pfeat
        (self.feed('Stack5_att_Qtemp3')
         .replicate(256, 3, name='Stack5_att_pfeat_replicate'))

        (self.feed('Stack5_linearfunc2_relu',
                   'Stack5_att_pfeat_replicate')
         .multiply2(name='Stack5_att_pfeat_multiply'))

# tmpOut 
        # (self.feed('Stack5_att_pfeat_multiply')
        #      .conv(1, 1, n_classes, 1, 1, biased=True, relu=False, name='Stack5_att_Heatmap', padding='SAME'))










# tmpOut att U input: Stack5_att_pfeat_multiply
        (self.feed('Stack5_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='Stack5_tmpOut_U', padding='SAME'))
# tmpOut att i=1 conv  C(1)
        with tf.variable_scope('Stack5_tmpOut_share_param', reuse=False):
            (self.feed('Stack5_tmpOut_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='Stack5_tmpOut_spConv1', padding='SAME'))
# tmpOut att Qtemp
        (self.feed('Stack5_tmpOut_spConv1',
                   'Stack5_tmpOut_U')
         .add(name='Stack5_tmpOut_Qtemp1_add')
         .sigmoid(name='Stack5_tmpOut_Qtemp1'))

# tmpOut att i=2 conv  C(2)
        with tf.variable_scope('Stack5_tmpOut_share_param', reuse=True):
            (self.feed('Stack5_tmpOut_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='Stack5_tmpOut_spConv1', padding='SAME'))
# tmpOut att Qtemp2
        (self.feed('Stack5_tmpOut_spConv1',
                   'Stack5_tmpOut_U')
         .add(name='Stack5_tmpOut_Qtemp2_add')
         .sigmoid(name='Stack5_tmpOut_Qtemp2'))

# tmpOut att i=3 conv  C(3)
        with tf.variable_scope('Stack5_tmpOut_share_param', reuse=True):
            (self.feed('Stack5_tmpOut_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='Stack5_tmpOut_spConv1', padding='SAME'))
# tmpOut att Qtemp
        (self.feed('Stack5_tmpOut_spConv1',
                   'Stack5_tmpOut_U')
         .add(name='Stack5_tmpOut_Qtemp3_add')
         .sigmoid(name='Stack5_tmpOut_Qtemp3'))
# tmpOut att pfeat
        (self.feed('Stack5_tmpOut_Qtemp3')
         .replicate(256, 3, name='Stack5_tmpOut_pfeat_replicate'))

        (self.feed('Stack5_att_pfeat_multiply',
                   'Stack5_tmpOut_pfeat_replicate')
         .multiply2(name='Stack5_tmpOut_pfeat_multiply'))

        (self.feed('Stack5_tmpOut_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='Stack5_tmpOut_s', padding='SAME'))

        (self.feed('Stack5_tmpOut_s')
         .replicate(16, 3, name='Stack5_Heatmap'))        


###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Stack5 postprocess ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

############################# generate input for next Stack ##########

# outmap
        (self.feed('Stack5_Heatmap')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack5_outmap', padding='SAME'))
# ll3
        (self.feed('Stack5_linearfunc1_relu')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack5_linearfunc3_conv1', padding='SAME')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack5_linearfunc3_batch1')
             .relu(name='Stack5_linearfunc3_relu'))
# tmointer
        (self.feed('Stack5_input',
                   'Stack5_outmap',
                   'Stack5_linearfunc3_relu')
         .add(name='Stack6_input'))

###^^^^^^^^^^^^^^^^^^^^^^^^^^^ generate input for next Stack ^^^^^^^^^^^^^^^^^^^^^^^^^^^^###
































































































#######################################  Stack6  #########################################
                 ####################  Hourglass  #####################
# res1
        (self.feed('Stack6_input')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack6_res1_batch1')
             .relu(name='Stack6_res1_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack6_res1_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack6_res1_batch2')
             .relu(name='Stack6_res1_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack6_res1_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack6_res1_batch3')
             .relu(name='Stack6_res1_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack6_res1_conv3'))

        (self.feed('Stack6_input')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack6_res1_skip'))

        (self.feed('Stack6_res1_conv3',
                   'Stack6_res1_skip')
         .add(name='Stack6_res1'))

# resPool1
        (self.feed('Stack6_res1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack6_resPool1_batch1')
             .relu(name='Stack6_resPool1_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack6_resPool1_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack6_resPool1_batch2')
             .relu(name='Stack6_resPool1_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack6_resPool1_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack6_resPool1_batch3')
             .relu(name='Stack6_resPool1_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack6_resPool1_conv3'))

        (self.feed('Stack6_res1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack6_resPool1_skip'))

        (self.feed('Stack6_res1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack6_resPool1_batch4')
             .relu(name='Stack6_resPool1_relu4')
             .max_pool(2, 2, 2, 2, name='Stack6_resPool1_pool')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='Stack6_resPool1_conv4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack6_resPool1_batch5')
             .relu(name='Stack6_resPool1_relu5')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='Stack6_resPool1_conv5')
             .upsample(64, 64, name='Stack6_resPool1_upSample'))


        (self.feed('Stack6_resPool1_conv3',
                   'Stack6_resPool1_skip',
                   'Stack6_resPool1_upSample')
         .add(name='Stack6_resPool1'))



# resPool2
        (self.feed('Stack6_resPool1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack6_resPool2_batch1')
             .relu(name='Stack6_resPool2_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack6_resPool2_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack6_resPool2_batch2')
             .relu(name='Stack6_resPool2_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack6_resPool2_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack6_resPool2_batch3')
             .relu(name='Stack6_resPool2_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack6_resPool2_conv3'))

        (self.feed('Stack6_resPool1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack6_resPool2_skip'))

        (self.feed('Stack6_resPool1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack6_resPool2_batch4')
             .relu(name='Stack6_resPool2_relu4')
             .max_pool(2, 2, 2, 2, name='Stack6_resPool2_pool')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='Stack6_resPool2_conv4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack6_resPool2_batch5')
             .relu(name='Stack6_resPool2_relu5')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='Stack6_resPool2_conv5')
             .upsample(64, 64, name='Stack6_resPool2_upSample'))


        (self.feed('Stack6_resPool2_conv3',
                   'Stack6_resPool2_skip',
                   'Stack6_resPool2_upSample')
         .add(name='Stack6_resPool2'))

# pool1
        (self.feed('Stack6_input')
             .max_pool(2, 2, 2, 2, name='Stack6_pool1'))


# res2
        (self.feed('Stack6_pool1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack6_res2_batch1')
             .relu(name='Stack6_res2_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack6_res2_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack6_res2_batch2')
             .relu(name='Stack6_res2_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack6_res2_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack6_res2_batch3')
             .relu(name='Stack6_res2_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack6_res2_conv3'))

        (self.feed('Stack6_pool1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack6_res2_skip'))

        (self.feed('Stack6_res2_conv3',
                   'Stack6_res2_skip')
         .add(name='Stack6_res2'))

# res3
        (self.feed('Stack6_res2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack6_res3_batch1')
             .relu(name='Stack6_res3_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack6_res3_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack6_res3_batch2')
             .relu(name='Stack6_res3_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack6_res3_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack6_res3_batch3')
             .relu(name='Stack6_res3_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack6_res3_conv3'))

        (self.feed('Stack6_res2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack6_res3_skip'))

        (self.feed('Stack6_res3_conv3',
                   'Stack6_res3_skip')
         .add(name='Stack6_res3'))

# resPool3
        (self.feed('Stack6_res3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack6_resPool3_batch1')
             .relu(name='Stack6_resPool3_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack6_resPool3_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack6_resPool3_batch2')
             .relu(name='Stack6_resPool3_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack6_resPool3_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack6_resPool3_batch3')
             .relu(name='Stack6_resPool3_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack6_resPool3_conv3'))

        (self.feed('Stack6_res3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack6_resPool3_skip'))

        (self.feed('Stack6_res3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack6_resPool3_batch4')
             .relu(name='Stack6_resPool3_relu4')
             .max_pool(2, 2, 2, 2, name='Stack6_resPool3_pool')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='Stack6_resPool3_conv4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack6_resPool3_batch5')
             .relu(name='Stack6_resPool3_relu5')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='Stack6_resPool3_conv5')
             .upsample(32, 32, name='Stack6_resPool3_upSample'))


        (self.feed('Stack6_resPool3_conv3',
                   'Stack6_resPool3_skip',
                   'Stack6_resPool3_upSample')
         .add(name='Stack6_resPool3'))




# pool2
        (self.feed('Stack6_res2')
             .max_pool(2, 2, 2, 2, name='Stack6_pool2'))


# res4
        (self.feed('Stack6_pool2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack6_res4_batch1')
             .relu(name='Stack6_res4_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack6_res4_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack6_res4_batch2')
             .relu(name='Stack6_res4_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack6_res4_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack6_res4_batch3')
             .relu(name='Stack6_res4_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack6_res4_conv3'))

        (self.feed('Stack6_pool2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack6_res4_skip'))

        (self.feed('Stack6_res4_conv3',
                   'Stack6_res4_skip')
         .add(name='Stack6_res4'))
# id:013 max-pooling
        # (self.feed('Stack6_pool3')
        #      .max_pool(2, 2, 2, 2, name='Stack6_pool4'))


# res5
        (self.feed('Stack6_res4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack6_res5_batch1')
             .relu(name='Stack6_res5_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack6_res5_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack6_res5_batch2')
             .relu(name='Stack6_res5_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack6_res5_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack6_res5_batch3')
             .relu(name='Stack6_res5_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack6_res5_conv3'))

        (self.feed('Stack6_res4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack6_res5_skip'))

        (self.feed('Stack6_res5_conv3',
                   'Stack6_res5_skip')
         .add(name='Stack6_res5'))


# pool3
        (self.feed('Stack6_res4')
             .max_pool(2, 2, 2, 2, name='Stack6_pool3'))


# res6
        (self.feed('Stack6_pool3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack6_res6_batch1')
             .relu(name='Stack6_res6_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack6_res6_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack6_res6_batch2')
             .relu(name='Stack6_res6_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack6_res6_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack6_res6_batch3')
             .relu(name='Stack6_res6_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack6_res6_conv3'))

        (self.feed('Stack6_pool3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack6_res6_skip'))

        (self.feed('Stack6_res6_conv3',
                   'Stack6_res6_skip')
         .add(name='Stack6_res6'))

# res7
        (self.feed('Stack6_res6')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack6_res7_batch1')
             .relu(name='Stack6_res7_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack6_res7_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack6_res7_batch2')
             .relu(name='Stack6_res7_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack6_res7_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack6_res7_batch3')
             .relu(name='Stack6_res7_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack6_res7_conv3'))

        (self.feed('Stack6_res6')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack6_res7_skip'))

        (self.feed('Stack6_res7_conv3',
                   'Stack6_res7_skip')
         .add(name='Stack6_res7'))


# pool4
        (self.feed('Stack6_res6')
             .max_pool(2, 2, 2, 2, name='Stack6_pool4'))

# res8
        (self.feed('Stack6_pool4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack6_res8_batch1')
             .relu(name='Stack6_res8_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack6_res8_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack6_res8_batch2')
             .relu(name='Stack6_res8_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack6_res8_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack6_res8_batch3')
             .relu(name='Stack6_res8_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack6_res8_conv3'))

        (self.feed('Stack6_pool4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack6_res8_skip'))

        (self.feed('Stack6_res8_conv3',
                   'Stack6_res8_skip')
         .add(name='Stack6_res8'))

# res9
        (self.feed('Stack6_res8')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack6_res9_batch1')
             .relu(name='Stack6_res9_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack6_res9_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack6_res9_batch2')
             .relu(name='Stack6_res9_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack6_res9_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack6_res9_batch3')
             .relu(name='Stack6_res9_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack6_res9_conv3'))

        (self.feed('Stack6_res8')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack6_res9_skip'))

        (self.feed('Stack6_res9_conv3',
                   'Stack6_res9_skip')
         .add(name='Stack6_res9'))

# res10
        (self.feed('Stack6_res9')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack6_res10_batch1')
             .relu(name='Stack6_res10_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack6_res10_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack6_res10_batch2')
             .relu(name='Stack6_res10_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack6_res10_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack6_res10_batch3')
             .relu(name='Stack6_res10_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack6_res10_conv3'))

        (self.feed('Stack6_res9')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack6_res10_skip'))

        (self.feed('Stack6_res10_conv3',
                   'Stack6_res10_skip')
         .add(name='Stack6_res10'))


# upsample1
        (self.feed('Stack6_res10')
             .upsample(8, 8, name='Stack6_upSample1'))

# upsample1 + up1(Hg_res7)
        (self.feed('Stack6_upSample1',
                   'Stack6_res7')
         .add(name='Stack6_add1'))


# res11
        (self.feed('Stack6_add1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack6_res11_batch1')
             .relu(name='Stack6_res11_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack6_res11_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack6_res11_batch2')
             .relu(name='Stack6_res11_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack6_res11_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack6_res11_batch3')
             .relu(name='Stack6_res11_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack6_res11_conv3'))

        (self.feed('Stack6_add1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack6_res11_skip'))

        (self.feed('Stack6_res11_conv3',
                   'Stack6_res11_skip')
         .add(name='Stack6_res11'))


# upsample2
        (self.feed('Stack6_res11')
             .upsample(16, 16, name='Stack6_upSample2'))

# upsample2 + up1(Hg_res5)
        (self.feed('Stack6_upSample2',
                   'Stack6_res5')
         .add(name='Stack6_add2'))


# res12
        (self.feed('Stack6_add2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack6_res12_batch1')
             .relu(name='Stack6_res12_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack6_res12_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack6_res12_batch2')
             .relu(name='Stack6_res12_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack6_res12_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack6_res12_batch3')
             .relu(name='Stack6_res12_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack6_res12_conv3'))

        (self.feed('Stack6_add2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack6_res12_skip'))

        (self.feed('Stack6_res12_conv3',
                   'Stack6_res12_skip')
         .add(name='Stack6_res12'))


# upsample3
        (self.feed('Stack6_res12')
             .upsample(32, 32, name='Stack6_upSample3'))

# upsample3 + Stack6_resPool3
        (self.feed('Stack6_upSample3',
                   'Stack6_resPool3')
         .add(name='Stack6_add3'))


# res13
        (self.feed('Stack6_add3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack6_res13_batch1')
             .relu(name='Stack6_res13_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack6_res13_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack6_res13_batch2')
             .relu(name='Stack6_res13_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack6_res13_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack6_res13_batch3')
             .relu(name='Stack6_res13_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack6_res13_conv3'))

        (self.feed('Stack6_add3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack6_res13_skip'))

        (self.feed('Stack6_res13_conv3',
                   'Stack6_res13_skip')
         .add(name='Stack6_res13'))


# upsample4
        (self.feed('Stack6_res13')
             .upsample(64, 64, name='Stack6_upSample4'))

# upsample4 + Stack6_resPool2
        (self.feed('Stack6_upSample4',
                   'Stack6_resPool2')
         .add(name='Stack6_add4'))

###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  Hourglass  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

################################# Stack6 postprocess #################
# Linear layer to produce first set of predictions
# ll1
        (self.feed('Stack6_add4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack6_linearfunc1_conv1', padding='SAME')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack6_linearfunc1_batch1')
             .relu(name='Stack6_linearfunc1_relu'))
# ll2
        (self.feed('Stack6_linearfunc1_relu')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack6_linearfunc2_conv1', padding='SAME')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack6_linearfunc2_batch1')
             .relu(name='Stack6_linearfunc2_relu'))

# att itersize=3

# att U input: ll2
        (self.feed('Stack6_linearfunc2_relu')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='Stack6_att_U', padding='SAME'))
# att i=1 conv  C(1)
        with tf.variable_scope('Stack6_att_share_param', reuse=False):
            (self.feed('Stack6_att_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='Stack6_att_spConv1', padding='SAME'))
# att Qtemp
        (self.feed('Stack6_att_spConv1',
                   'Stack6_att_U')
         .add(name='Stack6_att_Qtemp1_add')
         .sigmoid(name='Stack6_att_Qtemp1'))

# att i=2 conv  C(2)
        with tf.variable_scope('Stack6_att_share_param', reuse=True):
            (self.feed('Stack6_att_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='Stack6_att_spConv1', padding='SAME'))
# att Qtemp2
        (self.feed('Stack6_att_spConv1',
                   'Stack6_att_U')
         .add(name='Stack6_att_Qtemp2_add')
         .sigmoid(name='Stack6_att_Qtemp2'))

# att i=3 conv  C(3)
        with tf.variable_scope('Stack6_att_share_param', reuse=True):
            (self.feed('Stack6_att_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='Stack6_att_spConv1', padding='SAME'))
# att Qtemp
        (self.feed('Stack6_att_spConv1',
                   'Stack6_att_U')
         .add(name='Stack6_att_Qtemp3_add')
         .sigmoid(name='Stack6_att_Qtemp3'))
# att pfeat
        (self.feed('Stack6_att_Qtemp3')
         .replicate(256, 3, name='Stack6_att_pfeat_replicate'))

        (self.feed('Stack6_linearfunc2_relu',
                   'Stack6_att_pfeat_replicate')
         .multiply2(name='Stack6_att_pfeat_multiply'))

# tmpOut 
        # (self.feed('Stack6_att_pfeat_multiply')
        #      .conv(1, 1, n_classes, 1, 1, biased=True, relu=False, name='Stack6_att_Heatmap', padding='SAME'))










# tmpOut att U input: Stack6_att_pfeat_multiply
        (self.feed('Stack6_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='Stack6_tmpOut_U', padding='SAME'))
# tmpOut att i=1 conv  C(1)
        with tf.variable_scope('Stack6_tmpOut_share_param', reuse=False):
            (self.feed('Stack6_tmpOut_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='Stack6_tmpOut_spConv1', padding='SAME'))
# tmpOut att Qtemp
        (self.feed('Stack6_tmpOut_spConv1',
                   'Stack6_tmpOut_U')
         .add(name='Stack6_tmpOut_Qtemp1_add')
         .sigmoid(name='Stack6_tmpOut_Qtemp1'))

# tmpOut att i=2 conv  C(2)
        with tf.variable_scope('Stack6_tmpOut_share_param', reuse=True):
            (self.feed('Stack6_tmpOut_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='Stack6_tmpOut_spConv1', padding='SAME'))
# tmpOut att Qtemp2
        (self.feed('Stack6_tmpOut_spConv1',
                   'Stack6_tmpOut_U')
         .add(name='Stack6_tmpOut_Qtemp2_add')
         .sigmoid(name='Stack6_tmpOut_Qtemp2'))

# tmpOut att i=3 conv  C(3)
        with tf.variable_scope('Stack6_tmpOut_share_param', reuse=True):
            (self.feed('Stack6_tmpOut_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='Stack6_tmpOut_spConv1', padding='SAME'))
# tmpOut att Qtemp
        (self.feed('Stack6_tmpOut_spConv1',
                   'Stack6_tmpOut_U')
         .add(name='Stack6_tmpOut_Qtemp3_add')
         .sigmoid(name='Stack6_tmpOut_Qtemp3'))
# tmpOut att pfeat
        (self.feed('Stack6_tmpOut_Qtemp3')
         .replicate(256, 3, name='Stack6_tmpOut_pfeat_replicate'))

        (self.feed('Stack6_att_pfeat_multiply',
                   'Stack6_tmpOut_pfeat_replicate')
         .multiply2(name='Stack6_tmpOut_pfeat_multiply'))

        (self.feed('Stack6_tmpOut_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='Stack6_tmpOut_s', padding='SAME'))

        (self.feed('Stack6_tmpOut_s')
         .replicate(16, 3, name='Stack6_Heatmap'))        


###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Stack6 postprocess ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

############################# generate input for next Stack ##########

# outmap
        (self.feed('Stack6_Heatmap')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack6_outmap', padding='SAME'))
# ll3
        (self.feed('Stack6_linearfunc1_relu')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack6_linearfunc3_conv1', padding='SAME')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack6_linearfunc3_batch1')
             .relu(name='Stack6_linearfunc3_relu'))
# tmointer
        (self.feed('Stack6_input',
                   'Stack6_outmap',
                   'Stack6_linearfunc3_relu')
         .add(name='Stack7_input'))

###^^^^^^^^^^^^^^^^^^^^^^^^^^^ generate input for next Stack ^^^^^^^^^^^^^^^^^^^^^^^^^^^^###


















































































#######################################  Stack7  #########################################
                 ####################  Hourglass  #####################
# res1
        (self.feed('Stack7_input')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack7_res1_batch1')
             .relu(name='Stack7_res1_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack7_res1_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack7_res1_batch2')
             .relu(name='Stack7_res1_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack7_res1_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack7_res1_batch3')
             .relu(name='Stack7_res1_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack7_res1_conv3'))

        (self.feed('Stack7_input')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack7_res1_skip'))

        (self.feed('Stack7_res1_conv3',
                   'Stack7_res1_skip')
         .add(name='Stack7_res1'))

# resPool1
        (self.feed('Stack7_res1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack7_resPool1_batch1')
             .relu(name='Stack7_resPool1_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack7_resPool1_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack7_resPool1_batch2')
             .relu(name='Stack7_resPool1_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack7_resPool1_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack7_resPool1_batch3')
             .relu(name='Stack7_resPool1_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack7_resPool1_conv3'))

        (self.feed('Stack7_res1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack7_resPool1_skip'))

        (self.feed('Stack7_res1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack7_resPool1_batch4')
             .relu(name='Stack7_resPool1_relu4')
             .max_pool(2, 2, 2, 2, name='Stack7_resPool1_pool')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='Stack7_resPool1_conv4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack7_resPool1_batch5')
             .relu(name='Stack7_resPool1_relu5')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='Stack7_resPool1_conv5')
             .upsample(64, 64, name='Stack7_resPool1_upSample'))


        (self.feed('Stack7_resPool1_conv3',
                   'Stack7_resPool1_skip',
                   'Stack7_resPool1_upSample')
         .add(name='Stack7_resPool1'))



# resPool2
        (self.feed('Stack7_resPool1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack7_resPool2_batch1')
             .relu(name='Stack7_resPool2_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack7_resPool2_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack7_resPool2_batch2')
             .relu(name='Stack7_resPool2_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack7_resPool2_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack7_resPool2_batch3')
             .relu(name='Stack7_resPool2_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack7_resPool2_conv3'))

        (self.feed('Stack7_resPool1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack7_resPool2_skip'))

        (self.feed('Stack7_resPool1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack7_resPool2_batch4')
             .relu(name='Stack7_resPool2_relu4')
             .max_pool(2, 2, 2, 2, name='Stack7_resPool2_pool')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='Stack7_resPool2_conv4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack7_resPool2_batch5')
             .relu(name='Stack7_resPool2_relu5')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='Stack7_resPool2_conv5')
             .upsample(64, 64, name='Stack7_resPool2_upSample'))


        (self.feed('Stack7_resPool2_conv3',
                   'Stack7_resPool2_skip',
                   'Stack7_resPool2_upSample')
         .add(name='Stack7_resPool2'))

# pool1
        (self.feed('Stack7_input')
             .max_pool(2, 2, 2, 2, name='Stack7_pool1'))


# res2
        (self.feed('Stack7_pool1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack7_res2_batch1')
             .relu(name='Stack7_res2_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack7_res2_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack7_res2_batch2')
             .relu(name='Stack7_res2_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack7_res2_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack7_res2_batch3')
             .relu(name='Stack7_res2_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack7_res2_conv3'))

        (self.feed('Stack7_pool1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack7_res2_skip'))

        (self.feed('Stack7_res2_conv3',
                   'Stack7_res2_skip')
         .add(name='Stack7_res2'))

# res3
        (self.feed('Stack7_res2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack7_res3_batch1')
             .relu(name='Stack7_res3_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack7_res3_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack7_res3_batch2')
             .relu(name='Stack7_res3_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack7_res3_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack7_res3_batch3')
             .relu(name='Stack7_res3_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack7_res3_conv3'))

        (self.feed('Stack7_res2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack7_res3_skip'))

        (self.feed('Stack7_res3_conv3',
                   'Stack7_res3_skip')
         .add(name='Stack7_res3'))

# resPool3
        (self.feed('Stack7_res3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack7_resPool3_batch1')
             .relu(name='Stack7_resPool3_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack7_resPool3_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack7_resPool3_batch2')
             .relu(name='Stack7_resPool3_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack7_resPool3_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack7_resPool3_batch3')
             .relu(name='Stack7_resPool3_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack7_resPool3_conv3'))

        (self.feed('Stack7_res3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack7_resPool3_skip'))

        (self.feed('Stack7_res3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack7_resPool3_batch4')
             .relu(name='Stack7_resPool3_relu4')
             .max_pool(2, 2, 2, 2, name='Stack7_resPool3_pool')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='Stack7_resPool3_conv4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack7_resPool3_batch5')
             .relu(name='Stack7_resPool3_relu5')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='Stack7_resPool3_conv5')
             .upsample(32, 32, name='Stack7_resPool3_upSample'))


        (self.feed('Stack7_resPool3_conv3',
                   'Stack7_resPool3_skip',
                   'Stack7_resPool3_upSample')
         .add(name='Stack7_resPool3'))




# pool2
        (self.feed('Stack7_res2')
             .max_pool(2, 2, 2, 2, name='Stack7_pool2'))


# res4
        (self.feed('Stack7_pool2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack7_res4_batch1')
             .relu(name='Stack7_res4_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack7_res4_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack7_res4_batch2')
             .relu(name='Stack7_res4_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack7_res4_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack7_res4_batch3')
             .relu(name='Stack7_res4_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack7_res4_conv3'))

        (self.feed('Stack7_pool2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack7_res4_skip'))

        (self.feed('Stack7_res4_conv3',
                   'Stack7_res4_skip')
         .add(name='Stack7_res4'))
# id:013 max-pooling
        # (self.feed('Stack7_pool3')
        #      .max_pool(2, 2, 2, 2, name='Stack7_pool4'))


# res5
        (self.feed('Stack7_res4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack7_res5_batch1')
             .relu(name='Stack7_res5_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack7_res5_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack7_res5_batch2')
             .relu(name='Stack7_res5_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack7_res5_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack7_res5_batch3')
             .relu(name='Stack7_res5_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack7_res5_conv3'))

        (self.feed('Stack7_res4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack7_res5_skip'))

        (self.feed('Stack7_res5_conv3',
                   'Stack7_res5_skip')
         .add(name='Stack7_res5'))


# pool3
        (self.feed('Stack7_res4')
             .max_pool(2, 2, 2, 2, name='Stack7_pool3'))


# res6
        (self.feed('Stack7_pool3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack7_res6_batch1')
             .relu(name='Stack7_res6_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack7_res6_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack7_res6_batch2')
             .relu(name='Stack7_res6_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack7_res6_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack7_res6_batch3')
             .relu(name='Stack7_res6_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack7_res6_conv3'))

        (self.feed('Stack7_pool3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack7_res6_skip'))

        (self.feed('Stack7_res6_conv3',
                   'Stack7_res6_skip')
         .add(name='Stack7_res6'))

# res7
        (self.feed('Stack7_res6')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack7_res7_batch1')
             .relu(name='Stack7_res7_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack7_res7_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack7_res7_batch2')
             .relu(name='Stack7_res7_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack7_res7_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack7_res7_batch3')
             .relu(name='Stack7_res7_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack7_res7_conv3'))

        (self.feed('Stack7_res6')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack7_res7_skip'))

        (self.feed('Stack7_res7_conv3',
                   'Stack7_res7_skip')
         .add(name='Stack7_res7'))


# pool4
        (self.feed('Stack7_res6')
             .max_pool(2, 2, 2, 2, name='Stack7_pool4'))

# res8
        (self.feed('Stack7_pool4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack7_res8_batch1')
             .relu(name='Stack7_res8_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack7_res8_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack7_res8_batch2')
             .relu(name='Stack7_res8_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack7_res8_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack7_res8_batch3')
             .relu(name='Stack7_res8_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack7_res8_conv3'))

        (self.feed('Stack7_pool4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack7_res8_skip'))

        (self.feed('Stack7_res8_conv3',
                   'Stack7_res8_skip')
         .add(name='Stack7_res8'))

# res9
        (self.feed('Stack7_res8')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack7_res9_batch1')
             .relu(name='Stack7_res9_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack7_res9_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack7_res9_batch2')
             .relu(name='Stack7_res9_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack7_res9_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack7_res9_batch3')
             .relu(name='Stack7_res9_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack7_res9_conv3'))

        (self.feed('Stack7_res8')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack7_res9_skip'))

        (self.feed('Stack7_res9_conv3',
                   'Stack7_res9_skip')
         .add(name='Stack7_res9'))

# res10
        (self.feed('Stack7_res9')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack7_res10_batch1')
             .relu(name='Stack7_res10_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack7_res10_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack7_res10_batch2')
             .relu(name='Stack7_res10_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack7_res10_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack7_res10_batch3')
             .relu(name='Stack7_res10_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack7_res10_conv3'))

        (self.feed('Stack7_res9')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack7_res10_skip'))

        (self.feed('Stack7_res10_conv3',
                   'Stack7_res10_skip')
         .add(name='Stack7_res10'))


# upsample1
        (self.feed('Stack7_res10')
             .upsample(8, 8, name='Stack7_upSample1'))

# upsample1 + up1(Hg_res7)
        (self.feed('Stack7_upSample1',
                   'Stack7_res7')
         .add(name='Stack7_add1'))


# res11
        (self.feed('Stack7_add1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack7_res11_batch1')
             .relu(name='Stack7_res11_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack7_res11_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack7_res11_batch2')
             .relu(name='Stack7_res11_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack7_res11_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack7_res11_batch3')
             .relu(name='Stack7_res11_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack7_res11_conv3'))

        (self.feed('Stack7_add1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack7_res11_skip'))

        (self.feed('Stack7_res11_conv3',
                   'Stack7_res11_skip')
         .add(name='Stack7_res11'))


# upsample2
        (self.feed('Stack7_res11')
             .upsample(16, 16, name='Stack7_upSample2'))

# upsample2 + up1(Hg_res5)
        (self.feed('Stack7_upSample2',
                   'Stack7_res5')
         .add(name='Stack7_add2'))


# res12
        (self.feed('Stack7_add2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack7_res12_batch1')
             .relu(name='Stack7_res12_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack7_res12_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack7_res12_batch2')
             .relu(name='Stack7_res12_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack7_res12_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack7_res12_batch3')
             .relu(name='Stack7_res12_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack7_res12_conv3'))

        (self.feed('Stack7_add2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack7_res12_skip'))

        (self.feed('Stack7_res12_conv3',
                   'Stack7_res12_skip')
         .add(name='Stack7_res12'))


# upsample3
        (self.feed('Stack7_res12')
             .upsample(32, 32, name='Stack7_upSample3'))

# upsample3 + Stack7_resPool3
        (self.feed('Stack7_upSample3',
                   'Stack7_resPool3')
         .add(name='Stack7_add3'))


# res13
        (self.feed('Stack7_add3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack7_res13_batch1')
             .relu(name='Stack7_res13_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack7_res13_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack7_res13_batch2')
             .relu(name='Stack7_res13_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack7_res13_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack7_res13_batch3')
             .relu(name='Stack7_res13_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack7_res13_conv3'))

        (self.feed('Stack7_add3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack7_res13_skip'))

        (self.feed('Stack7_res13_conv3',
                   'Stack7_res13_skip')
         .add(name='Stack7_res13'))


# upsample4
        (self.feed('Stack7_res13')
             .upsample(64, 64, name='Stack7_upSample4'))

# upsample4 + Stack7_resPool2
        (self.feed('Stack7_upSample4',
                   'Stack7_resPool2')
         .add(name='Stack7_add4'))

###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  Hourglass  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

################################# Stack7 postprocess #################
# Linear layer to produce first set of predictions
# ll1
        (self.feed('Stack7_add4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack7_linearfunc1_conv1', padding='SAME')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack7_linearfunc1_batch1')
             .relu(name='Stack7_linearfunc1_relu'))
# ll2
        (self.feed('Stack7_linearfunc1_relu')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack7_linearfunc2_conv1', padding='SAME')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack7_linearfunc2_batch1')
             .relu(name='Stack7_linearfunc2_relu'))

# att itersize=3

# att U input: ll2
        (self.feed('Stack7_linearfunc2_relu')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='Stack7_att_U', padding='SAME'))
# att i=1 conv  C(1)
        with tf.variable_scope('Stack7_att_share_param', reuse=False):
            (self.feed('Stack7_att_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='Stack7_att_spConv1', padding='SAME'))
# att Qtemp
        (self.feed('Stack7_att_spConv1',
                   'Stack7_att_U')
         .add(name='Stack7_att_Qtemp1_add')
         .sigmoid(name='Stack7_att_Qtemp1'))

# att i=2 conv  C(2)
        with tf.variable_scope('Stack7_att_share_param', reuse=True):
            (self.feed('Stack7_att_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='Stack7_att_spConv1', padding='SAME'))
# att Qtemp2
        (self.feed('Stack7_att_spConv1',
                   'Stack7_att_U')
         .add(name='Stack7_att_Qtemp2_add')
         .sigmoid(name='Stack7_att_Qtemp2'))

# att i=3 conv  C(3)
        with tf.variable_scope('Stack7_att_share_param', reuse=True):
            (self.feed('Stack7_att_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='Stack7_att_spConv1', padding='SAME'))
# att Qtemp
        (self.feed('Stack7_att_spConv1',
                   'Stack7_att_U')
         .add(name='Stack7_att_Qtemp3_add')
         .sigmoid(name='Stack7_att_Qtemp3'))
# att pfeat
        (self.feed('Stack7_att_Qtemp3')
         .replicate(256, 3, name='Stack7_att_pfeat_replicate'))

        (self.feed('Stack7_linearfunc2_relu',
                   'Stack7_att_pfeat_replicate')
         .multiply2(name='Stack7_att_pfeat_multiply'))

# tmpOut 
        # (self.feed('Stack7_att_pfeat_multiply')
        #      .conv(1, 1, n_classes, 1, 1, biased=True, relu=False, name='Stack7_att_Heatmap', padding='SAME'))










# tmpOut att U input: Stack7_att_pfeat_multiply
        (self.feed('Stack7_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='Stack7_tmpOut_U', padding='SAME'))
# tmpOut att i=1 conv  C(1)
        with tf.variable_scope('Stack7_tmpOut_share_param', reuse=False):
            (self.feed('Stack7_tmpOut_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='Stack7_tmpOut_spConv1', padding='SAME'))
# tmpOut att Qtemp
        (self.feed('Stack7_tmpOut_spConv1',
                   'Stack7_tmpOut_U')
         .add(name='Stack7_tmpOut_Qtemp1_add')
         .sigmoid(name='Stack7_tmpOut_Qtemp1'))

# tmpOut att i=2 conv  C(2)
        with tf.variable_scope('Stack7_tmpOut_share_param', reuse=True):
            (self.feed('Stack7_tmpOut_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='Stack7_tmpOut_spConv1', padding='SAME'))
# tmpOut att Qtemp2
        (self.feed('Stack7_tmpOut_spConv1',
                   'Stack7_tmpOut_U')
         .add(name='Stack7_tmpOut_Qtemp2_add')
         .sigmoid(name='Stack7_tmpOut_Qtemp2'))

# tmpOut att i=3 conv  C(3)
        with tf.variable_scope('Stack7_tmpOut_share_param', reuse=True):
            (self.feed('Stack7_tmpOut_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='Stack7_tmpOut_spConv1', padding='SAME'))
# tmpOut att Qtemp
        (self.feed('Stack7_tmpOut_spConv1',
                   'Stack7_tmpOut_U')
         .add(name='Stack7_tmpOut_Qtemp3_add')
         .sigmoid(name='Stack7_tmpOut_Qtemp3'))
# tmpOut att pfeat
        (self.feed('Stack7_tmpOut_Qtemp3')
         .replicate(256, 3, name='Stack7_tmpOut_pfeat_replicate'))

        (self.feed('Stack7_att_pfeat_multiply',
                   'Stack7_tmpOut_pfeat_replicate')
         .multiply2(name='Stack7_tmpOut_pfeat_multiply'))

        (self.feed('Stack7_tmpOut_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='Stack7_tmpOut_s', padding='SAME'))

        (self.feed('Stack7_tmpOut_s')
         .replicate(16, 3, name='Stack7_Heatmap'))        


###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Stack7 postprocess ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

############################# generate input for next Stack ##########

# outmap
        (self.feed('Stack7_Heatmap')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack7_outmap', padding='SAME'))
# ll3
        (self.feed('Stack7_linearfunc1_relu')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack7_linearfunc3_conv1', padding='SAME')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack7_linearfunc3_batch1')
             .relu(name='Stack7_linearfunc3_relu'))
# tmointer
        (self.feed('Stack7_input',
                   'Stack7_outmap',
                   'Stack7_linearfunc3_relu')
         .add(name='Stack8_input'))

###^^^^^^^^^^^^^^^^^^^^^^^^^^^ generate input for next Stack ^^^^^^^^^^^^^^^^^^^^^^^^^^^^###





















































#######################################  Stack8  #########################################
                 ####################  Hourglass  #####################
# res1
        (self.feed('Stack8_input')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack8_res1_batch1')
             .relu(name='Stack8_res1_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack8_res1_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack8_res1_batch2')
             .relu(name='Stack8_res1_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack8_res1_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack8_res1_batch3')
             .relu(name='Stack8_res1_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack8_res1_conv3'))

        (self.feed('Stack8_input')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack8_res1_skip'))

        (self.feed('Stack8_res1_conv3',
                   'Stack8_res1_skip')
         .add(name='Stack8_res1'))

# resPool1
        (self.feed('Stack8_res1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack8_resPool1_batch1')
             .relu(name='Stack8_resPool1_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack8_resPool1_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack8_resPool1_batch2')
             .relu(name='Stack8_resPool1_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack8_resPool1_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack8_resPool1_batch3')
             .relu(name='Stack8_resPool1_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack8_resPool1_conv3'))

        (self.feed('Stack8_res1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack8_resPool1_skip'))

        (self.feed('Stack8_res1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack8_resPool1_batch4')
             .relu(name='Stack8_resPool1_relu4')
             .max_pool(2, 2, 2, 2, name='Stack8_resPool1_pool')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='Stack8_resPool1_conv4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack8_resPool1_batch5')
             .relu(name='Stack8_resPool1_relu5')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='Stack8_resPool1_conv5')
             .upsample(64, 64, name='Stack8_resPool1_upSample'))


        (self.feed('Stack8_resPool1_conv3',
                   'Stack8_resPool1_skip',
                   'Stack8_resPool1_upSample')
         .add(name='Stack8_resPool1'))



# resPool2
        (self.feed('Stack8_resPool1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack8_resPool2_batch1')
             .relu(name='Stack8_resPool2_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack8_resPool2_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack8_resPool2_batch2')
             .relu(name='Stack8_resPool2_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack8_resPool2_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack8_resPool2_batch3')
             .relu(name='Stack8_resPool2_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack8_resPool2_conv3'))

        (self.feed('Stack8_resPool1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack8_resPool2_skip'))

        (self.feed('Stack8_resPool1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack8_resPool2_batch4')
             .relu(name='Stack8_resPool2_relu4')
             .max_pool(2, 2, 2, 2, name='Stack8_resPool2_pool')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='Stack8_resPool2_conv4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack8_resPool2_batch5')
             .relu(name='Stack8_resPool2_relu5')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='Stack8_resPool2_conv5')
             .upsample(64, 64, name='Stack8_resPool2_upSample'))


        (self.feed('Stack8_resPool2_conv3',
                   'Stack8_resPool2_skip',
                   'Stack8_resPool2_upSample')
         .add(name='Stack8_resPool2'))

# pool1
        (self.feed('Stack8_input')
             .max_pool(2, 2, 2, 2, name='Stack8_pool1'))


# res2
        (self.feed('Stack8_pool1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack8_res2_batch1')
             .relu(name='Stack8_res2_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack8_res2_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack8_res2_batch2')
             .relu(name='Stack8_res2_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack8_res2_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack8_res2_batch3')
             .relu(name='Stack8_res2_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack8_res2_conv3'))

        (self.feed('Stack8_pool1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack8_res2_skip'))

        (self.feed('Stack8_res2_conv3',
                   'Stack8_res2_skip')
         .add(name='Stack8_res2'))

# res3
        (self.feed('Stack8_res2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack8_res3_batch1')
             .relu(name='Stack8_res3_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack8_res3_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack8_res3_batch2')
             .relu(name='Stack8_res3_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack8_res3_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack8_res3_batch3')
             .relu(name='Stack8_res3_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack8_res3_conv3'))

        (self.feed('Stack8_res2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack8_res3_skip'))

        (self.feed('Stack8_res3_conv3',
                   'Stack8_res3_skip')
         .add(name='Stack8_res3'))

# resPool3
        (self.feed('Stack8_res3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack8_resPool3_batch1')
             .relu(name='Stack8_resPool3_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack8_resPool3_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack8_resPool3_batch2')
             .relu(name='Stack8_resPool3_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack8_resPool3_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack8_resPool3_batch3')
             .relu(name='Stack8_resPool3_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack8_resPool3_conv3'))

        (self.feed('Stack8_res3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack8_resPool3_skip'))

        (self.feed('Stack8_res3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack8_resPool3_batch4')
             .relu(name='Stack8_resPool3_relu4')
             .max_pool(2, 2, 2, 2, name='Stack8_resPool3_pool')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='Stack8_resPool3_conv4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack8_resPool3_batch5')
             .relu(name='Stack8_resPool3_relu5')
             .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='Stack8_resPool3_conv5')
             .upsample(32, 32, name='Stack8_resPool3_upSample'))


        (self.feed('Stack8_resPool3_conv3',
                   'Stack8_resPool3_skip',
                   'Stack8_resPool3_upSample')
         .add(name='Stack8_resPool3'))




# pool2
        (self.feed('Stack8_res2')
             .max_pool(2, 2, 2, 2, name='Stack8_pool2'))


# res4
        (self.feed('Stack8_pool2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack8_res4_batch1')
             .relu(name='Stack8_res4_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack8_res4_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack8_res4_batch2')
             .relu(name='Stack8_res4_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack8_res4_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack8_res4_batch3')
             .relu(name='Stack8_res4_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack8_res4_conv3'))

        (self.feed('Stack8_pool2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack8_res4_skip'))

        (self.feed('Stack8_res4_conv3',
                   'Stack8_res4_skip')
         .add(name='Stack8_res4'))
# id:013 max-pooling
        # (self.feed('Stack8_pool3')
        #      .max_pool(2, 2, 2, 2, name='Stack8_pool4'))


# res5
        (self.feed('Stack8_res4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack8_res5_batch1')
             .relu(name='Stack8_res5_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack8_res5_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack8_res5_batch2')
             .relu(name='Stack8_res5_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack8_res5_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack8_res5_batch3')
             .relu(name='Stack8_res5_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack8_res5_conv3'))

        (self.feed('Stack8_res4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack8_res5_skip'))

        (self.feed('Stack8_res5_conv3',
                   'Stack8_res5_skip')
         .add(name='Stack8_res5'))


# pool3
        (self.feed('Stack8_res4')
             .max_pool(2, 2, 2, 2, name='Stack8_pool3'))


# res6
        (self.feed('Stack8_pool3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack8_res6_batch1')
             .relu(name='Stack8_res6_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack8_res6_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack8_res6_batch2')
             .relu(name='Stack8_res6_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack8_res6_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack8_res6_batch3')
             .relu(name='Stack8_res6_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack8_res6_conv3'))

        (self.feed('Stack8_pool3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack8_res6_skip'))

        (self.feed('Stack8_res6_conv3',
                   'Stack8_res6_skip')
         .add(name='Stack8_res6'))

# res7
        (self.feed('Stack8_res6')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack8_res7_batch1')
             .relu(name='Stack8_res7_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack8_res7_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack8_res7_batch2')
             .relu(name='Stack8_res7_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack8_res7_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack8_res7_batch3')
             .relu(name='Stack8_res7_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack8_res7_conv3'))

        (self.feed('Stack8_res6')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack8_res7_skip'))

        (self.feed('Stack8_res7_conv3',
                   'Stack8_res7_skip')
         .add(name='Stack8_res7'))


# pool4
        (self.feed('Stack8_res6')
             .max_pool(2, 2, 2, 2, name='Stack8_pool4'))

# res8
        (self.feed('Stack8_pool4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack8_res8_batch1')
             .relu(name='Stack8_res8_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack8_res8_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack8_res8_batch2')
             .relu(name='Stack8_res8_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack8_res8_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack8_res8_batch3')
             .relu(name='Stack8_res8_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack8_res8_conv3'))

        (self.feed('Stack8_pool4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack8_res8_skip'))

        (self.feed('Stack8_res8_conv3',
                   'Stack8_res8_skip')
         .add(name='Stack8_res8'))

# res9
        (self.feed('Stack8_res8')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack8_res9_batch1')
             .relu(name='Stack8_res9_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack8_res9_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack8_res9_batch2')
             .relu(name='Stack8_res9_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack8_res9_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack8_res9_batch3')
             .relu(name='Stack8_res9_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack8_res9_conv3'))

        (self.feed('Stack8_res8')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack8_res9_skip'))

        (self.feed('Stack8_res9_conv3',
                   'Stack8_res9_skip')
         .add(name='Stack8_res9'))

# res10
        (self.feed('Stack8_res9')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack8_res10_batch1')
             .relu(name='Stack8_res10_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack8_res10_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack8_res10_batch2')
             .relu(name='Stack8_res10_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack8_res10_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack8_res10_batch3')
             .relu(name='Stack8_res10_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack8_res10_conv3'))

        (self.feed('Stack8_res9')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack8_res10_skip'))

        (self.feed('Stack8_res10_conv3',
                   'Stack8_res10_skip')
         .add(name='Stack8_res10'))


# upsample1
        (self.feed('Stack8_res10')
             .upsample(8, 8, name='Stack8_upSample1'))

# upsample1 + up1(Hg_res7)
        (self.feed('Stack8_upSample1',
                   'Stack8_res7')
         .add(name='Stack8_add1'))


# res11
        (self.feed('Stack8_add1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack8_res11_batch1')
             .relu(name='Stack8_res11_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack8_res11_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack8_res11_batch2')
             .relu(name='Stack8_res11_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack8_res11_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack8_res11_batch3')
             .relu(name='Stack8_res11_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack8_res11_conv3'))

        (self.feed('Stack8_add1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack8_res11_skip'))

        (self.feed('Stack8_res11_conv3',
                   'Stack8_res11_skip')
         .add(name='Stack8_res11'))


# upsample2
        (self.feed('Stack8_res11')
             .upsample(16, 16, name='Stack8_upSample2'))

# upsample2 + up1(Hg_res5)
        (self.feed('Stack8_upSample2',
                   'Stack8_res5')
         .add(name='Stack8_add2'))


# res12
        (self.feed('Stack8_add2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack8_res12_batch1')
             .relu(name='Stack8_res12_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack8_res12_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack8_res12_batch2')
             .relu(name='Stack8_res12_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack8_res12_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack8_res12_batch3')
             .relu(name='Stack8_res12_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack8_res12_conv3'))

        (self.feed('Stack8_add2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack8_res12_skip'))

        (self.feed('Stack8_res12_conv3',
                   'Stack8_res12_skip')
         .add(name='Stack8_res12'))


# upsample3
        (self.feed('Stack8_res12')
             .upsample(32, 32, name='Stack8_upSample3'))

# upsample3 + Stack8_resPool3
        (self.feed('Stack8_upSample3',
                   'Stack8_resPool3')
         .add(name='Stack8_add3'))


# res13
        (self.feed('Stack8_add3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack8_res13_batch1')
             .relu(name='Stack8_res13_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='Stack8_res13_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack8_res13_batch2')
             .relu(name='Stack8_res13_relu2')
             #.pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='Stack8_res13_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack8_res13_batch3')
             .relu(name='Stack8_res13_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack8_res13_conv3'))

        (self.feed('Stack8_add3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack8_res13_skip'))

        (self.feed('Stack8_res13_conv3',
                   'Stack8_res13_skip')
         .add(name='Stack8_res13'))


# upsample4
        (self.feed('Stack8_res13')
             .upsample(64, 64, name='Stack8_upSample4'))

# upsample4 + Stack8_resPool2
        (self.feed('Stack8_upSample4',
                   'Stack8_resPool2')
         .add(name='Stack8_add4'))

###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  Hourglass  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

################################# Stack8 postprocess #################
# Linear layer to produce first set of predictions
# ll1
        (self.feed('Stack8_add4')
             .conv(1, 1, 512, 1, 1, biased=True, relu=False, name='Stack8_linearfunc1_conv1', padding='SAME')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack8_linearfunc1_batch1')
             .relu(name='Stack8_linearfunc1_relu'))
# ll2
        (self.feed('Stack8_linearfunc1_relu')
             .conv(1, 1, 512, 1, 1, biased=True, relu=False, name='Stack8_linearfunc2_conv1', padding='SAME')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack8_linearfunc2_batch1')
             .relu(name='Stack8_linearfunc2_relu'))

# att itersize=3

# att U input: ll2
        (self.feed('Stack8_linearfunc2_relu')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='Stack8_att_U', padding='SAME'))
# att i=1 conv  C(1)
        with tf.variable_scope('Stack8_att_share_param', reuse=False):
            (self.feed('Stack8_att_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='Stack8_att_spConv1', padding='SAME'))
# att Qtemp
        (self.feed('Stack8_att_spConv1',
                   'Stack8_att_U')
         .add(name='Stack8_att_Qtemp1_add')
         .sigmoid(name='Stack8_att_Qtemp1'))

# att i=2 conv  C(2)
        with tf.variable_scope('Stack8_att_share_param', reuse=True):
            (self.feed('Stack8_att_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='Stack8_att_spConv1', padding='SAME'))
# att Qtemp2
        (self.feed('Stack8_att_spConv1',
                   'Stack8_att_U')
         .add(name='Stack8_att_Qtemp2_add')
         .sigmoid(name='Stack8_att_Qtemp2'))

# att i=3 conv  C(3)
        with tf.variable_scope('Stack8_att_share_param', reuse=True):
            (self.feed('Stack8_att_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='Stack8_att_spConv1', padding='SAME'))
# att Qtemp
        (self.feed('Stack8_att_spConv1',
                   'Stack8_att_U')
         .add(name='Stack8_att_Qtemp3_add')
         .sigmoid(name='Stack8_att_Qtemp3'))
# att pfeat
        (self.feed('Stack8_att_Qtemp3')
         .replicate(512, 3, name='Stack8_att_pfeat_replicate'))

        (self.feed('Stack8_linearfunc2_relu',
                   'Stack8_att_pfeat_replicate')
         .multiply2(name='Stack8_att_pfeat_multiply'))

# tmpOut 
        # (self.feed('Stack8_att_pfeat_multiply')
        #      .conv(1, 1, n_classes, 1, 1, biased=True, relu=False, name='Stack8_att_Heatmap', padding='SAME'))










# tmpOut att U input: Stack8_att_pfeat_multiply
        (self.feed('Stack8_att_pfeat_multiply')
             .conv(3, 3, 1, 1, 1, biased=True, relu=False, name='Stack8_tmpOut_U', padding='SAME'))
# tmpOut att i=1 conv  C(1)
        with tf.variable_scope('Stack8_tmpOut_share_param', reuse=False):
            (self.feed('Stack8_tmpOut_U')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='Stack8_tmpOut_spConv1', padding='SAME'))
# tmpOut att Qtemp
        (self.feed('Stack8_tmpOut_spConv1',
                   'Stack8_tmpOut_U')
         .add(name='Stack8_tmpOut_Qtemp1_add')
         .sigmoid(name='Stack8_tmpOut_Qtemp1'))

# tmpOut att i=2 conv  C(2)
        with tf.variable_scope('Stack8_tmpOut_share_param', reuse=True):
            (self.feed('Stack8_tmpOut_Qtemp1')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='Stack8_tmpOut_spConv1', padding='SAME'))
# tmpOut att Qtemp2
        (self.feed('Stack8_tmpOut_spConv1',
                   'Stack8_tmpOut_U')
         .add(name='Stack8_tmpOut_Qtemp2_add')
         .sigmoid(name='Stack8_tmpOut_Qtemp2'))

# tmpOut att i=3 conv  C(3)
        with tf.variable_scope('Stack8_tmpOut_share_param', reuse=True):
            (self.feed('Stack8_tmpOut_Qtemp2')
                 .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='Stack8_tmpOut_spConv1', padding='SAME'))
# tmpOut att Qtemp
        (self.feed('Stack8_tmpOut_spConv1',
                   'Stack8_tmpOut_U')
         .add(name='Stack8_tmpOut_Qtemp3_add')
         .sigmoid(name='Stack8_tmpOut_Qtemp3'))
# tmpOut att pfeat
        (self.feed('Stack8_tmpOut_Qtemp3')
         .replicate(512, 3, name='Stack8_tmpOut_pfeat_replicate'))

        (self.feed('Stack8_att_pfeat_multiply',
                   'Stack8_tmpOut_pfeat_replicate')
         .multiply2(name='Stack8_tmpOut_pfeat_multiply'))

        (self.feed('Stack8_tmpOut_pfeat_multiply')
             .conv(1, 1, 1, 1, 1, biased=True, relu=False, name='Stack8_tmpOut_s', padding='SAME'))

        (self.feed('Stack8_tmpOut_s')
         .replicate(16, 3, name='Stack8_Heatmap'))        


###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Stack8 postprocess ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

# ############################# generate input for next Stack ##########

# # outmap
#         (self.feed('Stack8_Heatmap')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack8_outmap', padding='SAME'))
# # ll3
#         (self.feed('Stack8_linearfunc1_relu')
#              .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='Stack8_linearfunc3_conv1', padding='SAME')
#              .tf_batch_normalization(is_training=is_training, activation_fn=None, name='Stack8_linearfunc3_batch1')
#              .relu(name='Stack8_linearfunc3_relu'))
# # tmointer
#         (self.feed('Stack8_input',
#                    'Stack8_outmap',
#                    'Stack8_linearfunc3_relu')
#          .add(name='Stack8_input'))

# ###^^^^^^^^^^^^^^^^^^^^^^^^^^^ generate input for next Stack ^^^^^^^^^^^^^^^^^^^^^^^^^^^^###


















































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































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
