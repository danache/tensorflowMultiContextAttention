# Converted to TensorFlow .caffemodel
# with the DeepLab-ResNet configuration.
# The batch normalisation layer is provided by
# the slim library
# (https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim).

from kaffe.tensorflow import Network
import tensorflow as tf
import numpy as np

nFeats = 256
nStack = 8
nModule = 1
LRNKer = 1
nPool = 4
outputRes = 64


class MultiContextAttentionModel(Network):

    def setup(self, is_training, n_classes):
        '''Network definition.

        Args:
          is_training: whether to update the running mean and variance of the batch normalisation layer.
                       If the batch size is small, it is better to keep the running mean and variance of 
                       the-pretrained model frozen.'''
                                           ##   PreProcess   ##
        (self.feed('data')
            .preProcess(nFeats, name = 'PreProcessing'))

                                            ##   Stack 1   ##
        (self.feed('PreProcessing')
            .hourglass(nPool, nFeats, outputRes, nModule, name = 'Stack1_HourGlass')
            .lin(nFeats, nFeats, name = 'Stack1_ll1')
            .lin(nFeats, nFeats, name = 'Stack1_ll2')
            .AttentionPartsCRF(nFeats, LRNKer, 3, 0, name = 'Stack1_att')
            .conv_layer(1, 1, n_classes, 1, 1, biased=True, relu=False, name = 'Stack1_tmpOut', padding='SAME'))

        (self.feed('Stack1_tmpOut')
            .conv_layer(1, 1, 256, 1, 1, biased=True, relu=False, name = 'Stack1_outmap', padding='SAME'))

        (self.feed('Stack1_ll1')
            .lin(nFeats, nFeats, name = 'Stack1_ll3'))

        (self.feed('PreProcessing',
                    'Stack1_outmap',
                    'Stack1_ll3')
            .add(name = 'Stack2_input'))


                                           ##   Stack 2   ##
        (self.feed('Stack2_input')
            .hourglass(nPool, nFeats, outputRes, nModule, name = 'Stack2_HourGlass')
            .lin(nFeats, nFeats, name = 'Stack2_ll1')
            .lin(nFeats, nFeats, name = 'Stack2_ll2')
            .AttentionPartsCRF(nFeats, LRNKer, 3, 0, name = 'Stack2_att')
            .conv_layer(1, 1, n_classes, 1, 1, biased=True, relu=False, name = 'Stack2_tmpOut', padding='SAME'))

        (self.feed('Stack2_tmpOut')
            .conv_layer(1, 1, 256, 1, 1, biased=True, relu=False, name = 'Stack2_outmap', padding='SAME'))

        (self.feed('Stack2_ll1')
            .lin(nFeats, nFeats, name = 'Stack2_ll3'))

        (self.feed('Stack2_input',
                    'Stack2_outmap',
                    'Stack2_ll3')
            .add(name = 'Stack3_input'))



                                           ##   Stack 3   ##
        (self.feed('Stack3_input')
            .hourglass(nPool, nFeats, outputRes, nModule, name = 'Stack3_HourGlass')
            .lin(nFeats, nFeats, name = 'Stack3_ll1')
            .lin(nFeats, nFeats, name = 'Stack3_ll2')
            .AttentionPartsCRF(nFeats, LRNKer, 3, 0, name = 'Stack3_att')
            .conv_layer(1, 1, n_classes, 1, 1, biased=True, relu=False, name = 'Stack3_tmpOut', padding='SAME'))

        (self.feed('Stack3_tmpOut')
            .conv_layer(1, 1, 256, 1, 1, biased=True, relu=False, name = 'Stack3_outmap', padding='SAME'))

        (self.feed('Stack3_ll1')
            .lin(nFeats, nFeats, name = 'Stack3_ll3'))

        (self.feed('Stack3_input',
                    'Stack3_outmap',
                    'Stack3_ll3')
            .add(name = 'Stack4_input'))



                                           ##   Stack 4   ##
        (self.feed('Stack4_input')
            .hourglass(nPool, nFeats, outputRes, nModule, name = 'Stack4_HourGlass')
            .lin(nFeats, nFeats, name = 'Stack4_ll1')
            .lin(nFeats, nFeats, name = 'Stack4_ll2')
            .AttentionPartsCRF(nFeats, LRNKer, 3, 0, name = 'Stack4_att')
            .conv_layer(1, 1, n_classes, 1, 1, biased=True, relu=False, name = 'Stack4_tmpOut', padding='SAME'))

        (self.feed('Stack4_tmpOut')
            .conv_layer(1, 1, 256, 1, 1, biased=True, relu=False, name = 'Stack4_outmap', padding='SAME'))

        (self.feed('Stack4_ll1')
            .lin(nFeats, nFeats, name = 'Stack4_ll3'))

        (self.feed('Stack4_input',
                    'Stack4_outmap',
                    'Stack4_ll3')
            .add(name = 'Stack5_input'))


                                           ##   Stack 5   ##
        (self.feed('Stack5_input')
            .hourglass(nPool, nFeats, outputRes, nModule, name = 'Stack5_HourGlass')
            .lin(nFeats, nFeats, name = 'Stack5_ll1')
            .lin(nFeats, nFeats, name = 'Stack5_ll2')
            .AttentionPartsCRF(nFeats, LRNKer, 3, 0, name = 'Stack5_att')
            .AttentionPartsCRF(nFeats, LRNKer, 3, 1, name = 'Stack5_tmpOut'))

        (self.feed('Stack5_tmpOut')
            .conv_layer(1, 1, 256, 1, 1, biased=True, relu=False, name = 'Stack5_outmap', padding='SAME'))

        (self.feed('Stack5_ll1')
            .lin(nFeats, nFeats, name = 'Stack5_ll3'))

        (self.feed('Stack5_input',
                    'Stack5_outmap',
                    'Stack5_ll3')
            .add(name = 'Stack6_input'))



                                           ##   Stack 6   ##
        (self.feed('Stack6_input')
            .hourglass(nPool, nFeats, outputRes, nModule, name = 'Stack6_HourGlass')
            .lin(nFeats, nFeats, name = 'Stack6_ll1')
            .lin(nFeats, nFeats, name = 'Stack6_ll2')
            .AttentionPartsCRF(nFeats, LRNKer, 3, 0, name = 'Stack6_att')
            .AttentionPartsCRF(nFeats, LRNKer, 3, 1, name = 'Stack6_tmpOut'))

        (self.feed('Stack6_tmpOut')
            .conv_layer(1, 1, 256, 1, 1, biased=True, relu=False, name = 'Stack6_outmap', padding='SAME'))

        (self.feed('Stack6_ll1')
            .lin(nFeats, nFeats, name = 'Stack6_ll3'))

        (self.feed('Stack6_input',
                    'Stack6_outmap',
                    'Stack6_ll3')
            .add(name = 'Stack7_input'))



                                           ##   Stack 7   ##
        (self.feed('Stack7_input')
            .hourglass(nPool, nFeats, outputRes, nModule, name = 'Stack7_HourGlass')
            .lin(nFeats, nFeats, name = 'Stack7_ll1')
            .lin(nFeats, nFeats, name = 'Stack7_ll2')
            .AttentionPartsCRF(nFeats, LRNKer, 3, 0, name = 'Stack7_att')
            .AttentionPartsCRF(nFeats, LRNKer, 3, 1, name = 'Stack7_tmpOut'))

        (self.feed('Stack7_tmpOut')
            .conv_layer(1, 1, 256, 1, 1, biased=True, relu=False, name = 'Stack7_outmap', padding='SAME'))

        (self.feed('Stack7_ll1')
            .lin(nFeats, nFeats, name = 'Stack7_ll3'))

        (self.feed('Stack7_input',
                    'Stack7_outmap',
                    'Stack7_ll3')
            .add(name = 'Stack8_input'))


                                           ##   Stack 8   ##
        (self.feed('Stack8_input')
            .hourglass(nPool, nFeats, outputRes, nModule, name = 'Stack8_HourGlass')
            .lin(nFeats, nFeats*2, name = 'Stack8_ll1')
            .lin(nFeats*2, nFeats*2, name = 'Stack8_ll2')
            .AttentionPartsCRF(nFeats*2, LRNKer, 3, 0, name = 'Stack8_att')
            .AttentionPartsCRF(nFeats*2, LRNKer, 3, 1, name = 'Stack8_tmpOut'))







  