from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from utils import *
from LIP_model import *
import matplotlib.pyplot as plt
import scipy.misc
import scipy.io as sio


NUM_STEPS = 2001 # Number of images in the validation set.
BATCH_SIZE = 1
INPUT_SIZE = (256, 256)
N_CLASSES = 16
DATA_DIRECTORY = './datasets/mpii'
DATA_LIST_PATH = './datasets/mpii/list/train_2000.txt'
RESTORE_FROM = './checkpoint/hg'

def main():
    """Create the model and start the evaluation process."""
    
    # Create queue coordinator.
    coord = tf.train.Coordinator()
    h, w = INPUT_SIZE
    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(DATA_DIRECTORY, DATA_LIST_PATH, None, False, False, coord)
        image = reader.image
        image_rev = tf.reverse(image, tf.stack([1]))
        image_list = reader.image_list
    
    image_batch_origin = tf.stack([image, image_rev])
    image_batch = tf.image.resize_images(image_batch_origin, [int(h), int(w)])
    image_batch125 = tf.image.resize_images(image_batch_origin, [int(h * 1.25), int(w * 1.25)])
    image_batch075 = tf.image.resize_images(image_batch_origin, [int(h * 0.75), int(w * 0.75)])
    # image_batch150 = tf.image.resize_images(image_batch_origin, [int(h * 1.50), int(w * 1.50)])
    # image_batch050 = tf.image.resize_images(image_batch_origin, [int(h * 0.50), int(w * 0.50)])

    # Create network.
    with tf.variable_scope('', reuse=False):
        net_100 = MultiContextAttentionModel({'data': image_batch}, is_training=False, n_classes=N_CLASSES)
    with tf.variable_scope('', reuse=True):
        net_125 = MultiContextAttentionModel({'data': image_batch125}, is_training=False, n_classes=N_CLASSES)
    with tf.variable_scope('', reuse=True):
        net_075 = MultiContextAttentionModel({'data': image_batch075}, is_training=False, n_classes=N_CLASSES)
    # with tf.variable_scope('', reuse=True):
    #     net_150 = DeepLabResNetModel({'data': image_batch150}, is_training=False, n_classes=N_CLASSES)
    # with tf.variable_scope('', reuse=True):
    #     net_050 = DeepLabResNetModel({'data': image_batch050}, is_training=False, n_classes=N_CLASSES)

    # parsing net
    #parsing_out1_100 = net_100.layers['fc1_human']
    #parsing_out1_125 = net_125.layers['fc1_human']
    #parsing_out1_075 = net_075.layers['fc1_human']
    # parsing_out1_150 = net_150.layers['fc1_human']
    # parsing_out1_050 = net_050.layers['fc1_human']

    # pose net
    hgout8_100 = net_100.layers['Stack8_tmpOut']
    hgout8_125 = net_125.layers['Stack8_tmpOut']
    hgout8_075 = net_075.layers['Stack8_tmpOut']
    # resnet_fea_150 = net_150.layers['res4b22_relu']
    # resnet_fea_050 = net_050.layers['res4b22_relu']
    
    #with tf.variable_scope('', reuse=False):
    #    pose_out1_100, pose_fea_100 = pose_net(resnet_fea_100, 'fc1_pose')
    #    pose_out2_100 = pose_refine(pose_out1_100, parsing_out1_100, pose_fea_100, name='fc2_pose')
    #with tf.variable_scope('', reuse=True):
    #    pose_out1_125, pose_fea_125 = pose_net(resnet_fea_125, 'fc1_pose')
    #    pose_out2_125 = pose_refine(pose_out1_125, parsing_out1_125, pose_fea_125, name='fc2_pose')
    #with tf.variable_scope('', reuse=True):
    #    pose_out1_075, pose_fea_075 = pose_net(resnet_fea_075, 'fc1_pose')
    #    pose_out2_075 = pose_refine(pose_out1_075, parsing_out1_075, pose_fea_075, name='fc2_pose')
    # with tf.variable_scope('', reuse=True):
    #     pose_out1_150, pose_fea_150 = pose_net(resnet_fea_150, 'fc1_pose')
    #     pose_out2_150 = pose_refine(pose_out1_150, parsing_out1_150, pose_fea_150, name='fc2_pose')
    # with tf.variable_scope('', reuse=True):
    #     pose_out1_050, pose_fea_050 = pose_net(resnet_fea_050, 'fc1_pose')
    #     pose_out2_050 = pose_refine(pose_out1_050, parsing_out1_050, pose_fea_050, name='fc2_pose')

    pose_out2 = tf.reduce_mean(tf.stack([tf.image.resize_nearest_neighbor(hgout8_100, tf.shape(image_batch_origin)[1:3,]),
                                         tf.image.resize_nearest_neighbor(hgout8_125, tf.shape(image_batch_origin)[1:3,]),
                                         tf.image.resize_nearest_neighbor(hgout8_075, tf.shape(image_batch_origin)[1:3,])]), axis=0)
                                         # tf.image.resize_nearest_neighbor(pose_out2_150, tf.shape(image_batch_origin)[1:3,]),
                                         # tf.image.resize_nearest_neighbor(pose_out2_050, tf.shape(image_batch_origin)[1:3,])]), axis=0)

    head_output, tail_output = tf.unstack(pose_out2, num=2, axis=0)
    tail_list = tf.unstack(tail_output, num=16, axis=2)
    tail_list_rev = [None] * 16
    tail_list_rev[0] = tail_list[5]
    tail_list_rev[1] = tail_list[4]
    tail_list_rev[2] = tail_list[3]
    tail_list_rev[3] = tail_list[2]
    tail_list_rev[4] = tail_list[1]
    tail_list_rev[5] = tail_list[0]
    tail_list_rev[10] = tail_list[15]
    tail_list_rev[11] = tail_list[14]
    tail_list_rev[12] = tail_list[13]
    tail_list_rev[13] = tail_list[12]
    tail_list_rev[14] = tail_list[11]
    tail_list_rev[15] = tail_list[10]
    tail_list_rev[6] = tail_list[6]
    tail_list_rev[7] = tail_list[7]
    tail_list_rev[8] = tail_list[8]
    tail_list_rev[9] = tail_list[9]
    tail_output_rev = tf.stack(tail_list_rev, axis=2)
    tail_output_rev = tf.reverse(tail_output_rev, tf.stack([1]))
    # final_output =  tf.stack([head_output, tail_output_rev], axis=0)

    output_all = tf.reduce_mean(tf.stack([head_output, tail_output_rev]), axis=0)
    output_all = tf.expand_dims(output_all, dim=0)
    # outmap_all = tf.reduce_sum(output_all, 3)

    # Which variables to load.
    restore_var = tf.global_variables()

    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    
    # Load weights.
    loader = tf.train.Saver(var_list=restore_var)
    if RESTORE_FROM is not None:
        if load(loader, sess, RESTORE_FROM):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
    
    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # Iterate over training steps.
    for step in range(NUM_STEPS):
        predict_ = sess.run(output_all)
        save_lip_images(image_list[step], predict_)
        if step % 100 == 0:
            print('step {:d}'.format(step))
            print (image_list[step])

    coord.request_stop()
    coord.join(threads)
   

def save_lip_images(image_path, samples, output_set='test_hg'):
    img_A = scipy.misc.imread(image_path).astype(np.float)
    rows = img_A.shape[0]
    cols = img_A.shape[1]
    image = samples[0]
    img_split = image_path.split('/')
    img_id = img_split[-1][:-4]
    with open('./output/pose/{}/{}.txt'.format(output_set, img_id), 'w') as f:
        for p in xrange(image.shape[2]):
            channel_ = image[:,:,p]
            if channel_.shape[0] != rows or channel_.shape[1] != cols:
                print ('sizes do not match...')
                channel_ = scipy.misc.imresize(channel_, [rows, cols], interp='nearest')
            r_, c_ = np.unravel_index(channel_.argmax(), channel_.shape)
            f.write('%d %d ' % (int(c_), int(r_)))


if __name__ == '__main__':
    main()


##################show examples codes#####################33
    # Iterate over training steps.
    # for step in range(NUM_STEPS):

        # imgmap, resultmap, resmap = sess.run([image_batch, outmap, outmap_all])
    #     print labelmap.shape
    #     print imgmap.shape
    #     print fullmap.shape
        # fig = plt.figure()
        # a=fig.add_subplot(1,4,1)
        # plt.imshow(imgmap[0,:,:,0])
        # a=fig.add_subplot(1,4,2)
        # plt.imshow(resultmap[0,:,:])
        # a=fig.add_subplot(1,4,3)
        # plt.imshow(resultmap[1,:,:])
        # a=fig.add_subplot(1,4,4)
        # plt.imshow(resmap[0,:,:])
        # plt.show()       
