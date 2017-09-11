from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]="3"
from utils import *
from LIP_model import *
import matplotlib.pyplot as plt
import scipy.misc
import scipy.io as sio


NUM_STEPS = 10000 # Number of images in the validation set.
BATCH_SIZE = 2
INPUT_SIZE = (256, 256)
N_CLASSES = 16
DATA_DIR = './datasets/human'
LIST_PATH = './datasets/human/list/test.txt'
DATA_ID_LIST = './datasets/human/list/test_id.txt'
RESTORE_FROM = './checkpoint/joint_resnet'

def main():
    """Create the model and start the evaluation process."""
    
    # Create queue coordinator.
    coord = tf.train.Coordinator()
    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ParsingPoseReader(DATA_DIR, LIST_PATH, DATA_ID_LIST, INPUT_SIZE, False, False, False, coord)
        image = reader.image
        image_list = reader.image_list
    image_batch = tf.expand_dims(image, dim=0)
    
    # Create network.
    with tf.variable_scope('', reuse=False):
        net = StackedHourglassModel({'data': image_batch}, is_training=False, n_classes=N_CLASSES)

    # parsing net
    # parsing_fea = net.layers['res5d_branch2c_parsing']
    Stack8_out = net.layers['Stack8_Heatmap']
    # pose net
    #resnet_fea = net.layers['res4b22_relu']
    #pose_out1, pose_fea = pose_net(resnet_fea, 'fc1_pose')

    #pos_par1 = tf.concat([pose_out1, parsing_out1, pose_fea], 3)
    #pose_out2 = cpm_stage_x(pos_par1, name='fc2_pose')

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
        predict_ = sess.run(Stack8_out)
        save_lip_images(image_list[step], predict_)
        if step % 100 == 0:
            print('step {:d}'.format(step))
            print (image_list[step])

    coord.request_stop()
    coord.join(threads)
   

def save_lip_images(image_path, samples, output_set='test'):
    img_A = scipy.misc.imread(image_path).astype(np.float)
    rows = img_A.shape[0]
    cols = img_A.shape[1]
    image = samples[0]
    img_split = image_path.split('/')
    img_id = img_split[-1][:-4]
    with open('./output/pose/{}/{}.txt'.format(output_set, img_id), 'w') as f:
        for p in xrange(image.shape[2]):
            channel_ = image[:,:,p]
            channel_ = scipy.misc.imresize(channel_, [rows, cols])
            r_, c_ = np.unravel_index(channel_.argmax(), channel_.shape)
            f.write('%d %d ' % (int(c_), int(r_)))


if __name__ == '__main__':
    main()


##################show examples codes#####################33
    # Iterate over training steps.
    # for step in range(NUM_STEPS):
    #     imgmap, resultmap = sess.run([image_batch, outmap])
    # #     print labelmap.shape
    # #     print imgmap.shape
    # #     print fullmap.shape

    #     fig = plt.figure()
    #     a=fig.add_subplot(1,2,1)
    #     plt.imshow(imgmap[0,:,:,0])
    #     a=fig.add_subplot(1,2,2)
    #     plt.imshow(resultmap[0,:,:])
    #     plt.show()       
