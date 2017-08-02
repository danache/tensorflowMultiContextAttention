from __future__ import print_function
import os
import time
import tensorflow as tf
import numpy as np
import random
from utils import *
from LIP_model import *
import matplotlib.pyplot as plt

# Set gpus
gpus = [0,3] # Here I set CUDA to only see one GPU
os.environ["CUDA_VISIBLE_DEVICES"]=','.join([str(i) for i in gpus])
num_gpus = len(gpus) # number of GPUs to use

### parameters setting
N_CLASSES = 16
INPUT_SIZE = (256, 256)
BATCH_SIZE = 4
BATCH_I = 2
SHUFFLE = True
RANDOM_SCALE = False
RANDOM_MIRROR = True
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
POWER = 0.9
NUM_STEPS = 3808 * 40 + 1
SAVE_PRED_EVERY = 3808 
p_Weight = 1
s_Weight = 1
DATA_DIR = './datasets/human'
LIST_PATH = './datasets/human/list/train_rev.txt'
DATA_ID_LIST = './datasets/human/list/train_id.txt'
SNAPSHOT_DIR = './checkpoint/hg'
RESTORE_FROM = './model/hg'
LOG_DIR = './logs/hg'


def main():
    RANDOM_SEED = random.randint(1000, 9999)
    tf.set_random_seed(RANDOM_SEED)

    ## Create queue coordinator.
    coord = tf.train.Coordinator()

    ## Load reader.
    with tf.name_scope("create_inputs"):
        reader = ParsingPoseReader(DATA_DIR, LIST_PATH, DATA_ID_LIST, INPUT_SIZE, RANDOM_SCALE, RANDOM_MIRROR, SHUFFLE, coord)
        image_batch, label_batch, heatmap_batch = reader.dequeue(BATCH_SIZE)
        heatmap_batch = tf.scalar_mul(1.0/255, tf.cast(heatmap_batch, tf.float32))
        # heatmap = tf.reduce_sum(heatmap_batch, 3)

    tower_grads = []
    reuse1 = False
    ## Define loss and optimisation parameters.
    base_lr = tf.constant(LEARNING_RATE)
    step_ph = tf.placeholder(dtype=tf.float32, shape=())
    learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - step_ph / NUM_STEPS), POWER))
    optim = tf.train.RMSPropOptimizer(learning_rate)

    for i in xrange (num_gpus):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('Tower_%d' % (i)) as scope:
                if i == 0:
                    reuse1 = False
                else:
                    reuse1 = True
                next_image = image_batch[i*BATCH_I:(i+1)*BATCH_I,:]
                next_heatmap = heatmap_batch[i*BATCH_I:(i+1)*BATCH_I,:]

                ## Create network.
                with tf.variable_scope('', reuse=reuse1):
                    net = StackedHourglassModel({'data': next_image}, is_training=False, n_classes=N_CLASSES)
                    
                    hg1_out = net.layers['Stack1_Heatmap']
                    hg2_out = net.layers['Stack2_Heatmap']
                    hg3_out = net.layers['Stack3_Heatmap']
                    hg4_out = net.layers['Stack4_Heatmap']
                    hg5_out = net.layers['Stack5_Heatmap']
                    hg6_out = net.layers['Stack6_Heatmap']
                    hg7_out = net.layers['Stack7_Heatmap']
                    hg8_out = net.layers['Stack8_Heatmap']

                loss_hg1 = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(next_heatmap, hg1_out)), [1, 2, 3])))
                loss_hg2 = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(next_heatmap, hg2_out)), [1, 2, 3])))
                loss_hg3 = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(next_heatmap, hg3_out)), [1, 2, 3])))
                loss_hg4 = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(next_heatmap, hg4_out)), [1, 2, 3])))
                loss_hg5 = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(next_heatmap, hg5_out)), [1, 2, 3])))
                loss_hg6 = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(next_heatmap, hg6_out)), [1, 2, 3])))
                loss_hg7 = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(next_heatmap, hg7_out)), [1, 2, 3])))
                loss_hg8 = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(next_heatmap, hg8_out)), [1, 2, 3])))

                reduced_loss = loss_hg1 + loss_hg2 + loss_hg3 + loss_hg4 + loss_hg5 + loss_hg6 + loss_hg7 + loss_hg8
                
                trainable_variable = tf.trainable_variables()
                grads = optim.compute_gradients(reduced_loss, var_list=trainable_variable)
                
                tower_grads.append(grads)

                tf.add_to_collection('loss_hg1', loss_hg1)
                tf.add_to_collection('loss_hg2', loss_hg2)
                tf.add_to_collection('loss_hg3', loss_hg3)
                tf.add_to_collection('loss_hg4', loss_hg4)
                tf.add_to_collection('loss_hg5', loss_hg5)
                tf.add_to_collection('loss_hg6', loss_hg6)
                tf.add_to_collection('loss_hg7', loss_hg7)
                tf.add_to_collection('loss_hg8', loss_hg8)
                tf.add_to_collection('reduced_loss', reduced_loss)

    ## Average the gradients
    grads_ave = average_gradients(tower_grads)
    ## apply the gradients with our optimizers
    train_op = optim.apply_gradients(grads_ave)

    loss_hg1_ave = tf.reduce_mean(tf.get_collection('loss_hg1'))
    loss_hg2_ave = tf.reduce_mean(tf.get_collection('loss_hg2'))
    loss_hg3_ave = tf.reduce_mean(tf.get_collection('loss_hg3'))
    loss_hg4_ave = tf.reduce_mean(tf.get_collection('loss_hg4'))
    loss_hg5_ave = tf.reduce_mean(tf.get_collection('loss_hg5'))
    loss_hg6_ave = tf.reduce_mean(tf.get_collection('loss_hg6'))
    loss_hg7_ave = tf.reduce_mean(tf.get_collection('loss_hg7'))
    loss_hg8_ave = tf.reduce_mean(tf.get_collection('loss_hg8'))

    loss_ave = tf.reduce_mean(tf.get_collection('reduced_loss'))

    loss_summary_hg1 = tf.summary.scalar("loss_hg1_ave", loss_hg1_ave)
    loss_summary_hg2 = tf.summary.scalar("loss_hg2_ave", loss_hg2_ave)
    loss_summary_hg3 = tf.summary.scalar("loss_hg3_ave", loss_hg3_ave)
    loss_summary_hg4 = tf.summary.scalar("loss_hg4_ave", loss_hg4_ave)
    loss_summary_hg5 = tf.summary.scalar("loss_hg5_ave", loss_hg5_ave)
    loss_summary_hg6 = tf.summary.scalar("loss_hg6_ave", loss_hg6_ave)
    loss_summary_hg7 = tf.summary.scalar("loss_hg7_ave", loss_hg7_ave)
    loss_summary_hg8 = tf.summary.scalar("loss_hg8_ave", loss_hg8_ave)

    loss_summary_ave = tf.summary.scalar("loss_ave", loss_ave)
    loss_summary = tf.summary.merge([loss_summary_ave, loss_summary_hg1, loss_summary_hg2, loss_summary_hg3, 
                                     loss_summary_hg4, loss_summary_hg5, loss_summary_hg6, loss_summary_hg7, loss_summary_hg8])
    summary_writer = tf.summary.FileWriter(LOG_DIR, graph=tf.get_default_graph())

    ## Set up tf session and initialize variables.
    config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)

    ## Saver for storing checkpoints of the model.
    all_saver_var = tf.global_variables()
    restore_var = [v for v in all_saver_var if 'pose' not in v.name and 'Momentum' not in v.name and 'parsing' not in v.name]
    saver = tf.train.Saver(var_list=all_saver_var, max_to_keep=50)
    loader = tf.train.Saver(var_list=restore_var)

    if load(loader, sess, RESTORE_FROM):
        print(" [*] Load SUCCESS")
    else:
        print(" [!] Load failed...")    

    ## Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    ## Iterate over training steps.
    for step in range(NUM_STEPS):
        start_time = time.time()
        loss_value = 0
        feed_dict = { step_ph : step }

        ## Apply gradients.
        summary, loss_value, _ = sess.run([loss_summary, reduced_loss, train_op], feed_dict=feed_dict)
        summary_writer.add_summary(summary, step)
        if step % SAVE_PRED_EVERY == 0:
            save(saver, sess, SNAPSHOT_DIR, step)

        duration = time.time() - start_time
        print('step {:d} \t loss = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, duration))


    # for step in range(NUM_STEPS):
    #     imgmap, fullmap = sess.run([image_batch, heatmap_batch])
    # #     print labelmap.shape
    # #     print imgmap.shape
    #     # print (fullmap.shape)

    #     fig = plt.figure()
    #     a=fig.add_subplot(1,2,1)
    #     plt.imshow(imgmap[0,:,:,0])
    #     a=fig.add_subplot(1,2,2)
    #     plt.imshow(fullmap[0,:,:,0])

    #     plt.show()

    coord.request_stop()
    coord.join(threads)


def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

if __name__ == '__main__':
    main()


##################show examples codes#####################
    # Iterate over training steps.
    # for step in range(NUM_STEPS):
    #     imgmap, labelmap, fullmap = sess.run([image_batch, label_batch, heatmap])
    # #     print labelmap.shape
    # #     print imgmap.shape
    #     # print (fullmap.shape)

    #     fig = plt.figure()
    #     a=fig.add_subplot(1,3,1)
    #     plt.imshow(imgmap[0,:,:,0])
    #     a=fig.add_subplot(1,3,2)
    #     plt.imshow(labelmap[0,:,:,0])
    #     a=fig.add_subplot(1,3,3)
    #     plt.imshow(fullmap[0,:,:])

    #     plt.show()
