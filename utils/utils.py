from PIL import Image
import numpy as np
import tensorflow as tf
import os
import scipy.misc
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

n_classes = 20
# colour map
label_colours = [(0,0,0)
                # 0=Background
                ,(128,0,0),(255,0,0),(0,85,0),(170,0,51),(255,85,0)
                # 1=Hat,  2=Hair,    3=Glove, 4=Sunglasses, 5=UpperClothes
                ,(0,0,85),(0,119,221),(85,85,0),(0,85,85),(85,51,0)
                # 6=Dress, 7=Coat, 8=Socks, 9=Pants, 10=Jumpsuits
                ,(52,86,128),(0,128,0),(0,0,255),(51,170,221),(0,255,255)
                # 11=Scarf, 12=Skirt, 13=Face, 14=LeftArm, 15=RightArm
                ,(85,255,170),(170,255,85),(255,255,0),(255,170,0)]
                # 16=LeftLeg, 17=RightLeg, 18=LeftShoe, 19=RightShoe
# image mean
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
    
def decode_labels(mask, num_images=1):
    """Decode batch of segmentation masks.
    
    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
    
    Returns:
      A batch with num_images RGB images of the same size as the input. 
    """
    n, h, w, c = mask.shape
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
      img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
      pixels = img.load()
      for j_, j in enumerate(mask[i, :, :, 0]):
          for k_, k in enumerate(j):
              if k < n_classes:
                  pixels[k_,j_] = label_colours[k]
      outputs[i] = np.array(img)
    return outputs

def prepare_label(input_batch, new_size, one_hot=True):
    """Resize masks and perform one-hot encoding.

    Args:
      input_batch: input tensor of shape [batch_size H W 1].
      new_size: a tensor with new height and width.

    Returns:
      Outputs a tensor of shape [batch_size h w 21]
      with last dimension comprised of 0's and 1's only.
    """
    with tf.name_scope('label_encode'):
        input_batch = tf.image.resize_nearest_neighbor(input_batch, new_size) # as labels are integer numbers, need to use NN interp.
        input_batch = tf.squeeze(input_batch, squeeze_dims=[3]) # reducing the channel dimension.
        if one_hot:
          input_batch = tf.one_hot(input_batch, depth=n_classes)
    return input_batch

def inv_preprocess(imgs, num_images):
  """Inverse preprocessing of the batch of images.
     Add the mean vector and convert from BGR to RGB.
   
  Args:
    imgs: batch of input images.
    num_images: number of images to apply the inverse transformations on.
  
  Returns:
    The batch of the size num_images with the same spatial dimensions as the input.
  """
  n, h, w, c = imgs.shape
  assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
  outputs = np.zeros((num_images, h, w, c), dtype=np.uint8)
  for i in range(num_images):
    outputs[i] = (imgs[i] + IMG_MEAN)[:, :, ::-1].astype(np.uint8)
  return outputs


def save(saver, sess, logdir, step):
    '''Save weights.   
    Args:
     saver: TensorFlow Saver object.
     sess: TensorFlow session.
     logdir: path to the snapshots directory.
     step: current training step.
    '''
    if not os.path.exists(logdir):
        os.makedirs(logdir)   
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
      
    if not os.path.exists(logdir):
      os.makedirs(logdir)
    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')

def load(saver, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    ''' 
    ckpt = tf.train.get_checkpoint_state(ckpt_path)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(ckpt_path, ckpt_name))
        print("Restored model parameters from {}".format(ckpt_name))
        return True
    else:
        return False  

def load_lip_data(image_id, phrase):
    parsing_size = 368
    pose_size = 46
    image_id = image_id[:-1] 
    # print image_id
    image_path = './datasets/human/images/{}.jpg'.format(image_id)
    img = scipy.misc.imread(image_path).astype(np.float)
    parsing_path = './datasets/human/labels/{}.png'.format(image_id)
    parsing = scipy.misc.imread(parsing_path).astype(np.float)
    rows = img.shape[0]
    cols = img.shape[1]
    origin_g = scipy.misc.imresize(img, [parsing_size, parsing_size])
    parsing_g = scipy.misc.imresize(parsing, [parsing_size, parsing_size])
    # img_g = np.concatenate((origin_g, parsing_g[:,:,np.newaxis]), axis=2)
    if phrase == 'test':
        # return img_g
        return origin_g[np.newaxis,:,:,:]
    parsing_d = scipy.misc.imresize(parsing, [pose_size, pose_size])
    heatmap = np.zeros((pose_size, pose_size, 16), dtype=np.float64)
    with open('./datasets/human/pose/{}.txt'.format(image_id), 'r') as f:
        lines = f.readlines()
    points = lines[0].split(',')
    for idx, point in enumerate(points):
        if idx % 2 == 0:
            c_ = int(point)
            c_ = min(c_, cols-1)
            c_ = max(c_, 0)
            c_ = int(pose_size * 1.0 * c_ / cols)
        else:
            r_ = int(point)
            r_ = min(r_, rows-1)
            r_ = max(r_, 0)
            r_ = int(pose_size * 1.0 * r_ / rows)
            if c_ + r_ == 0:
                heatmap[:,:,int(idx / 2)] = 0
                continue
            var = multivariate_normal(mean=[r_, c_], cov=2)
            l1 = max(r_-5, 0)
            r1 = min(r_+5, pose_size-1)
            l2 = max(c_-5, 0)
            r2 = min(c_+5, pose_size-1)
            for i in xrange(l1, r1):
                for j in xrange(l2, r2):
                    heatmap[i, j, int(idx / 2)] = var.pdf([i, j]) * 10
    # heatsum = np.sum(heatmap, axis=2)
    # plt.clf()
    # # plt.imshow(heatmap[:,:,int(idx/2)].T)
    # plt.imshow(heatsum)
    # plt.show()
    # wait = raw_input()

    # origin_g = origin_g / 127.5 - 1.
    # origin_d = origin_d / 127.5 - 1.
    
    # img_d = np.concatenate((parsing_d[:,:,np.newaxis], heatmap), axis=2)
    # return img_g, img_d
    return origin_g, heatmap

def save_lip_images(img_id, samples, output_set):
    img_id = img_id[:-1]
    image_path = './datasets/human/masks/{}.png'.format(img_id)
    print img_id
    img_A = scipy.misc.imread(image_path).astype(np.float)
    rows = img_A.shape[0]
    cols = img_A.shape[1]
    image = samples[0]
    with open('./output/pose/{}/{}.txt'.format(output_set, img_id), 'w') as f:
        for p in xrange(image.shape[2]):
            channel_ = image[:,:,p]
            channel_ = scipy.misc.imresize(channel_, [rows, cols])
            r_, c_ = np.unravel_index(channel_.argmax(), channel_.shape)
                # r_ = r_ * rows * 1.0 / channel_.shape[0]
                # c_ = c_ * cols * 1.0 / channel_.shape[1]
                # save_path = './{}/pose/{}_{}.png'.format(output_set, img_id, point_name[p])
                # scipy.misc.imsave(save_path, channel_)
            f.write('%d %d ' % (int(c_), int(r_)))
            # print ('id: {}, p_: {}, r_: {}, c_: {}'.format(p, p_, r_, c_))
            # plt.clf()
            # plt.imshow(channel_.T)
            # plt.show()
            # wait = raw_input()
        # resultmap = np.sum(image, axis=2)
        # plt.clf()
        # plt.imshow(resultmap)
        # plt.show()
    # sio.savemat('./{}/pose/{}.mat'.format(output_set, img_id), {'result': image})