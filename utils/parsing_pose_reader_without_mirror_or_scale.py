import os

import numpy as np
import tensorflow as tf

IGNORE_LABEL = 255
NUM_POSE = 16
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

def image_scaling(img, label, heatmap):
    """
    Randomly scales the images between 0.5 to 1.5 times the original size.

    Args:
      img: Training image to scale.
      label: Segmentation mask to scale.
    """
    
    scale = tf.random_uniform([1], minval=0.75, maxval=1.25, dtype=tf.float32, seed=None)
    h_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[0]), scale))
    w_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[1]), scale))
    
    new_shape = tf.squeeze(tf.stack([h_new, w_new]), squeeze_dims=[1])
    img = tf.image.resize_images(img, new_shape)
    label = tf.image.resize_nearest_neighbor(tf.expand_dims(label, 0), new_shape)
    label = tf.squeeze(label, squeeze_dims=[0])
    heatmap = tf.image.resize_images(heatmap, new_shape)
    return img, label, heatmap

def image_mirroring(img, label):
    """
    Randomly mirrors the images.

    Args:
      img: Training image to mirror.
      label: Segmentation mask to mirror.
    """
    
    distort_left_right_random = tf.random_uniform([1], 0, 1.0, dtype=tf.float32)[0]
    mirror = tf.less(tf.stack([1.0, distort_left_right_random, 1.0]), 0.5)
    img = tf.reverse(img, mirror)
    reversed_label = tf.reverse(label, mirror)

    return img, reversed_label

def resize_img_labels(image, label, heatmap, resized_h, resized_w, ignore_label=255):

    new_shape = tf.stack([tf.to_int32(resized_h), tf.to_int32(resized_w)])
    img = tf.image.resize_images(image, new_shape)
    label = tf.image.resize_nearest_neighbor(tf.expand_dims(label, 0), new_shape)
    label = tf.squeeze(label, squeeze_dims=[0])
    new_shape = tf.stack([tf.to_int32(resized_h / 8.0), tf.to_int32(resized_w / 8.0)])
    heatmap = tf.image.resize_images(heatmap, new_shape)
    return img, label, heatmap

def random_crop_and_pad_image_and_labels(image, label, heatmap, crop_h, crop_w, ignore_label=255):
    """
    Randomly crop and pads the input images.

    Args:
      image: Training image to crop/ pad.
      label: Segmentation mask to crop/ pad.
      crop_h: Height of cropped segment.
      crop_w: Width of cropped segment.
      ignore_label: Label to ignore during the training.
    """

    label = tf.cast(label, dtype=tf.float32)
    label = label - ignore_label # Needs to be subtracted and later added due to 0 padding.
    heatmap = tf.cast(heatmap, dtype=tf.float32)
    combined = tf.concat([image, label, heatmap], 2) 
    image_shape = tf.shape(image)
    combined_pad = tf.image.pad_to_bounding_box(combined, 0, 0, tf.maximum(crop_h, image_shape[0]), tf.maximum(crop_w, image_shape[1]))
    
    last_image_dim = tf.shape(image)[-1]
    last_label_dim = tf.shape(label)[-1]
    combined_crop = tf.random_crop(combined_pad, [crop_h,crop_w,4+NUM_POSE])
    img_crop = combined_crop[:, :, :last_image_dim]
    label_crop = combined_crop[:, :, last_image_dim:last_image_dim+last_label_dim]
    heatmap_crop = combined_crop[:, :, last_image_dim+last_label_dim:]
    label_crop = label_crop + ignore_label
    label_crop = tf.cast(label_crop, dtype=tf.uint8)
    
    # Set static shape so that tensorflow knows shape at compile time. 
    img_crop.set_shape((crop_h, crop_w, 3))
    label_crop.set_shape((crop_h,crop_w, 1))
    heatmap_crop.set_shape((crop_h, crop_w, NUM_POSE))
    return img_crop, label_crop, heatmap_crop  

def read_labeled_image_list(data_dir, data_list):
    """Reads txt file containing paths to images and ground truth masks.
    
    Args:
      data_dir: path to the directory with images and masks.
      data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.
       
    Returns:
      Two lists with all file names for images and masks, respectively.
    """
    f = open(data_list, 'r')
    images = []
    masks = []
    for line in f:
        try:
            image, mask = line.strip("\n").split(' ')
        except ValueError: # Adhoc for test.
            image = mask = line.strip("\n")
        images.append(data_dir + image)
        masks.append(data_dir + mask)
    return images, masks

def read_pose_list(data_dir, data_id_list):
    f = open(data_id_list, 'r')
    poses = []
    for line in f:
        pose = line.strip("\n")
        poses.append(data_dir + '/heatmap/' + pose)
    return poses


def read_images_from_disk(input_queue, input_size, random_scale, random_mirror=False): # optional pre-processing arguments
    """Read one image and its corresponding mask with optional pre-processing.
    
    Args:
      input_queue: tf queue with paths to the image and its mask.
      input_size: a tuple with (height, width) values.
                  If not given, return images of original size.
      random_scale: whether to randomly scale the images prior
                    to random crop.
      random_mirror: whether to randomly mirror the images prior
                    to random crop.
      
    Returns:
      Two tensors: the decoded image and its mask.
    """

    img_contents = tf.read_file(input_queue[0])
    label_contents = tf.read_file(input_queue[1])
    
    img = tf.image.decode_jpeg(img_contents, channels=3)
    img_r, img_g, img_b = tf.split(value=img, num_or_size_splits=3, axis=2)
    img = tf.cast(tf.concat([img_b, img_g, img_r], 2), dtype=tf.float32)
    # Extract mean.
    img -= IMG_MEAN

    label = tf.image.decode_png(label_contents, channels=1)

    pose_id = input_queue[2]
    pose = []
    for i in xrange(NUM_POSE):
        pose_contents = tf.read_file(pose_id+'_{}.png'.format(i))
        pose_i = tf.image.decode_png(pose_contents, channels=1)
        pose.append(pose_i)
    heatmap = tf.concat(pose, axis=2)

    if input_size is not None:
        h, w = input_size

        # Randomly scale the images and labels.
        if random_scale:
            img, label, heatmap = image_scaling(img, label, heatmap)

        # Randomly mirror the images and labels.

        if random_mirror:
            img, label = image_mirroring(img, label)

        # Randomly crops the images and labels.
        img, label, heatmap = resize_img_labels(img, label, heatmap, h, w, IGNORE_LABEL)
        # img, label, heatmap = random_crop_and_pad_image_and_labels(img, label, heatmap, h, w, IGNORE_LABEL)

    return img, label, heatmap

class ParsingPoseReader(object):
    '''Generic ImageReader which reads images and corresponding segmentation
       masks from the disk, and enqueues them into a TensorFlow queue.
    '''

    def __init__(self, data_dir, data_list, data_id_list, input_size, random_scale,
                 random_mirror, shuffle, coord):
        '''Initialise an ImageReader.
        
        Args:
          data_dir: path to the directory with images and masks.
          data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.
          data_id_list: path to the file of image id.
          input_size: a tuple with (height, width) values, to which all the images will be resized.
          random_scale: whether to randomly scale the images prior to random crop.
          random_mirror: whether to randomly mirror the images prior to random crop.
          coord: TensorFlow queue coordinator.
        '''
        self.data_dir = data_dir
        self.data_list = data_list
        self.data_id_list = data_id_list
        self.input_size = input_size
        self.coord = coord

        self.image_list, self.label_list = read_labeled_image_list(self.data_dir, self.data_list)
        self.pose_list = read_pose_list(self.data_dir, self.data_id_list)
        self.images = tf.convert_to_tensor(self.image_list, dtype=tf.string)
        self.labels = tf.convert_to_tensor(self.label_list, dtype=tf.string)
        self.poses = tf.convert_to_tensor(self.pose_list, dtype=tf.string)
        self.queue = tf.train.slice_input_producer([self.images, self.labels, self.poses], shuffle=shuffle) 
        self.image, self.label, self.heatmap = read_images_from_disk(self.queue, self.input_size, random_scale, random_mirror) 

    def dequeue(self, num_elements):
        '''Pack images and labels into a batch.
        
        Args:
          num_elements: the batch size.
          
        Returns:
          Two tensors of size (batch_size, h, w, {3, 1}) for images and masks.'''
        batch_list = [self.image, self.label, self.heatmap]
        image_batch, label_batch, heatmap_batch = tf.train.batch([self.image, self.label, self.heatmap], num_elements)
        return image_batch, label_batch, heatmap_batch
