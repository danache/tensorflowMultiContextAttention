import os
import os.path
import shutil
from xml.etree import ElementTree as ET
import numpy as np
import scipy.io as sio
from scipy.misc import imread
import json
import math

def get_aux_label(pose):
    
    head = np.array([pose[9*3+0] ,pose[9*3+1]])
    neck = np.array([pose[8*3+0] ,pose[8*3+1]])
    diff = head - neck
    height = np.sqrt(np.sum(diff*diff))  * 0.8

    scale = height / 25.0  # assuming 200 px height human has 25 pixel head
    if math.isnan(scale):
        scale = 1

    max_y = 0
    min_y = 100000
    max_x = 0
    min_x = 100000
    
    for i in [6,12,13]:
        if math.isnan(pose[3*i+1]) or math.isnan(pose[3*i+0]):
            continue

        if pose[3*i+1] > max_y:
            max_y = pose[3*i+1]
        if pose[3*i+1] < min_y:
            min_y = pose[3*i+1]

        if pose[3*i+0] > max_x:
            max_x = pose[3*i+0]
        if pose[3*i+0] < min_x:
            min_x = pose[3*i+0]

    sigma = height / 3
    
    center = np.random.normal([(max_x + min_x) / 2, (max_y + min_y) / 2], [sigma,sigma])
    center = [int(max(1,center[0])), int(max(1,center[1]))]
    
    return (center,scale)

def parse_xml(xml_filepath):
    # x,y,visible
    pose = [float('nan')] * 48

    point_index = ['R_Ankle','R_Knee','R_Hip','L_Hip','L_Knee','L_Ankle',
                    'B_Pelvis','B_Spine','B_Neck','B_Head',
                    'R_Wrist','R_Elbow','R_Shoulder','L_Shoulder','L_Elbow','L_Wrist'];

    with open(xml_filepath,'r') as xml_file:
      tree = ET.parse(xml_file)
      for node in tree.findall('.//keypoint'):
        name = node.attrib.get('name')
        try:
            index = point_index.index(name)
        except:

          continue
        x = float(int(float(node.attrib.get('x'))))
        y = float(int(float(node.attrib.get('y'))))
        # in order to follow the lsp annotation
        v = float(int((1+float(node.attrib.get('visible')))%2))
        pose[index*3] = int(max(1,x))
        pose[index*3 + 1] = int(max(1,y))
        pose[index*3 + 2] = int(v)
    return pose

def parse_anno():
    img_id = 1
    
    anno_ext = '_0.xml'

    anno_vec = []
    centers = []
    scales = []
    
    is_train = []

    imagenames = []
    save_mat = '../mat/lip_train_test.mat'
    img_root = '../LIP_dataset'
    dataset_num = 500000

    release = False
    for package_path in ['train_set', 'val_set', 'test_set']:
        
        img_folder = os.path.join(img_root,package_path,'images')
        anno_folder = os.path.join(img_root,package_path,'infos')
        for pa,sub,fns in os.walk(img_folder):

          for fn in fns:

            if img_id > dataset_num:
              break
            origin_img_path =  os.path.join(pa,fn)
            im = imread(origin_img_path)
            if len(im.shape) != 3:
                # only process on color image
                print 'Warning: not color at image ', origin_img_path

            img_ext = fn.split('.')[-1]
            img_pre = fn.split('.')[0]

            annotation_file = os.path.join(anno_folder,img_pre + anno_ext)
            if not os.path.exists(annotation_file):
                print 'File not exist, ', annotation_file
                continue
            ##
            ## start parse info
            ##
            anno = parse_xml(annotation_file)
            (center,scale) = get_aux_label(anno)
            
            if math.isnan(scale):
                print 'invalid at ', origin_img_path
                continue
            
            dst_img_path = os.path.join(package_path,fn)
            imagenames.append(dst_img_path.split('/')[-1])
            
            centers.append(center)
            scales.append(scale)
            img_id = img_id + 1
            if img_id % 1000 == 0:
              print 'processed ',img_id
            
            if 'test' in package_path and release: 
                anno  = [float('nan')] * 48

            anno_vec.append(anno)

            if 'test' in package_path : 
                is_train.append('0')
            elif 'train' in package_path:
                is_train.append('1')
            else:
                is_train.append('2')

            

    img_count  = img_id - 1
    print img_count
    anno_vec = np.array(anno_vec)
    anno_vec = np.reshape(anno_vec,(anno_vec.shape[0],anno_vec.shape[1]/3,3))
   
    anno_vec = anno_vec.transpose()
    sio.savemat(save_mat,{'RELEASE': {'joints':np.array(anno_vec),'image_names': imagenames,
                            'centers':np.array(centers),
                            'scales': np.array(scales).transpose(),'is_train': is_train }}, oned_as='column')
   
if __name__ == "__main__":
  parse_anno()
