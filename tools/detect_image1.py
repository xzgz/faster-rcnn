#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import time
print(cv2.__version__)
# CLASSES = ('__background__',
#            'aeroplane', 'bicycle', 'bird', 'boat',
#            'bottle', 'bus', 'car', 'cat', 'chair',
#            'cow', 'diningtable', 'dog', 'horse',
#            'motorbike', 'person', 'pottedplant',
#            'sheep', 'sofa', 'train', 'tvmonitor')
CLASSES = ('__background__', 'car')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
               'ZF_faster_rcnn_final.caffemodel')}


def vis_detections_video(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""

    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return im

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        cv2.rectangle(im,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),2)
        cv2.rectangle(im,(int(bbox[0]),int(bbox[1])-10),(int(bbox[0]+200),int(bbox[1])+10),(10,10,10),-1)
        cv2.putText(im,'{:s} {:.3f}'.format(class_name, score),(int(bbox[0]),int(bbox[1]-2)),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255))#,cv2.CV_AA)
    return im


def demo_video(net, im):
    """Detect object classes in an image using pre-computed object proposals."""
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
            '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.75

    NMS_THRESH = 0.2
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
 
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        im = vis_detections_video(im, cls, dets, thresh = CONF_THRESH)
    return im


'''Detect all videos of a specified directory'''
def detect_directory_video(video_path, result_path):
    if os.path.exists(video_path) == False:
        print 'no video file'
        return

    if os.path.exists(result_path) == False:
        os.makedirs(result_path)

    videonames = []
    video_prefixes = []
    for root, dirs, files in os.walk(video_path, topdown = False):
        for name in files:
            videonames.append(os.path.join(root, name))
            p = name.split('.')
            prefix = ''
            p = p[:-1]
            for x in p:
                prefix += x
            video_prefixes.append(os.path.join(result_path, prefix))

    # fourcc = cv2.cv.CV_FOURCC(*'XVID')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    for i in range(len(videonames)):
        print 'start detect video ' + videonames[i]
        video = cv2.VideoCapture(videonames[i])
        ret, image = video.read()
        video_detected_name = video_prefixes[i] + '_detected.avi'
        video_detected = cv2.VideoWriter(video_detected_name, fourcc, 25.0, (image.shape[1], image.shape[0]))
        while video.isOpened():
            ret, image = video.read()
            if ret == True:
                print image.shape
                image = demo_video(net, image)
                video_detected.write(image)
            else:
                print 'video ' + videonames[i] + ' detect complete'
                video.release()
                video_detected.release()
                break
    print 'done'


'''Detect all images of a specified directory'''
def detect_directory_image(image_path, result_path):
    if os.path.exists(image_path) == False:
        print 'no image file'
        return
        
    if os.path.exists(result_path) == False:
        os.makedirs(result_path)
        
    imagenames = []
    image_prefixes = []
    for root, dirs, files in os.walk(image_path, topdown = False):
        for name in files:
            imagenames.append(os.path.join(root, name))
            p = name.split('.')
            prefix = ''
            p = p[:-1]
            for x in p:
                prefix += x
            image_prefixes.append(os.path.join(result_path, prefix))

    for i in range(len(imagenames)):
        image = cv2.imread(imagenames[i])
        image = demo_video(net, image)
        print 'image ' + imagenames[i] + ' detect complete'
        image_detected_name = image_prefixes[i] + '_detected.jpg'
        cv2.imwrite(image_detected_name, image)
    print 'done'

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description = 'Faster R-CNN demo')
    parser.add_argument('--gpu', dest = 'gpu_id', help = 'GPU device id to use [0]',
                        default = 1, type = int)
    parser.add_argument('--cpu', dest = 'cpu_mode',
                        help = 'Use CPU mode (overrides --gpu)',
                        action = 'store_true')
    parser.add_argument('--net', dest = 'demo_net', help = 'Network to use [vgg16]',
                        choices = NETS.keys(), default = 'vgg16')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    # prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
    #                         'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    # caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
    #                           NETS[args.demo_net][1])
    prototxt = '/home/gysj/caffe-code/faster-rcnn/models/pascal_voc/VGG16/faster_rcnn_end2end/test.prototxt'
    caffemodel = '/home/gysj/caffe-code/faster-rcnn/output/faster_rcnn_end2end/voc_2007_trainval/' \
                 'vgg16_faster_rcnn_iter_70000.caffemodel'

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))
    
    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    print '\n\nLoaded network {:s}'.format(caffemodel)
    
    # detect_directory_image('/home/gysj/caffe-code/faster-rcnn/data/demo',
    #                        '/home/gysj/caffe-code/faster-rcnn/data/demo_result/det7w')
    detect_directory_video('/home/gysj/caffe-code/faster-rcnn/data/video',
                           '/home/gysj/caffe-code/faster-rcnn/data/video_result/det7w')









