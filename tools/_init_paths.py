# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Set up paths for Fast R-CNN."""

import os.path as osp
import sys
import os

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)
os.chdir(osp.join(this_dir, '..'))

# Add caffe to PYTHONPATH
caffe_path = osp.join(this_dir, '..', 'caffe-fast-rcnn', 'python')
# caffe_path = '/usr/lib/python2.7/dist-packages'
print('Add follwing caffe path to PYTHONPATH')
print(caffe_path)
add_path(caffe_path)
# faster_rcnn_root_path = osp.join(this_dir, '../')
# add_path(faster_rcnn_root_path)
# print(faster_rcnn_root_path)

# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, '..', 'lib')
print('Add follwing lib path to PYTHONPATH')
print(lib_path)
add_path(lib_path)
