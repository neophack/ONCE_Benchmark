#!/usr/bin/python
import sys
import os
import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from pcdet.models import build_network

def input_callback(ros_cloud):
    points_list = []

    for data in pc2.read_points(ros_cloud, skip_nans=True):
        points_list.append([data[0], data[1], data[2], data[3]])
    points_array = np.array(points_list, dtype=np.float32)
    print(points_array.shape)

if __name__ == '__main__':
    rospy.init_node('once_detector', anonymous=True)
    rospy.Subscriber('/Geometry_Data_of_Detection', PointCloud2, input_callback) # 
    
    MODEL.PRE_PATH = '/home/work/user-job-dir/PCDet/checkpoints/checkpoint_epoch_20.pth'
    
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test)
    model.cuda()