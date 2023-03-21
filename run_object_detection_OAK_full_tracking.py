#!/usr/bin/env python3

import argparse
import threading
from grasp_int import HandDetectors as hd
from grasp_int import Object2DDetectors as o2d
from grasp_int import ObjectPoseEstimators as ope
from grasp_int import Devices as dv
from grasp_int import Scene as sc
import torch
import gc
import os
import cv2
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def report_gpu():
   print(torch.cuda.list_gpu_processes())
   gc.collect()
   print(torch.cuda.memory_snapshot())
   torch.cuda.empty_cache()

class GraspingDetector:
    def __init__(self, device_name='OAK', hand_detection='mediapipe',  object_detection='cosypose') -> None:
        
        self.hand_detection_mode = hand_detection
        self.object_detection_mode = object_detection
        self.device_name = device_name
        self.device = dv.get_device(device_name)
        self.object_detector = o2d.get_object_detector(object_detection, self.device)
        self.object_pose_estimator = ope.get_pose_estimator(object_detection, self.device, use_tracking = True)
        self.scene = sc.Scene(device=self.device, name = 'Full tracking')
        self.object_detections = None
        self.detect = True
        if self.device_name != 'OAK' and self.hand_detection_mode != 'mediapipe':
            print('depthai may only be used on OAK device')
            raise ValueError


    def run(self):
        print(self.__dict__)
        self.device.start()
        print('start')
        while self.device.isOn():
            success, img = self.device.next_frame()
            if not success:
                continue     

            if self.detect :
                self.object_detections = self.object_detector.detect(img)

                if self.object_detections is not None:
                    self.detect=False
            else:
                self.object_detections = None
            self.objects_pose = self.object_pose_estimator.estimate(img, detections = self.object_detections)
            self.scene.update_objects(self.objects_pose)
            img.flags.writeable = True
            k = cv2.waitKey(5)
            if k == 32:
                self.detect=True
            if self.scene.render(img):
                print('end')
                self.stop()
                break
        exit()

    def stop(self):
        self.device.stop()
        self.object_detector.stop()
        exit()


        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device_name', choices=['OAK', 'monocular_webcam', 'stereo_webcam'],
                        default='OAK', help="Video input device")
    parser.add_argument('-hd', '--hand_detection', choices=['mediapipe', 'depthai'],
                        default = 'mediapipe', help="Hand pose reconstruction solution")
    parser.add_argument('-od', '--object_detection', choices=['cosypose, megapose'],
                        default = 'cosypose', help="Object pose reconstruction detection")
    args = vars(parser.parse_args())

    # if args.hand_detection == 'mediapipe':
    #     import mediapipe as mp
    # else:
    #     import depthai as dai
    
    # if args.object_detection == 'cosypose':
    #     import cosypose
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    report_gpu()
    grasp_int = GraspingDetector(**args)
    grasp_int.run()