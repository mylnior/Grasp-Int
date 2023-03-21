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
    def __init__(self, device_name='OAK', hand_detection='OAK',  object_detection='cosypose') -> None:
        
        self.hand_detection_mode = hand_detection
        self.object_detection_mode = object_detection
        self.device_name = device_name
        self.device = dv.get_device(device_name)
        self.hand_detector = hd.get_hand_detector(hand_detection, self.device)
        self.object_detector = o2d.get_object_detector(object_detection, self.device)
        self.object_pose_estimator = ope.get_pose_estimator(object_detection, self.device, use_tracking = True, fuse_detections=False)
        self.scene = sc.Scene(device=self.device,hand_detector=self.hand_detector, name = 'Full tracking')
        self.object_detections = None
        self.detect = True
        self.img = None
        self.do = False
        if self.device_name != 'OAK' and self.hand_detection_mode != 'mediapipe':
            print('depthai may only be used on OAK device')
            raise ValueError

    def objects_task(self):
        while self.device.isOn():
            if self.do:
                if self.img is None:
                    continue
                if self.detect :
                    self.object_detections = self.object_detector.detect(self.img)

                    if self.object_detections is not None:
                        self.detect=False
                else:
                    self.object_detections = None
                self.objects_pose = self.object_pose_estimator.estimate(self.img, detections = self.object_detections)
                self.scene.update_objects(self.objects_pose)

    def hands_task(self):
        while self.device.isOn() :  
            if self.do:
                if self.img is None:
                    continue
                hands = self.hand_detector.get_hands(self.img)
                if hands is not None and len(hands)>0:
                    self.scene.update_hands(hands)

    def run(self):
        print(self.__dict__)
        self.device.start()
        print('start')
        self.t_hand = threading.Thread(target=self.hands_task)
        self.t_obj = threading.Thread(target=self.objects_task)
        self.t_hand.start()
        self.t_obj.start()
        while self.device.isOn():
            success, img = self.device.next_frame()
            if not success:
                self.img = None
                continue     
            else:
                self.img = img
            self.img.flags.writeable = True
            k = cv2.waitKey(10)
            if k == 32 and not self.do:
                print('DOOOOOOOOOOOOOOOOOOOO')
                print('DOOOOOOOOOOOOOOOOOOOO')
                print('DOOOOOOOOOOOOOOOOOOOO')
                print('DOOOOOOOOOOOOOOOOOOOO')
                print('DOOOOOOOOOOOOOOOOOOOO')
                print('DOOOOOOOOOOOOOOOOOOOO')
                print('DOOOOOOOOOOOOOOOOOOOO')
                print('DOOOOOOOOOOOOOOOOOOOO')
                print('DOOOOOOOOOOOOOOOOOOOO')
                self.do = True
            if k == 32 and self.do:
                print('DETEEEEEEEEEEEEEEEEEECT')
                print('DETEEEEEEEEEEEEEEEEEECT')
                print('DETEEEEEEEEEEEEEEEEEECT')
                print('DETEEEEEEEEEEEEEEEEEECT')
                print('DETEEEEEEEEEEEEEEEEEECT')
                print('DETEEEEEEEEEEEEEEEEEECT')
                print('DETEEEEEEEEEEEEEEEEEECT')
                print('DETEEEEEEEEEEEEEEEEEECT')
                self.detect=True
            if self.scene.render(self.img):
                print('end')
                self.stop()
                break
        exit()

    def stop(self):
        self.t_hand.join()
        self.t_obj.join()
        self.device.stop()
        self.object_detector.stop()
        exit()


        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device_name', choices=['OAK', 'monocular_webcam', 'stereo_webcam'],
                        default='OAK', help="Video input device")
    parser.add_argument('-hd', '--hand_detection', choices=['mediapipe', 'depthai'],
                        default = 'depthai', help="Hand pose reconstruction solution")
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