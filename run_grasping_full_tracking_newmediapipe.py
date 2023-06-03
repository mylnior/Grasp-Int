#!/usr/bin/env python3

import argparse
import threading
from grasp_int import HandDetectors as hd
from grasp_int import Object2DDetectors2 as o2d
from grasp_int import ObjectPoseEstimators2 as ope
from grasp_int import Devices as dv
from grasp_int import Scene as sc
import torch
import gc
import os
import cv2
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def report_gpu():
   print(torch.cuda.list_gpu_processes())
   gc.collect()
   print(torch.cuda.memory_snapshot())
   torch.cuda.empty_cache()

class GraspingDetector:
    def __init__(self, hand_detection='OAK',  object_detection='cosypose') -> None:
        
        self.hand_detection_mode = hand_detection
        self.object_detection_mode = object_detection
        self.hand_detector = hd.get_hand_detector(hand_detection, None)
        self.object_detector = o2d.get_object_detector(object_detection, self.hand_detector.resolution)
        self.object_pose_estimator = ope.get_pose_estimator(object_detection, self.hand_detector.matrix, self.hand_detector.resolution, use_tracking = True, fuse_detections=False)
        self.scene = sc.Scene(hand_detector=self.hand_detector, name = 'Full tracking')
        self.object_detections = None
        self.img = None
        self.alpha = -0.2
        self.beta = -30
        self.alpha = 0.5
        self.beta = -81
        self.filter = False
        self.contrast = False
        self.no_blur_zone = np.array([0])
        self.is_hands= False

    def estimate_objects_task(self, start_event, estimate_event):
        while self.hand_detector.isOn():
            start_flag = start_event.wait(1)
            if start_flag:
                if estimate_event.wait(1):
                    self.objects_pose = self.object_pose_estimator.estimate(self.img, detections = self.object_detections)
                    self.scene.update_objects(self.objects_pose)

    def detect_objects_task(self, start_event, detect_event, estimate_event):
        while self.hand_detector.isOn():
            start_flag = start_event.wait(1)
            if start_flag:
                detect_flag = detect_event.wait(1)
                if detect_flag:
                    self.object_detections = self.object_detector.detect(self.img)

                    if self.object_detections is not None:
                        detect_event.clear()
                        estimate_event.set()
                else:
                    self.object_detections = None

    def hands_task(self, start_event, new_image_event):
        while self.hand_detector.isOn() :  
            start_flag = start_event.wait(1)
            if start_flag:
                img_flag = new_image_event.wait(1)
                if img_flag:
                    hands = self.hand_detector.get_hands()
                    if hands is not None :
                    # if hands is not None and len(hands)>0:
                        self.scene.update_hands(hands)

    def run(self):
        print(self.__dict__)
        self.hand_detector.start()
        print('start')
        start_event = threading.Event()
        detect_event = threading.Event()
        estimate_event = threading.Event()
        new_image_event = threading.Event()
        self.t_hand = threading.Thread(target=self.hands_task, args=(start_event, new_image_event,))
        self.t_obj_d = threading.Thread(target=self.detect_objects_task, args=(start_event, detect_event,estimate_event,))
        self.t_obj_e = threading.Thread(target=self.estimate_objects_task, args=(start_event, estimate_event,))
        self.t_hand.start()
        self.t_obj_d.start()
        self.t_obj_e.start()
        while self.hand_detector.isOn():
            success, img = self.hand_detector.next_frame()
            if not success:
                self.img = None
                continue     
            else:
                new_image_event.set()
                new_image_event.clear()
                estimate_event.set()
                estimate_event.clear()
                self.img = img
            self.img.flags.writeable = True
            k = cv2.waitKey(10)
            if k == 32:
                print('DOOOOOOOOOOOOOOOOOOOO')
                print('DOOOOOOOOOOOOOOOOOOOO')
                print('DOOOOOOOOOOOOOOOOOOOO')
                print('DOOOOOOOOOOOOOOOOOOOO')
                print('DOOOOOOOOOOOOOOOOOOOO')
                print('DOOOOOOOOOOOOOOOOOOOO')
                print('DOOOOOOOOOOOOOOOOOOOO')
                print('DOOOOOOOOOOOOOOOOOOOO')
                print('DOOOOOOOOOOOOOOOOOOOO')
                start_event.set()
            if k == 32:
                print('DETEEEEEEEEEEEEEEEEEECT')
                print('DETEEEEEEEEEEEEEEEEEECT')
                print('DETEEEEEEEEEEEEEEEEEECT')
                print('DETEEEEEEEEEEEEEEEEEECT')
                print('DETEEEEEEEEEEEEEEEEEECT')
                print('DETEEEEEEEEEEEEEEEEEECT')
                print('DETEEEEEEEEEEEEEEEEEECT')
                print('DETEEEEEEEEEEEEEEEEEECT')
                detect_event.set()
                estimate_event.set()
            if self.scene.render(self.img):
                print('end')
                self.stop()
                break
        exit()

    def stop(self):
        self.t_hand.join()
        self.t_obj.join()
        self.object_detector.stop()
        exit()


        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-hd', '--hand_detection', choices=['mediapipe', 'depthai', 'hybridOAKMediapipe'],
                        default = 'hybridOAKMediapipe', help="Hand pose reconstruction solution")
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