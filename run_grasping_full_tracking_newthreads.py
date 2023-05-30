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
import numpy as np
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
        self.alpha = -0.2
        self.beta = -30
        self.alpha = 0.5
        self.beta = -81
        self.filter = False
        self.contrast = False
        if self.device_name != 'OAK' and self.hand_detection_mode != 'mediapipe':
            print('depthai may only be used on OAK device')
            raise ValueError
        self.no_blur_zone = np.array([0])
        self.is_hands= False

    def estimate_objects_task(self, start_event, estimate_event):
        while self.device.isOn():
            start_flag = start_event.wait(1)
            if start_flag:
                if estimate_event.wait(1):
                    self.objects_pose = self.object_pose_estimator.estimate(self.img, detections = self.object_detections)
                    self.scene.update_objects(self.objects_pose)
            print('ixiiiiiiiiiiiiiiiiiiiiiiiiiiii')

    def detect_objects_task(self, start_event, detect_event, estimate_event):
        while self.device.isOn():
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
            print('laaaaaaaaaaaaaaaaaaaaaaaaaaaaa')

    def hands_task(self, start_event, new_image_event):
        while self.device.isOn() :  
            start_flag = start_event.wait(1)
            if start_flag:
                img_flag = new_image_event.wait(1)
                if img_flag:
                    hands = self.hand_detector.get_hands(self.img)
                    if hands is not None and len(hands)>0:
                        self.scene.update_hands(hands)

    def run(self):
        print(self.__dict__)
        self.device.start()
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
        while self.device.isOn():
            success, img = self.device.next_frame()
            if self.contrast:
                img = cv2.convertScaleAbs(img,alpha=self.alpha, beta=self.beta)
            if self.filter:
                kernel3 = np.array([[0, -1,  0],
                    [-1,  5, -1],
                        [0, -1,  0]])
                img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel3)
            # if self.is_hands:
            #     blur = cv2.blur(img,(15,15),0)
            #     mask = self.no_blur_zone<1
            #     print(mask.shape)
            #     print(img.shape)
            #     img[mask] = blur[mask]
            # print('alpha : '+str(self.alpha))
            # print('beta : '+str(self.beta))
            # img = cv2.convertScaleAbs(img,alpha=0.2)
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
            if k == ord('a'):
                self.alpha+=0.2
            if k == ord('z'):
                self.alpha-=0.2
            if k== ord('b'):
                self.beta+=0.2
            if k== ord('n'):
                self.beta-=0.2
            if k==ord('f'):
                self.filter = not self.filter
            if k==ord('c'):
                self.contrast=not self.contrast
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