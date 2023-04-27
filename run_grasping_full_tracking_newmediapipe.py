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
        self.detect = True
        self.img = None
        self.do = False
        self.alpha = -0.2
        self.beta = -30
        self.alpha = 0.5
        self.beta = -81
        self.filter = False
        self.contrast = False
        self.no_blur_zone = np.array([0])
        self.is_hands= False

    def objects_task(self):
        while self.hand_detector.isOn():
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
        while self.hand_detector.isOn() :  
            if self.do:
                if self.img is None:
                    continue
                hands = self.hand_detector.get_hands()
                if hands is not None and len(hands)>0:
                    self.scene.update_hands(hands)
                    # for hand in hands:
                    #     # self.no_blur_zone[hand]
                    #     self.is_hands=True
                    #     print('rect:')
                    #     rec = hand.rect_points
                    #     print(rec)
                    #     max_x = np.max([p[0] for p in rec])+100
                    #     min_x = np.min([p[0] for p in rec])-100
                    #     max_y = np.max([p[1] for p in rec])+100
                    #     min_y = np.min([p[1] for p in rec])-100
                    #     self.no_blur_zone=np.zeros(self.device.img_resolution)
                    #     self.no_blur_zone[min_x:max_x, min_y:max_y] = 1
                    #     self.no_blur_zone = self.no_blur_zone.T

    def run(self):
        print(self.__dict__)
        self.hand_detector.start()
        print('start')
        self.t_hand = threading.Thread(target=self.hands_task)
        self.t_obj = threading.Thread(target=self.objects_task)
        self.t_hand.start()
        self.t_obj.start()
        while self.hand_detector.isOn():
            success, img = self.hand_detector.next_frame()
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