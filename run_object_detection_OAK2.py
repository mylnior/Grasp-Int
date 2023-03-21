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
        self.object_pose_estimator = ope.get_pose_estimator(object_detection, self.device)
        self.scene = sc.Scene(device=self.device)
        self.new_detect = True
        self.new_estim = True
        self.detect_task_done = True
        self.estimate_task_done = True
        self.object_detections = None

        if self.device_name != 'OAK' and self.hand_detection_mode != 'mediapipe':
            print('depthai may only be used on OAK device')
            raise ValueError

    def detect_task(self, img):
        self.object_detections = None
        self.object_detections = self.object_detector.detect(img)
        self.scene.update_detections_fps()
        self.detect_task_done = True
        #print('DETECT')
        #print(self.object_detections)

    def estimate_task(self,img):
        self.objects_pose = self.object_pose_estimator.estimate(img, detections = self.object_detections)
        self.estimate_task_done = True


    def run(self):
        print(self.__dict__)
        self.device.start()
        print('start')
        while self.device.isOn():
            success, img = self.device.next_frame()
            if not success:
                continue     

            if self.new_detect:
                self.new_detect = False
                self.t_detect = threading.Thread(target=self.detect_task, args=(img,))
                self.t_detect.start()

            if self.new_estim:
                self.new_estim = False
                self.t_estim = threading.Thread(target=self.estimate_task, args = (img,))     
                self.t_estim.start()  

            if self.detect_task_done:
                self.t_detect.join()
                self.new_detect=True
                self.detect_task_done = False

            if self.estimate_task_done:
                self.t_estim.join()
                self.scene.update_objects(self.objects_pose)
                self.new_estim=True
                self.estimate_task_done = False


            #self.objects_pose = self.object_pose_estimator.estimate(img, detections = self.object_detections)
            #if objects is not None:
            #    print(objects)
            #self.scene.update_objects(self.objects)
            #img.flags.writeable = False
            img.flags.writeable = True

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