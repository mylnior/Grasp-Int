#!/usr/bin/env python3

import argparse
import threading
import torch.multiprocessing as mp
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

def detect_task(object_detector, device_on, detect, detect_task_done, scene, img, object_detections, new_detection):
    while device_on:
        if detect:
            print('DETECTION')
            detect = False
            detect_task_done=False
            object_detections = object_detector.detect(img)
            scene.update_detections_fps()
            new_detection = True
            detect_task_done = True
            print('detect done :'+str(detect_task_done)+' : '+str(n_detect))
            n_detect+=1
class GraspingDetector:
    def __init__(self, device_name='OAK', hand_detection='mediapipe',  object_detection='cosypose') -> None:
        
        self.hand_detection_mode = hand_detection
        self.object_detection_mode = object_detection
        self.device_name = device_name
        self.device = dv.get_device(device_name)
        self.object_detector = o2d.get_object_detector(object_detection, self.device)
        self.object_pose_estimator = ope.get_pose_estimator(object_detection, self.device, use_tracking = True, fuse_detections=False)
        self.scene = sc.Scene(device=self.device, name = 'Full tracking')
        self.object_detections = None
        self.detect = False
        self.estimate = False
        self.detect_task_done=True
        self.estim_task_done = True
        self.new_detection = False
        self.n_detect = 0
        self.img=None
        self.device_on = False
        if self.device_name != 'OAK' and self.hand_detection_mode != 'mediapipe':
            print('depthai may only be used on OAK device')
            raise ValueError


    def estimate_task(self):
        while self.device.isOn():
            if self.estimate:
                self.estimate = False
                self.estim_task_done = False
                if  self.new_detection:
                    #used_detections = self.object_detections
                    used_detections = None
                    self.new_detection=False
                else:
                    used_detections = None
                self.objects_pose = self.object_pose_estimator.estimate(self.img, detections = used_detections)
                self.scene.update_objects(self.objects_pose)
                self.estim_task_done = True

    def try_new_detect(self):
        self.detect =  self.detect_task_done
        #print('try new detect')
        #print(self.detect)
        if self.n_detect>=5:
            #exit()
            pass

    def try_new_estimate(self):
        self.estimate = self.estim_task_done

    def set_new_img(self, img):
        self.img = img
        self.img.flags.writeable = True
        self.try_new_detect()
        #self.try_new_estimate()


    def run(self):
        print(self.__dict__)
        self.device.start()
        print('start')
        self.t_detect = mp.Process(target=detect_task, args=(self.object_detector, self.device_on, self.detect, self.detect_task_done, self.scene, self.img, self.object_detections, self.new_detection))
        self.t_estim = threading.Thread(target=self.estimate_task)     
        #self.t_render = threading.Thread(target=self.render_task)     
        self.t_detect.start()
        #self.t_estim.start()  
        #self.t_render.start()  
        while self.device.isOn():
            self.device_on = self.device.isOn()
            success, img = self.device.next_frame()
            if not success:
                continue     
            else:
                self.set_new_img(img)

                if self.scene.render(img):
                    print('end')
                    self.stop()
                    break
        exit()

    def stop(self):
        self.device.stop()
        self.object_detector.stop()
        self.t_detect.join()
        self.t_estim.join()
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
    
    mp.set_start_method("spawn")
    report_gpu()
    grasp_int = GraspingDetector(**args)
    grasp_int.run()