#!/usr/bin/env python3

import argparse
import multiprocessing as mp
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

def detect_loop(object_detector,  detect_img_queue, object_detections_queue, stop_event):
    while not stop_event.is_set():
        img = detect_img_queue.get()
        print('DETECT')
        if img is not None:
            object_detections_queue.put(object_detector.detect(img)) 

            
class GraspingDetector:
    def __init__(self, device_name='OAK', hand_detection='mediapipe',  object_detection='cosypose') -> None:
        
        self.hand_detection_mode = hand_detection
        self.object_detection_mode = object_detection
        self.device_name = device_name
        self.device = dv.get_device(device_name)
        self.object_detector = o2d.get_object_detector(object_detection, self.device)
        self.object_pose_estimator = ope.get_pose_estimator(object_detection, self.device, use_tracking=True, fuse_detections=False)
        self.scene = sc.Scene(device=self.device, name='Parallel, intermitent detections')
        self.new_detect = True
        self.new_estim = True
        self.detect_task_done = True
        self.estimate_task_done = True
        manager = mp.Manager()
        self.detect_img_queue = manager.Queue(maxsize=1)
        self.estim_img_queue = mp.Queue(maxsize=1)
        self.object_detections_queue = mp.Queue(maxsize=1)
        self.closables = [self.detect_img_queue, self.estim_img_queue, self.object_detections_queue]
        self.objects_pose_queue = mp.Queue(maxsize=1)
        self.estimations = None

        if self.device_name != 'OAK' and self.hand_detection_mode != 'mediapipe':
            print('depthai may only be used on OAK device')
            raise ValueError


    def estimate_loop(self):
        i = 0
        while not self.stop_event.is_set() :
            img = self.estim_img_queue.get()
            if self.object_detections_queue.empty():
                detection = None
            else:
                detection = self.object_detections_queue.get()
            if self.estimations is not None:
                detection = None
            if img is not None :
                self.estimations = self.object_pose_estimator.estimate(img, detections = detection)
                self.scene.update_objects(self.estimations)
            i +=1

    def run(self):
        print(self.__dict__)
        self.device.start()
        print('start')
        self.stop_event = mp.Event()
        detec = mp.Process(target=detect_loop, args=(self.object_detector,
                                                            self.detect_img_queue, self.object_detections_queue, self.stop_event))
        estim = threading.Thread(target=self.estimate_loop)
        detec.start()
        estim.start()
        while self.device.isOn():
            success, img = self.device.next_frame()
            if not success:
                continue     
            if not self.detect_img_queue.full():
                self.detect_img_queue.put(img)
                self.scene.update_detections_fps()
            if not self.estim_img_queue.full():
                self.estim_img_queue.put(img)
                
            img.flags.writeable = True
            k = cv2.waitKey(5)
            if k == 32:
                self.estimations=None
                print('REINIT')
            if self.scene.render(img):
                print('end')
                self.stop_event.set()
                detec.join()
                estim.join()
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
    
    mp.set_start_method("spawn")
    report_gpu()
    grasp_int = GraspingDetector(**args)
    grasp_int.run()