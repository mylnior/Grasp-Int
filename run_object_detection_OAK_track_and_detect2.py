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

def detect_loop(object_detector, scene, detect_img_queue, object_detections_queue, stop_event):
    while not stop_event.is_set():
        img = detect_img_queue.get()
        if img is not None:
            object_detections_queue.put(object_detector.detect(img)) 
            scene.update_detections_fps()

def estimate_loop(object_pose_estimator,scene, estim_img_queue, object_detections_queue, stop_event):
    while not stop_event.is_set() :
        img = estim_img_queue.get()
        if object_detections_queue.empty():
            detection = None
        else:
            detection = object_detections_queue.get()
        if img is not None :
            scene.update_objects(object_pose_estimator.estimate(img, detections = detection))
            
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
        self.estim_img_queue = manager.Queue(maxsize=1)
        self.object_detections_queue = manager.Queue(maxsize=1)
        self.closables = [self.detect_img_queue, self.estim_img_queue, self.object_detections_queue]
        self.objects_pose_queue = mp.Queue(maxsize=1)

        if self.device_name != 'OAK' and self.hand_detection_mode != 'mediapipe':
            print('depthai may only be used on OAK device')
            raise ValueError



    def run(self):
        print(self.__dict__)
        self.device.start()
        print('start')
        stop_event = mp.Event()
        detec = mp.Process(target=detect_loop, args=(self.object_detector, self.scene,
                                                            self.detect_img_queue, self.object_detections_queue,stop_event))
        estim = threading.Thread(target=estimate_loop, args=(self.object_pose_estimator, self.scene,
                                                            self.estim_img_queue, self.object_detections_queue,stop_event))
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

            if self.scene.render(img):
                print('end')
                stop_event.set()
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