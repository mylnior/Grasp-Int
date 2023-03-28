#!/usr/bin/env python3

import argparse
import threading
from grasp_int import HandDetectors as hd
from grasp_int import ObjectDetectors as od
from grasp_int import Devices as dv
from grasp_int import Scene as sc
import matplotlib.pyplot as plt
import networkx
import pickle

class GraspingDetector:
    def __init__(self, device_name='OAK', hand_detection='mediapipe',  object_detection='cosypose') -> None:
        
        self.hand_detection_mode = hand_detection
        self.object_detection_mode = object_detection
        self.device_name = device_name
        self.device = dv.get_device(device_name)
        self.hand_detector = hd.get_hand_detector(hand_detection, self.device)
        self.scene = sc.Scene(device=self.device, hand_detector= self.hand_detector)
        self.object_task_done = True

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
            hands = self.hand_detector.get_hands(img)
            self.scene.update_hands(hands)
            #img.flags.writeable = False
            img.flags.writeable = True
            if self.scene.render(img):
                print('end')
                self.stop()
                break
        exit()

    def stop(self):
        self.device.stop()
        self.hand_detector.stop()
        self.scene.stop()
        G=self.scene.mesh_scene.copy().graph
        with open('my_graph', 'wb') as f:
            pickle.dump(G, f)

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
    grasp_int = GraspingDetector(**args)
    grasp_int.run()