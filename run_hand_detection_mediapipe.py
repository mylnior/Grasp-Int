#!/usr/bin/env python3

import argparse
import threading
from grasp_int import HandDetectors as hd
from grasp_int import ObjectDetectors as od
from grasp_int import Devices as dv
from grasp_int import Scene as sc
import pickle
import cv2

class GraspingDetector:
    def __init__(self, hand_detection='hybridOAKMediapipe',  object_detection='cosypose') -> None:
        
        self.hand_detection_mode = hand_detection
        self.object_detection_mode = object_detection
        self.hand_detector = hd.get_hand_detector(hand_detection, None)
        self.scene = sc.Scene( hand_detector= self.hand_detector)
        self.object_task_done = True



    def run(self):
        print(self.__dict__)
        self.hand_detector.start()
        print('start')
        while self.hand_detector.isOn():
            success, img = self.hand_detector.next_frame()
            if not success:
                continue
            hands = self.hand_detector.get_hands()
            self.scene.update_hands(hands)
            #img.flags.writeable = False
            img.flags.writeable = True
            if self.scene.render(img):
                print('end')
                self.stop()
                break
        exit()

    def stop(self):
        self.hand_detector.stop()
        self.scene.stop()
        G=self.scene.mesh_scene.copy().graph
        with open('my_graph', 'wb') as f:
            pickle.dump(G, f)

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
    grasp_int = GraspingDetector(**args)
    grasp_int.run()