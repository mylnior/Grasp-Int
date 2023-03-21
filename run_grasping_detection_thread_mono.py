#!/usr/bin/env python3

import argparse
import threading
from grasp_int import HandDetectors as hd
from grasp_int import ObjectDetectors as od
from grasp_int import Devices as dv
from grasp_int import Scene as sc


class GraspingDetector:
    def __init__(self, device_name='OAK', hand_detection='mediapipe',  object_detection='cosypose') -> None:
        
        self.hand_detection_mode = hand_detection
        self.object_detection_mode = object_detection
        self.device_name = device_name
        self.device = dv.get_device(device_name)
        self.hand_detector = hd.get_hand_detector(hand_detection, self.device)
        self.object_detector = od.get_object_detector(object_detection, self.device)
        self.scene = sc.Scene(self.hand_detector, self.object_detector)
        self.object_task_done = True

        if self.device_name != 'OAK' and self.hand_detection_mode != 'mediapipe':
            print('depthai may only be used on OAK device')
            raise ValueError

        # if self.hand_detection == 'mediapipe':
        #     if
        #     self.hand_detection = hd
    def hands_task(self, img):
        hands = self.hand_detector.get_hands(img)
        self.scene.update_hands(hands)

    def objects_task(self,img):
        self.object_task_done = False
        self.objects = self.object_detector.get_objects(img)
        self.object_task_done = True

    def run(self):
        print(self.__dict__)
        self.device.start()
        print('start')
        while self.device.isOn():
            success, img = self.device.next_frame()
            if not success:
                continue
            t_hand = threading.Thread(target=self.hands_task, args=(img,))
            t_hand.start()
            print(self.object_task_done)
            if self.object_task_done:
                self.t_obj = threading.Thread(target=self.objects_task, args=(img,))
                self.t_obj.start()
            t_hand.join()
            print(self.object_task_done)
            if self.object_task_done:
                self.t_obj.join()
                self.scene.update_objects(self.objects)
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
        self.object_detector.stop()
        exit()


        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device_name', choices=['OAK', 'monocular_webcam', 'stereo_webcam'],
                        default='monocular_webcam', help="Video input device")
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
    grasp_int = GraspingDetector(**args)
    grasp_int.run()