#!/usr/bin/env python3
import abc
import cv2
import yaml
import depthai as dai
import numpy as np
import json
from pathlib import Path
from grasp_int.depthai_hand_tracker.HandTrackerEdge import HandTracker

_DEFAULT_CAM_SETTINGS_PATH = './default_cam_settings.yaml'
_DEFAULT_CALIBRATION_PARAMS_PATH = './default_calibration_parameters.yaml'
_DEFAULT_OKA_HANDTRACKER_ARGUMENT_PATH = './default_OAK_HandTracker_parameters.yaml'
_DEFAUL_CAM_INTRINSICS = './camera0_intrinsics.json'

class Device(abc.ABC):
    def __init__(self) -> None:
        self.name='jkl'

    @abc.abstractmethod
    def start(self):
        self.on = True
        
    @abc.abstractmethod
    def stop(self):
        self.on = False
        
    
    def isOn(self):
        return self.on
    
    @abc.abstractmethod
    def next_frame(self):
        pass

class MonocularWebcam(Device):
    def __init__(self, path = _DEFAULT_CAM_SETTINGS_PATH, cam_info_path=_DEFAUL_CAM_INTRINSICS) -> None:
        self.type = 'monocular_webcam'
        self.load_settings(path)
        with open(cam_info_path, 'r') as json_file:
            camera_data = json.load(json_file)
            self.matrix = np.array(camera_data['matrix'])

    def load_settings(self, settings_path):
        # load camera settings
        print('Using for camera settings: ', settings_path)
        with open(settings_path) as f:
            self.settings = yaml.safe_load(f)
            self.img_resolution = (self.settings['frame_width'],self.settings['frame_height'])

    def start(self):
        super().start()
        self.cap = cv2.VideoCapture(self.settings['camera0'],cv2.CAP_V4L2)
        self.cap.set(3, self.settings['frame_width'])
        self.cap.set(4, self.settings['frame_height'])

    def next_frame(self):
        success, image = self.cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            return success, None

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        return success, image
    
    def stop(self):
        super().stop()
        self.cap.release()

class StereoWebcam(Device):
    def __init__(self) -> None:
        self.type = 'stereo_webcam'

    def start(self):
        return super().start()

    def next_frame(self):
        return super().next_frame()

class OAK(Device, HandTracker):
    def __init__(self, path = _DEFAULT_OKA_HANDTRACKER_ARGUMENT_PATH) -> None:
        self.type = 'OAK'
        with open(path) as f:
            args = yaml.safe_load(f)
        HandTracker.__init__(self, **args)
        calibFile = str((Path(__file__).parent / Path(f"calib_{self.device.getMxId()}.json")).resolve().absolute())
        calibData = self.device.readCalibration()
        calibData.eepromToJsonFile(calibFile)
        M_rgb, cam_width, cam_height = calibData.getDefaultIntrinsics(dai.CameraBoardSocket.RGB)
        #self.extrinsics = calibData.getCameraExtrinsics(dai.CameraBoardSocket.RGB)
        print("RGB Camera Default intrinsics...")
        print(M_rgb)
        print(self.resolution)
        print(cam_width, cam_height)
        if False:
            M_rgb, cam_width, cam_height = calibData.getCameraIntrinsics(dai.CameraBoardSocket.RGB)
            print("RGB Camera Not Default intrinsics...")
            print(M_rgb)
            print(cam_width, cam_height)
        res_factor = self.img_w/cam_width
    
        self.matrix = np.array(M_rgb)*res_factor
        self.img_resolution = (self.img_w, self.img_h)
        print(self.img_resolution)

    def start(self):
        return super().start()

    def next_frame(self):
        in_video = self.q_video.get()
        video_frame = in_video.getCvFrame()
        success = video_frame is not None
        return success, video_frame
    
    def stop(self):
        super().stop()
        self.exit()

def get_device(type):
    if type == 'OAK':
        device = OAK()
    elif type == 'monocular_webcam':
        device = MonocularWebcam()
    else:
        device = StereoWebcam()
    return device