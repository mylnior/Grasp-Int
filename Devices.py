#!/usr/bin/env python3
import abc
import cv2
import yaml

_DEFAULT_CAM_SETTINGS_PATH = './default_cam_settings.yaml'
_DEFAULT_CALIBRATION_PARAMS_PATH = './default_calibration_parameters.yaml'

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
    def __init__(self, path = _DEFAULT_CAM_SETTINGS_PATH) -> None:
        self.type = 'monocular_webcam'
        self.load_settings(path)

    def load_settings(self, settings_path):
        # load camera settings
        print('Using for camera settings: ', settings_path)
        with open(settings_path) as f:
            self.settings = yaml.safe_load(f)
            self.frame_shape = (self.settings['frame_width'],self.settings['frame_height'])

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

class OAK(Device):
    def __init__(self) -> None:
        self.type = 'OAK'

    def start(self):
        return super().start()

    def next_frame(self):
        return super().next_frame()
    

def get_device(type):
    if type == 'OAK':
        device = OAK()
    elif type == 'monocular_webcam':
        device = MonocularWebcam()
    else:
        device = StereoWebcam()
    return device