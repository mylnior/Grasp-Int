#!/usr/bin/env python3
import abc
import yaml
import mediapipe as mp
import cv2
import numpy as np
import time
import marshal

from collections import namedtuple
import grasp_int.depthai_hand_tracker.mediapipe_utils as mpu
import depthai as dai
from grasp_int.depthai_hand_tracker.FPS import now
from grasp_int.depthai_hand_tracker.HandTracker import to_planar
from grasp_int.depthai_hand_tracker.HandTrackerRenderer import HandTrackerRenderer

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

_DEFAULT_HAND_PARAMS_PATH = './default_hand_detection_parameters.yaml'

class Hands:
    def __init__(self) -> None:
        self.results = None
        self.target = None
        self.reachable_objects = list()
        self.position = np.array([0,0,0])
        self.velocity = np.array([0,0,0])
        self.last_time=time.time()
        self.target_threshold = 0.1
        self.distance_threshold = 0.6
        self.cone_angle = np.pi/8

class HandDetector(abc.ABC):
    def __init__(self) -> None:
        pass
    
    @abc.abstractmethod
    def get_hands(self, img):
        return None
        
    def stop(self):
        exit()
        pass

class MediaPipeMonocularHandDetector(mp_hands.Hands, HandDetector):
    def __init__(self, device, path = _DEFAULT_HAND_PARAMS_PATH) -> None:
        self.type = 'monocular_hand_detector'
        self.load_parameters(path)
        mp_hands.Hands.__init__(self,
               static_image_mode=self.parameters['static_image_mode'],
               max_num_hands=self.parameters['max_num_hands'],
               model_complexity=self.parameters['model_complexity'],
               min_detection_confidence=self.parameters['min_detection_confidence'],
               min_tracking_confidence=self.parameters['min_tracking_confidence'])
        HandDetector.__init__(self)

    def load_parameters(self, parameters_path):
        # load camera parameters
        print('Using for hand detector parameters: ', parameters_path)
        with open(parameters_path) as f:
            self.parameters = yaml.safe_load(f)
    
    def get_hands(self, img):

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.process(img)
        print(self.results)
        return self.results

    def draw_landmarks(self, img):
        if self.results:
            if self.results.multi_hand_landmarks:
                for hand_landmarks in self.results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        img,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
    
    def write_pos(self, img):
        pass

class MediaPipeStereoHandDetector(HandDetector):
    def __init__(self, device) -> None:
        self.device = device
        pass

class OAKHandDetector(HandDetector):
    def __init__(self, device) -> None:
        self.device = device
        self.renderer = HandTrackerRenderer(device)
        
    def get_hands(self, img):

        # Get result from device
        res = marshal.loads(self.device.q_manager_out.get().getData())
        hands = []
        for i in range(len(res.get("lm_score",[]))):
            hand = self.device.extract_hand_data(res, i)
            hands.append(hand)
        self.results = hands
        return hands

    def get_hands_host(self, img):
        self.bag = {}
        if not self.device.use_previous_landmarks:
            # Send image manip config to the device
            cfg = dai.ImageManipConfig()
            # We prepare the input to the Palm detector
            cfg.setResizeThumbnail(self.device.pd_input_length, self.device.pd_input_length)
            self.device.q_manip_cfg.send(cfg)
        if self.device.pad_h:
            square_frame = cv2.copyMakeBorder(img, self.device.pad_h, self.device.pad_h, self.device.pad_w, self.device.pad_w, cv2.BORDER_CONSTANT)
        else:
            square_frame = img

        # Get palm detection
        if self.device.use_previous_landmarks:
            self.device.hands = self.device.hands_from_landmarks
        else:
            inference = self.device.q_pd_out.get()
            hands = self.device.pd_postprocess(inference)
            if self.device.trace & 1:
                print(f"Palm detection - nb hands detected: {len(hands)}")
            self.device.nb_frames_pd_inference += 1  
            self.bag["pd_inference"] = 1 
            if not self.device.solo and self.device.nb_hands_in_previous_frame == 1 and len(hands) <= 1:
                self.device.hands = self.device.hands_from_landmarks
            else:
                self.device.hands = hands

        if len(self.device.hands) == 0: self.device.nb_frames_no_hand += 1
        
        if self.device.use_lm:
            nb_lm_inferences = len(self.device.hands)
            # Hand landmarks, send requests
            for i,h in enumerate(self.device.hands):
                img_hand = mpu.warp_rect_img(h.rect_points, square_frame, self.device.lm_input_length, self.device.lm_input_length)
                nn_data = dai.NNData()   
                nn_data.setLayer("input_1", to_planar(img_hand, (self.device.lm_input_length, self.device.lm_input_length)))
                self.device.q_lm_in.send(nn_data)
                if i == 0: lm_rtrip_time = now() # We measure only for the first hand
            # Get inference results
            for i,h in enumerate(self.device.hands):
                inference = self.device.q_lm_out.get()
                if i == 0: self.device.glob_lm_rtrip_time += now() - lm_rtrip_time
                self.device.lm_postprocess(h, inference)
            self.bag["lm_inference"] = len(self.device.hands)
            self.device.hands = [ h for h in self.device.hands if h.lm_score > self.device.lm_score_thresh]

            if self.device.trace & 1:
                print(f"Landmarks - nb hands detected : {len(self.device.hands)}")

            # Check that 2 detected hands do not correspond to the same hand in the image
            # That may happen when one hand in the image cross another one
            # A simple method is to assure that the center of the rotated rectangles are not too close
            if len(self.device.hands) == 2: 
                dist_rect_centers = mpu.distance(np.array((self.device.hands[0].rect_x_center_a, self.device.hands[0].rect_y_center_a)), np.array((self.device.hands[1].rect_x_center_a, self.device.hands[1].rect_y_center_a)))
                if dist_rect_centers < 5:
                    # Keep the hand with higher landmark score
                    if self.device.hands[0].lm_score > self.device.hands[1].lm_score:
                        self.device.hands = [self.device.hands[0]]
                    else:
                        self.device.hands = [self.device.hands[1]]
                    if self.device.trace & 1: print("!!! Removing one hand because too close to the other one")

            if self.device.xyz:
                self.device.query_xyz(self.device.spatial_loc_roi_from_wrist_landmark)

            self.device.hands_from_landmarks = [mpu.hand_landmarks_to_rect(h) for h in self.device.hands]
            
            nb_hands = len(self.device.hands)

            if self.device.use_handedness_average:
                if not self.device.use_previous_landmarks or self.device.nb_hands_in_previous_frame != nb_hands:
                    for i in range(self.device.max_hands):
                        self.device.handedness_avg[i].reset()
                for i in range(nb_hands):
                    self.device.hands[i].handedness = self.device.handedness_avg[i].update(self.device.hands[i].handedness)

            # In duo mode , make sure only one left hand and one right hand max is returned everytime
            if not self.device.solo and nb_hands == 2 and (self.device.hands[0].handedness - 0.5) * (self.device.hands[1].handedness - 0.5) > 0:
                self.device.hands = [self.device.hands[0]] # We keep the hand with best score
                nb_hands = 1
                if self.device.trace & 1: print("!!! Removing one hand because same handedness")

            if not self.device.solo:
                if nb_hands == 1:
                    self.device.single_hand_count += 1
                else:
                    self.device.single_hand_count = 0

            # Stats
            if nb_lm_inferences: self.device.nb_frames_lm_inference += 1
            self.device.nb_lm_inferences += nb_lm_inferences
            self.device.nb_failed_lm_inferences += nb_lm_inferences - nb_hands 
            if self.device.use_previous_landmarks: self.device.nb_frames_lm_inference_after_landmarks_ROI += 1

            self.device.use_previous_landmarks = True
            if nb_hands == 0:
                self.device.use_previous_landmarks = False
            elif not self.device.solo and nb_hands == 1:
                    if self.device.single_hand_count >= self.device.single_hand_tolerance_thresh:
                        self.device.use_previous_landmarks = False
                        self.device.single_hand_count = 0
            
            self.device.nb_hands_in_previous_frame = nb_hands           
            
            for hand in self.device.hands:
                # If we added padding to make the image square, we need to remove this padding from landmark coordinates and from rect_points
                if self.device.pad_h > 0:
                    hand.landmarks[:,1] -= self.device.pad_h
                    for i in range(len(hand.rect_points)):
                        hand.rect_points[i][1] -= self.device.pad_h
                if self.device.pad_w > 0:
                    hand.landmarks[:,0] -= self.device.pad_w
                    for i in range(len(hand.rect_points)):
                        hand.rect_points[i][0] -= self.device.pad_w

                # Set the hand label
                hand.label = "right" if hand.handedness > 0.5 else "left"       

        else: # not use_lm
            if self.device.xyz:
                self.device.query_xyz(self.device.spatial_loc_roi_from_palm_center)
        self.results = self.device.hands
        return self.results
    
    def draw_landmarks(self, img):
        self.renderer.draw(img, self.results)
    
    def stop(self):
        self.renderer.exit()
        self.device.exit()

def get_hand_detector(type, device):

    if type == 'mediapipe':
        hand_detector = MediaPipeMonocularHandDetector(device)
    elif type == 'monocular_webcam':
        hand_detector = MediaPipeStereoHandDetector(device)
    else:
        hand_detector = OAKHandDetector(device)
    return hand_detector