#!/usr/bin/env python3
import abc
import yaml
import mediapipe as mp
import cv2
import numpy as np
import time
import marshal
import math
import re

from collections import namedtuple
import grasp_int.depthai_hand_tracker.mediapipe_utils as mpu
import depthai as dai
from grasp_int.depthai_hand_tracker.FPS import now
from grasp_int.depthai_hand_tracker.HandTracker import to_planar
from grasp_int.depthai_hand_tracker.HandTrackerRenderer import HandTrackerRenderer

from grasp_int import Devices as dv
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

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
        self.type = 'OAKHandDetector'
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


class HybridOAKMediapipeDetector(dv.Device):
    def __init__(self) -> None:
        self.device = dai.Device()
        self.type = 'HybridOAKMediapipeDetector'
        self.fps = 40
        self.lensPos = 150
        self.expTime = 8000
        self.sensIso = 500    
        self.wbManual = 4000
        self.rgb_res = dai.ColorCameraProperties.SensorResolution.THE_1080_P
        self.mono_res = dai.MonoCameraProperties.SensorResolution.THE_400_P
        self.left_right_initial_res = 720
        self.resolutions = [    np.array([1920, 1080]),    np.array([1280, 720]),    np.array([854, 480]),    np.array([640, 360]),    np.array([426, 240])]
        self.res_idx = 1
        self.device.startPipeline(self.create_pipeline())

        self.crop = True
        self.resolution = self.resolutions[self.res_idx]
        if self.crop:
            self.croped_resolution = (self.resolution[1], self.resolution[1])
        else:
            self.croped_resolution = self.resolution
        self.rgbQ = self.device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
        self.depthQ = self.device.getOutputQueue(name="depth", maxSize=1, blocking=False)

        # THE_720_P => 720
        print()
        resolution_num = int(re.findall("\d+", str(self.mono_res))[0])
        self.stereoInference = StereoInference(self.device, (400,400))

        self.detection_result=None
        self.hands=[]
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='/home/emoullet/Mediapipe2/hand_landmarker.task'),
            running_mode=VisionRunningMode.LIVE_STREAM,
            num_hands=2,
            min_hand_detection_confidence = 0.6,
            min_hand_presence_confidence = 0.6,
            result_callback=self.extract_hands)
        self.landmarker = HandLandmarker.create_from_options(options)
        self.format=mp.ImageFormat.SRGB

        self.margin = 10  # pixels
        self.font_size = 1
        self.font_thickness = 1
        self.handedness_text_color = (88, 205, 54) # vibrant green

    def get_res(self):
        return self.resolution
    
    def start(self):
        return super().start()
    
    def stop(self):
        super().stop()
        exit()
    
    def create_pipeline(self):
        print("Creating pipeline...")
        # Start defining a pipeline
        pipeline = dai.Pipeline()
        # ColorCamera
        print("Creating Color Camera...")
        camRgb = pipeline.createColorCamera()
        camRgb.setResolution(self.rgb_res)
        controlIn = pipeline.create(dai.node.XLinkIn)
        controlIn.setStreamName('control')
        controlIn.out.link(camRgb.inputControl)

        camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        camRgb.setInterleaved(False)
        camRgb.setIspScale(2, 3)
        print("Setting manual exposure, time: ", self.expTime, "iso: ", self.sensIso)
        camRgb.initialControl.setManualExposure(self.expTime, self.sensIso)
        # cam.initialControl.setAutoExposureEnable()
        camRgb.initialControl.setManualWhiteBalance(self.wbManual)
        # cam.initialControl.setManualFocus(self.lensPos)
        # cam.setIspScale(self.scale_nd[0], self.scale_nd[1])
        camRgb.setFps(self.fps)

        calibData = self.device.readCalibration()
        self.matrix, cam_width, cam_height = calibData.getDefaultIntrinsics(dai.CameraBoardSocket.RGB)
        print(self.matrix)
        self.matrix = np.array(self.matrix)
        
        camLeft = pipeline.create(dai.node.MonoCamera)
        camRight = pipeline.create(dai.node.MonoCamera)
        camLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        camRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        # cam.setVideoSize(self.resolution)
        for monoCam in (camLeft, camRight):  # Common config
            monoCam.setResolution(self.mono_res)
            monoCam.setFps(self.fps)


        self.cam_out = pipeline.createXLinkOut()
        self.cam_out.setStreamName("rgb")
        self.cam_out.input.setQueueSize(1)
        self.cam_out.input.setBlocking(False)
        camRgb.isp.link(self.cam_out.input)

        # Create StereoDepth node that will produce the depth map
        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.initialConfig.setConfidenceThreshold(245)
        stereo.initialConfig.setMedianFilter(dai.StereoDepthProperties.MedianFilter.KERNEL_7x7)
        stereo.setLeftRightCheck(True)
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        camLeft.out.link(stereo.left)
        camRight.out.link(stereo.right)

        # Closer-in minimum depth, disparity range is doubled (from 95 to 190):
        extended_disparity = True
        # Better accuracy for longer distance, fractional disparity 32-levels:
        subpixel = False
        # Better handling for occlusions:
        lr_check = True

        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        # Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
        stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
        stereo.setLeftRightCheck(lr_check)
        stereo.setExtendedDisparity(extended_disparity)
        stereo.setSubpixel(subpixel)

        self.depth_out = pipeline.create(dai.node.XLinkOut)
        self.depth_out.setStreamName("depth")
        stereo.depth.link(self.depth_out.input)
        print("Pipeline created.")
        return pipeline


    def extract_hands(self, result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        # print('hand landmarker result: {}'.format(result))
        self.detection_result = result
        if self.detection_result is not None:
            hands = []
            hand_landmarks_list = self.detection_result.hand_landmarks
            hand_world_landmarks_list = self.detection_result.hand_world_landmarks
            handedness_list = self.detection_result.handedness

            # Loop through the detected hands to visualize.
            for idx in range(len(hand_landmarks_list)):
                hand_landmarks = hand_landmarks_list[idx]
                hand_world_landmarks = hand_world_landmarks_list[idx]
                handedness = handedness_list[idx]

                # Draw the hand landmarks.
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
                ])
                hand = Hand(handedness, hand_landmarks, hand_world_landmarks, hand_landmarks_proto, self.croped_resolution)
                xyz, roi = self.stereoInference.calc_spatials(hand.landmarks,self.depth_map)
                hand.set_xyz(xyz)
                hand.roi = roi
                hands.append(hand)
            self.hands = hands

    def draw_landmarks_on_image(self, rgb_image): #TODO : MODIFY BY HAND
        if len(self.hands)>0:
            annotated_image = np.copy(rgb_image)
            for hand in self.hands:
                solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand.landmarks_proto,
                solutions.hands.HAND_CONNECTIONS,
                solutions.drawing_styles.get_default_hand_landmarks_style(),
                solutions.drawing_styles.get_default_hand_connections_style())

                # Get the top left corner of the detected hand's bounding box.
                if len(annotated_image.shape)<3:
                    height, width = annotated_image.shape
                else:
                    height, width, _ = annotated_image.shape
                x_coordinates = [landmark.x for landmark in hand.landmarks]
                y_coordinates = [landmark.y for landmark in hand.landmarks]
                text_x = int(min(x_coordinates) * width)
                text_y = int(min(y_coordinates) * height) - self.margin

                # Draw handedness (left or right hand) on the image.
                cv2.putText(annotated_image, f"{hand.handedness[0].category_name}",
                            (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                            self.font_size, self.handedness_text_color, self.font_thickness, cv2.LINE_AA)
            return annotated_image
        else:
            return rgb_image
        
    def next_frame(self):
        d_frame = self.depthQ.get()
        r_frame = self.rgbQ.get()
        if d_frame is not None:
            frame = d_frame.getFrame()
            frame = cv2.resize(frame, self.resolution)
            frame = crop_to_rect(frame)
            self.depth_map=cv2.flip(frame,1)

            depthFrameColor = cv2.normalize(self.depth_map, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
            depthFrameColor = cv2.equalizeHist(depthFrameColor)
            depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_JET)
            cv2.imshow('depth', depthFrameColor)

        if r_frame is not None:
            frame = r_frame.getCvFrame()
            # cv2.imshow('raw_'+name, frame)
            frame = cv2.resize(frame, self.resolution)
            frame = crop_to_rect(frame)
            frame=cv2.flip(frame,1)
            mp_frame = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
            if frame is not None:
            # if frame is not None and depthFrame is not None:
                frame_timestamp_ms = round(time.time()*1000)
                mp_image = mp.Image(image_format=self.format, data=mp_frame)
                self.landmarker.detect_async(mp_image, frame_timestamp_ms)
                success = True
        return success, frame
    
    def get_hands(self):
        return self.hands


class Hand:
    def __init__(self, handedness, landmarks, world_landmarks,landmarks_proto, img_res) -> None:
        self.handedness = handedness
        self.landmarks = np.array([[l.x*img_res[0], l.y*img_res[1], l.z] for l in landmarks])
        self.world_landmarks = np.array([[l.x*img_res[0], l.y*img_res[1], l.z] for l in world_landmarks])
        self.label = handedness[0].category_name.lower()
        self.landmarks_proto = landmarks_proto
        self.xyz = np.array([0,0,0])
        self.mp_landmarks = landmarks
        self.mp_world_landmarks =  world_landmarks
        self.roi = None
        self.show_handedness = True
        self.show_xyz = True
        self.show_roi = True
        # print(landmarks)
        self.margin = 10  # pixels
        self.font_size = 1
        self.font_thickness = 1
        self.handedness_text_color = (88, 205, 54) # vibrant green
        self.font_size_xyz = 0.5

    def set_xyz(self, xyz):
        self.xyz = xyz
    
    def draw(self, img):
        landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in self.mp_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
        img,
        self.landmarks_proto,
        solutions.hands.HAND_CONNECTIONS,
        solutions.drawing_styles.get_default_hand_landmarks_style(),
        solutions.drawing_styles.get_default_hand_connections_style())

        if self.show_handedness:
            # Get the top left corner of the detected hand's bounding box.
            if len(img.shape)<3:
                height, width = img.shape
            else:
                height, width, _ = img.shape
            x_coordinates = [landmark.x for landmark in self.mp_landmarks]
            y_coordinates = [landmark.y for landmark in self.mp_landmarks]
            text_x = int(min(x_coordinates) * width)
            text_y = int(min(y_coordinates) * height) - self.margin

            # Draw handedness (left or right hand) on the image.
            cv2.putText(img, f"{self.handedness[0].category_name}",
                        (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                        self.font_size, self.handedness_text_color, self.font_thickness, cv2.LINE_AA)
        if self.show_roi:
            cv2.rectangle(img, (self.roi[0],self.roi[1]),(self.roi[2],self.roi[3]),self.handedness_text_color)

        if self.show_xyz:
            # Get the top left corner of the detected hand's bounding box.z
            
            #print(f"{self.label} --- X: {self.xyz[0]/10:3.0f}cm, Y: {self.xyz[0]/10:3.0f} cm, Z: {self.xyz[0]/10:3.0f} cm")
            if len(img.shape)<3:
                height, width = img.shape
            else:
                height, width, _ = img.shape
            x_coordinates = [landmark.x for landmark in self.mp_landmarks]
            y_coordinates = [landmark.y for landmark in self.mp_landmarks]
            x0 = int(max(x_coordinates) * width)
            y0 = int(max(y_coordinates) * height) + self.margin

            # Draw handedness (left or right hand) on the image.
            cv2.putText(img, f"X:{self.xyz[0]/10:3.0f} cm", (x0+10, y0+20), cv2.FONT_HERSHEY_DUPLEX, self.font_size_xyz, (20,180,0), self.font_thickness, cv2.LINE_AA)
            cv2.putText(img, f"Y:{self.xyz[1]/10:3.0f} cm", (x0+10, y0+45), cv2.FONT_HERSHEY_DUPLEX, self.font_size_xyz, (255,0,0), self.font_thickness, cv2.LINE_AA)
            cv2.putText(img, f"Z:{self.xyz[2]/10:3.0f} cm", (x0+10, y0+70), cv2.FONT_HERSHEY_DUPLEX, self.font_size_xyz, (0,0,255), self.font_thickness, cv2.LINE_AA)


class StereoInference:
    def __init__(self, device: dai.Device, resolution, resize=False, width=300, heigth=300) -> None:
        calibData = device.readCalibration()
        baseline = calibData.getBaselineDistance(useSpecTranslation=True) * 10  # mm

        # Original mono frames shape
        self.original_heigth = resolution[1]
        self.original_width = resolution[0]
        self.original_width = 1080
        self.hfov = calibData.getFov(dai.CameraBoardSocket.RIGHT)
        self.hfov = np.deg2rad(73.5)

        focalLength = self.get_focal_length_pixels(self.original_width, self.hfov)
        self.dispScaleFactor = baseline * focalLength

        # Cropped frame shape
        if resize :
            self.mono_width = width
            self.mono_heigth = heigth
        else:
            self.mono_width = self.original_width
            self.mono_heigth = self.original_heigth

        # Our coords are normalized for 300x300 image. 300x300 was downscaled from
        # 720x720 (by ImageManip), so we need to multiple coords by 2.4 to get the correct disparity.
        self.resize_factor = self.original_heigth / self.mono_heigth
        self.depth_thres_high = 3000
        self.depth_thres_low = 50

    def get_focal_length_pixels(self, pixel_width, hfov):
        return pixel_width * 0.5 / math.tan(hfov * 0.5 * math.pi/180)

    def calculate_depth(self, disparity_pixels: float):
        try:
            return self.dispScaleFactor / disparity_pixels
        except ZeroDivisionError:
            return 0 # Or inf?

    def calculate_distance(self, c1, c2):
        # Our coords are normalized for 300x300 image. 300x300 was downscaled from 720x720 (by ImageManip),
        # so we need to multiple coords by 2.4 (if using 720P resolution) to get the correct disparity.
        c1 = np.array(c1) * self.resize_factor
        c2 = np.array(c2) * self.resize_factor

        x_delta = c1[0] - c2[0]
        y_delta = c1[1] - c2[1]
        return math.sqrt(x_delta ** 2 + y_delta ** 2)

    def calc_angle(self, offset):
            return math.atan(math.tan(self.hfov / 2.0) * offset / (self.original_width / 2.0))

    def calc_spatials(self, np_landmarks, depth_map, averaging_method=np.mean):
        if depth_map is None:
            print('No depth map available yet')
            return np.array([0,0,0])
        wrist = np_landmarks[0]
        thumb = np_landmarks[1]
        box_size = 10
        #box_size = max(5, int(np.linalg.norm(wrist-thumb)))/2
        xmin = max(int(wrist[0]-box_size),0)
        xmax = min(int(wrist[0]+box_size), int(depth_map.shape[0]))
        ymin = max(int(wrist[1]-box_size),0 )
        ymax = min(int(wrist[1]+box_size), int(depth_map.shape[0]))
        if xmin > xmax:  # bbox flipped
            xmin, xmax = xmax, xmin
        if ymin > ymax:  # bbox flipped
            ymin, ymax = ymax, ymin

        if xmin == xmax or ymin == ymax: # Box of size zero
            return None

        # Calculate the average depth in the ROI.
        depthROI = depth_map[ymin:ymax, xmin:xmax]
        inThreshRange = (self.depth_thres_low < depthROI) & (depthROI < self.depth_thres_high)
        if depthROI.any():
            averageDepth = averaging_method(depthROI[inThreshRange])
        else:
            averageDepth = 0
        #print(f"Average depth: {averageDepth}")


        mid_w = int(depth_map.shape[0] / 2) # middle of the depth img
        mid_h = int(depth_map.shape[1] / 2) # middle of the depth img
        bb_x_pos = wrist[0] - mid_w
        bb_y_pos = wrist[1] - mid_h

        angle_x = self.calc_angle(bb_x_pos)
        angle_y = self.calc_angle(bb_y_pos)

        z = averageDepth
        x = z * math.tan(angle_x)
        y = -z * math.tan(angle_y)

        #print(f"DEPTH MAP --- X: {x/10:3.0f}cm, Y: {y/10:3.0f} cm, Z: {z/10:3.0f} cm")
        return np.array([x,y,z]), (xmin, ymin, xmax, ymax)
    
def crop_to_rect(frame):
    height = frame.shape[0]
    width  = frame.shape[1]
    delta = int((width-height) / 2)
    # print(height, width, delta)
    return frame[0:height, delta:width-delta]

def get_hand_detector(type, device):

    if type == 'mediapipe':
        hand_detector = MediaPipeMonocularHandDetector(device)
    elif type == 'monocular_webcam':
        hand_detector = MediaPipeStereoHandDetector(device)
    elif type == 'hybridOAKMediapipe':
        hand_detector = HybridOAKMediapipeDetector()
    else:
        hand_detector = OAKHandDetector(device)
    return hand_detector