#!/usr/bin/env python3


from HandTrackerRenderer import HandTrackerRenderer
import argparse

import cv2
import os
from pose_estimation_cv2 import Predictor

from cosypose.config import LOCAL_DATA_DIR

_RELATIVE_MODEL_PATH_TLESS = '/bop_datasets/tless/models_cad/'

class PoseEstimationLoop:
    def __init__(self, cam_setting_path, cam_calib_path, camera_index, allow_tracking, dataset):
        self.camera_index = camera_index
        self.allow_tracking = allow_tracking
        self.dataset = dataset
        self.models_path = str(LOCAL_DATA_DIR)+_RELATIVE_MODEL_PATH_TLESS
        self.prediction = None
        self.found_objects = []
        self.predictor = Predictor(cam_setting_path, cam_calib_path, dataset)

    def visualize_objects(self):
        for obj in self.found_objects:
            m_file = self.models_path+obj["label"]+'.ply'
            cv2.viz.read_mesh(m_file)

    def main(self):
        cap = cv2.VideoCapture(self.camera_index,cv2.CAP_V4L2)
        cap.set(3, 1280)
        cap.set(4, 720)
        while cap.isOpened() :
            success, img = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue
            prior = []
            if self.allow_tracking and self.prediction :
                prior = self.found_objects
            
            self.found_objects = self.predictor.pose_estimation(img, prior)
            for obj in self.found_objects:
                print('object detected : '+ str(obj))
                #self.visualize_objects()
            else:
                print('nothing')
            cv2.imshow('MediaPipe Hands', img)
            
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()
    def main_single_frame(self):
        prior = []
        img = cv2.imread('./tset.png')        
        self.found_objects = self.predictor.pose_estimation(img, prior)
        for obj in self.found_objects:
            print('object detected : '+ str(obj))
            #self.visualize_objects()
        else:
            print('nothing')


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--edge', action="store_true",
                        help="Use Edge mode (postprocessing runs on the device)")
    parser_tracker = parser.add_argument_group("Tracker arguments")
    parser_tracker.add_argument('-i', '--input', type=str, 
                        help="Path to video or image file to use as input (if not specified, use OAK color camera)")
    parser_tracker.add_argument("--pd_model", type=str,
                        help="Path to a blob file for palm detection model")
    parser_tracker.add_argument('--no_lm', action="store_true", 
                        help="Only the palm detection model is run (no hand landmark model)")
    parser_tracker.add_argument("--lm_model", type=str,
                        help="Landmark model 'full', 'lite', 'sparse' or path to a blob file")
    parser_tracker.add_argument('--use_world_landmarks', action="store_true", 
                        help="Fetch landmark 3D coordinates in meter")
    parser_tracker.add_argument('-s', '--solo', action="store_true", 
                        help="Solo mode: detect one hand max. If not used, detect 2 hands max (Duo mode)")                    
    parser_tracker.add_argument('-xyz', "--xyz", action="store_true", 
                        help="Enable spatial location measure of palm centers")
    parser_tracker.add_argument('-g', '--gesture', action="store_true", 
                        help="Enable gesture recognition")
    parser_tracker.add_argument('-c', '--crop', action="store_true", 
                        help="Center crop frames to a square shape")
    parser_tracker.add_argument('-f', '--internal_fps', type=int, 
                        help="Fps of internal color camera. Too high value lower NN fps (default= depends on the model)")                    
    parser_tracker.add_argument("-r", "--resolution", choices=['full', 'ultra'], default='full',
                        help="Sensor resolution: 'full' (1920x1080) or 'ultra' (3840x2160) (default=%(default)s)")
    parser_tracker.add_argument('--internal_frame_height', type=int,                                                                                 
                        help="Internal color camera frame height in pixels")   
    parser_tracker.add_argument("-lh", "--use_last_handedness", action="store_true",
                        help="Use last inferred handedness. Otherwise use handedness average (more robust)")                            
    parser_tracker.add_argument('--single_hand_tolerance_thresh', type=int, default=10,
                        help="(Duo mode only) Number of frames after only one hand is detected before calling palm detection (default=%(default)s)")
    parser_tracker.add_argument('--dont_force_same_image', action="store_true",
                        help="(Edge Duo mode only) Don't force the use the same image when inferring the landmarks of the 2 hands (slower but skeleton less shifted)")
    parser_tracker.add_argument('-lmt', '--lm_nb_threads', type=int, choices=[1,2], default=2, 
                        help="Number of the landmark model inference threads (default=%(default)i)")  
    parser_tracker.add_argument('-t', '--trace', type=int, nargs="?", const=1, default=0, 
                        help="Print some debug infos. The type of info depends on the optional argument.")                
    parser_renderer = parser.add_argument_group("Renderer arguments")
    parser_renderer.add_argument('-o', '--output', 
                        help="Path to output video file")
    args = parser.parse_args()
    dargs = vars(args)
    tracker_args = {a:dargs[a] for a in ['pd_model', 'lm_model', 'internal_fps', 'internal_frame_height'] if dargs[a] is not None}

    if args.edge:
        from HandTrackerEdge import HandTracker
        tracker_args['use_same_image'] = not args.dont_force_same_image
    else:
        from HandTracker2 import HandTracker


    tracker = HandTracker(
            input_src=args.input, 
            use_lm= not args.no_lm, 
            use_world_landmarks=args.use_world_landmarks,
            use_gesture=args.gesture,
            xyz=args.xyz,
            solo=args.solo,
            crop=args.crop,
            resolution=args.resolution,
            stats=True,
            trace=args.trace,
            use_handedness_average=not args.use_last_handedness,
            single_hand_tolerance_thresh=args.single_hand_tolerance_thresh,
            lm_nb_threads=args.lm_nb_threads,
            **tracker_args
            )

    renderer = HandTrackerRenderer(
            tracker=tracker,
            output=args.output)

    while True:
        # Run hand tracker on next frame
        # 'bag' contains some information related to the frame 
        # and not related to a particular hand like body keypoints in Body Pre Focusing mode
        # Currently 'bag' contains meaningful information only when Body Pre Focusing is used
        frame, hands, bag = tracker.next_frame()
        if frame is None: break
        # Draw hands
        frame = renderer.draw(frame, hands, bag)
        key = renderer.waitKey(delay=1)
        if key == 27 or key == ord('q'):
            break
    renderer.exit()
    tracker.exit()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    camera_index = 0
    allow_tracking = False
    dataset = "tless"
    cam_setting_path = './default_cam_settings.yaml'
    cam_calib_path = './camera0_intrinsics.json'
    PoseEstimationLoop(cam_setting_path, cam_calib_path, camera_index, allow_tracking, dataset).main()


