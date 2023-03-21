#!/usr/bin/env python3
import torch
import json
import yaml
import os
import cv2
import threading
import numpy as np
import pandas as pd
from grasp_int.image_utils import make_cameras
from grasp_int.model_utils import load_detector, load_pose_predictor
from cosypose.utils.tensor_collection import PandasTensorCollection,  fuse

from grasp_int import Scene as sc
import torch.multiprocessing as mp

_DEFAULT_CAM_SETTINGS_PATH = './default_cam_settings.yaml'
_DEFAUL_CAM_INTRINSICS = './camera0_intrinsics.json'

mp = mp.get_context('spawn')
class RigidObjectPredictor:
    def __init__(self,
                 object_coarse_run_id,
                 object_refiner_run_id,
                 intrinsics,
                 use_prior = True):
        print(intrinsics)
        self.cameras = make_cameras([intrinsics])
        self.pose_predictor = load_pose_predictor(object_coarse_run_id,
                                                  object_refiner_run_id,
                                                  preload_cache=True,
                                                  n_workers=4)
        self.pose_predictions = None
        self.pose_estimation_prior = None
        self.use_prior = use_prior
        self.threshold_nb_iter = 20

        self.K = self.cameras.K.cuda().float()
        self.n_refiner_iterations = 2
        self.emptyPrediction = RigidObjectPredictor.emptyPrediction()
        self.prediction_score_threshold = 0.8

        self.scene_objects = dict()
        self.it =0


    def predict(self, image, detections):        
        # assert len(images) == 1, 'Multi camera not supported for now'

        n_coarse_iterations=1

        predict = self.pose_estimation_prior is not None or self.detections is not None
        predict = False
        if predict:
            self.pose_predictions, _ = self.pose_predictor.get_predictions(
                images=image, K=self.K,
                data_TCO_init=self.pose_estimation_prior,
                n_coarse_iterations=n_coarse_iterations,
                n_refiner_iterations=self.n_refiner_iterations,
                detections=detections
            )
            # print('ITER')
            # print(self.pose_predictions.poses)
            # print(self.pose_predictions.poses_input)
            # print(self.pose_predictions.K_crop)
            # print(self.pose_predictions.boxes_rend)
            # print(self.pose_predictions.boxes_crop)
        else:
                self.pose_predictions = None

        if self.use_prior :
            self.pose_estimation_prior = self.pose_predictions
        else:
            self.pose_estimation_prior = None

            if self.nb_iter_without_detection <= self.threshold_nb_iter:
                self.nb_iter_without_detection +=1
            else:
                self.nb_iter_without_detection = 0
                self.pose_estimation_prior = None
                self.detect = True
        return self.pose_predictions


    def emptyPrediction():
        return PandasTensorCollection(infos=pd.DataFrame(dict(label=[],)),
                                      poses=torch.empty((0, 4, 4)).float().cuda())



class KnownObjectDetector:
    def __init__(self, device,  dataset = 'tless', render_txt =False, render_overlay = False, render_bboxes = True, detection_threshold=0.8):

        self.render_txt = render_txt
        self.render_overlay = render_overlay
        self.render_bboxes = render_bboxes
        self.img_ratio = 4/3
        cam_mat = device.matrix
        # Prepare camera infos
        intrinsics = dict(
            fx=cam_mat[0,0], cx=cam_mat[0,2],
            fy=cam_mat[1,1], cy=cam_mat[1,2],
            resolution=device.img_resolution,
        )
        self.dataset = dataset
        #self.windows = [((0,0),(1152,648))]

        if(dataset == "ycbv"):
            object_coarse_run_id = 'coarse-bop-ycbv-synt+real--822463'
            object_refiner_run_id = 'refiner-bop-ycbv-synt+real--631598'
        elif(dataset == "tless"):
            object_coarse_run_id = 'coarse-bop-tless-synt+real--160982'
            object_refiner_run_id = 'refiner-bop-tless-synt+real--881314'
        else:
            assert False
            
        self.predictor = RigidObjectPredictor(
            object_coarse_run_id,
            object_refiner_run_id,
            intrinsics
        )
        self.debug_converter = None

    def get_objects(self, image):

        # Predict poses using cosypose
        
        pose_predictions = self.predictor.predict([image, ])
        #pose_predictions = self.predictor.predict(image, detector_kwargs=detector_kwargs)
        return pose_predictions
    
    def format(self, img):
        padded_img = cv2.copyMakeBorder(img, 0,self.pad_h,0,0,cv2.BORDER_CONSTANT)
        print(padded_img.shape)
        return padded_img
    
    def format_crop(self, img):
        res=(640, 480)
        croped = img[:res[0],:res[1]]
        return croped


    def stop(self):
        pass

def get_object_detector(type, device):

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if type == 'cosypose':
        detector = KnownObjectDetector(device)
    return detector