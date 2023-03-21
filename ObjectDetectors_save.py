#!/usr/bin/env python3
import torch
import json
import yaml
import os
import numpy as np
import pandas as pd
from image_utils import make_cameras
from model_utils import load_detector, load_pose_predictor
from cosypose.utils.tensor_collection import PandasTensorCollection
from scipy.spatial.transform import Rotation as R

_DEFAULT_CAM_SETTINGS_PATH = './default_cam_settings.yaml'
_DEFAUL_CAM_INTRINSICS = './camera0_intrinsics.json'

class Position:
    def __init__(self, vect) -> None:
        self.x = vect[0]
        self.y = vect[1]
        self.z = vect[2]
        self.v = vect
    def __str__(self):
        return str(self.v)

class Orientation:
    def __init__(self, mat) -> None:
        self.r = R.from_matrix(mat)
        self.v = self.r.as_euler('xyz')
        self.x = self.v[0]
        self.y = self.v[1]
        self.z = self.v[2]
        self.q = self.r.as_quat()
        self.qx = self.q[0]
        self.qy = self.q[1]
        self.qz = self.q[2]
        self.qw = self.q[3]

    def __str__(self):
        return str(self.v)
class Pose:
    def __init__(self, tensor) -> None:
        self.mat = tensor.cpu().numpy()
        self.position = Position(self.mat[:3,3])
        self.orientation = Orientation(self.mat[:3,:3])
    
    def __str__(self):
        out = 'position : ' + str(self.position) + ' -- orientation : ' + str(self.orientation)
        return out

class RigidObject:
    def __init__(self, image_id = None, label = None, pose=None, score = None) -> None:
        self.image_id = image_id
        self.label = label
        self.pose = Pose(pose)
        self.score = score

    def __str__(self):
        out = 'label: ' + str(self.label) + '\n pose: {' +str(self.pose)+'}'
        return out
        
class RigidObjectPredictor:
    def __init__(self,
                 object_coarse_run_id,
                 object_refiner_run_id,
                 object_detector_run_id,
                 intrinsics,
                 use_prior = True):

        self.cameras = make_cameras([intrinsics])
        self.detector = load_detector(object_detector_run_id)
        self.pose_predictor = load_pose_predictor(object_coarse_run_id,
                                                  object_refiner_run_id,
                                                  preload_cache=True,
                                                  n_workers=4)
        self.pose_estimation_prior = None
        self.use_prior = use_prior
        self.nb_iter = 0

    def predict(self, images,  detector_kwargs=None):
        print('ici : ' +str(self.nb_iter))
        print('poseprior : '+str(self.pose_estimation_prior))
        if detector_kwargs is None:
            detector_kwargs = dict()

        images = torch.as_tensor(np.stack(images)).permute(0, 3, 1, 2).cuda().float() / 255
        K = self.cameras.K.cuda().float()

        if self.pose_estimation_prior == None:
            self.detections = self.detector(images, **detector_kwargs)
            n_refiner_iterations = 4
            if len(self.detections) > 0:
                self.pose_predictions, _ = self.pose_predictor.get_predictions(
                    images=images, K=K,
                    n_coarse_iterations=1,
                    n_refiner_iterations=n_refiner_iterations,
                    detections=self.detections,
                )
            else:
                self.pose_predictions = RigidObjectPredictor.emptyPrediction()
        else:
            print('la')
            print('poseprior : '+str(self.pose_estimation_prior))
            self.pose_predictions, _ = self.pose_predictor.get_predictions(
                images=images, K=K,
                data_TCO_init=self.pose_estimation_prior,
                n_coarse_iterations=0,
                n_refiner_iterations=4,
            )
        self.nb_iter+=1

        assert len(images) == 1, 'Multi camera not supported for now'
        if self.pose_predictions is not None:
            objects = self.pose_predictions
            objects = [RigidObject(
                image_id=objects.infos.loc[n, 'batch_im_id'],
                label=objects.infos.loc[n, 'label'],
                pose=objects.poses[n],
                score=objects.infos['score'][n]
            ) for n in range(len(self.pose_predictions))]
        else:
            objects = None
        if self.use_prior:
            #self.get_prior_from_objects(objects)
            self.pose_estimation_prior = self.pose_predictions
        return objects

    def emptyPrediction():
        return PandasTensorCollection(infos=pd.DataFrame(dict(label=[],)),
                                      poses=torch.empty((0, 4, 4)).float().cuda())

    def get_prior_from_objects(self, objects):
        # If warm started, prepare data
        self.pose_estimation_prior = None
        if(len(objects) > 0):
            labels = []
            poses = []
            batch_im_ids = []
            for object in objects:
                position = [object.pose.position.x, object.pose.position.y, object.pose.position.z]
                orientation = [object.pose.orientation.qx, object.pose.orientation.qy, object.pose.orientation.qz, object.pose.orientation.qw]

                poses.append([position, orientation])
                labels.append(object.label)
                batch_im_ids.append(0)
            self.pose_estimation_prior = PandasTensorCollection(infos=pd.DataFrame(dict(label=labels,batch_im_id=batch_im_ids)),
                                                           poses=torch.tensor(poses).float().cuda())
        return self.pose_estimation_prior

class KnownObjectDetector:
    def __init__(self, cam_setting_path = _DEFAULT_CAM_SETTINGS_PATH, cam_info_path=_DEFAUL_CAM_INTRINSICS, dataset = 'tless'):
        with open(cam_info_path, 'r') as json_file:
            camera_data = json.load(json_file)
            cam_mat = np.array(camera_data['matrix'])
        with open(cam_setting_path, 'r') as f:
            cam_setting = yaml.safe_load(f)

        # Prepare camera infos
        intrinsics = dict(
            fx=cam_mat[0,0], cx=cam_mat[0,2],
            fy=cam_mat[1,1], cy=cam_mat[1,2],
            resolution=(cam_setting['frame_width'], cam_setting['frame_height']),
        )
        self.dataset = dataset

        if(dataset == "ycbv"):
            object_coarse_run_id = 'coarse-bop-ycbv-synt+real--822463'
            object_refiner_run_id = 'refiner-bop-ycbv-synt+real--631598'
            object_detector_run_id = 'detector-bop-ycbv-synt+real--292971'
        elif(dataset == "tless"):
            object_coarse_run_id = 'coarse-bop-tless-synt+real--160982'
            object_refiner_run_id = 'refiner-bop-tless-synt+real--881314'
            object_detector_run_id = 'detector-bop-tless-synt+real--452847'
        else:
            assert False

        self.predictor = RigidObjectPredictor(
            object_coarse_run_id,
            object_refiner_run_id,
            object_detector_run_id,
            intrinsics
        )
        print('init known')
        print(self.predictor.pose_estimation_prior)
        self.debug_converter = None

    def get_objects(self, image, detection_threshold=0.7):

        detector_kwargs = dict(one_instance_per_class=False, detection_th=detection_threshold)

        # Predict poses using cosypose
        
        pose_predictions = self.predictor.predict([image, ], detector_kwargs=detector_kwargs)

        return pose_predictions



def get_object_detector(type):

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if type == 'cosypose':
        detector = KnownObjectDetector()
    return detector