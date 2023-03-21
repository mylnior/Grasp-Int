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
                 object_detector_run_id,
                 intrinsics,
                 use_prior = False,
                 windows = None,
                 detector_kwargs = None):
        print(intrinsics)
        self.cameras = make_cameras([intrinsics])
        os.system('nvidia-smi | grep python')
        self.detector = load_detector(object_detector_run_id)
        os.system('nvidia-smi | grep python')
        self.detector_windows = load_detector(object_detector_run_id)
        os.system('nvidia-smi | grep python')
        #self.pose_predictor = load_pose_predictor(object_coarse_run_id,
         #                                         object_refiner_run_id,
          #                                        preload_cache=True,
           #                                       n_workers=4)
        self.pose_predictions = None
        self.pose_estimation_prior = None
        self.use_prior = use_prior
        self.nb_iter_without_detection = 0
        self.threshold_nb_iter = 20
        self.detect = True
        self.detector_kwargs = detector_kwargs

        self.K = self.cameras.K.cuda().float()
        self.n_refiner_iterations = 2
        self.emptyPrediction = RigidObjectPredictor.emptyPrediction()
        self.prediction_score_threshold = 0.8

        self.scene_objects = dict()
        self.windows = windows
        self.it =0


    def forward_pass_full_detector(self):
        self.detections = self.detector(self.images_windows,**self.detector_kwargs)

    def forward_pass_windows_detector(self):
        self.detections_windows = self.detector_windows(self.images_windows,**self.detector_kwargs)


    def predict(self, image):        
        # assert len(images) == 1, 'Multi camera not supported for now'
        with torch.no_grad():

            n_coarse_iterations=1

            # images = torch.as_tensor(np.stack(images)).permute(0, 3, 1, 2).cuda().float() / 255

            # if self.detect:
            #     self.detections = self.detector(images, **detector_kwargs)
            #     if len(self.detections) <= 0:
            #         self.detections=None
            #     self.detect = False
            #     #fuse(self.pose_predictions, self.detections)
            # else:
            #     self.detections = None
            print(self.windows)
            # win_images = [torch.as_tensor(image[w[0][0] : w[1][0], w[0][1] : w[1][1]] ).permute(2, 0, 1) / 255 for w in self.windows]
            print('ITERATION NUMBER '+str(self.it+1))
            os.system('nvidia-smi | grep python')
            self.images_windows = torch.as_tensor(np.stack([image[0][w[0][1] : w[1][1], w[0][0] : w[1][0]] for w in self.windows] )).permute(0, 3, 1, 2).cuda().float()/ 255
            
            os.system('nvidia-smi | grep python')
            self.image_full = torch.as_tensor(np.stack(image)).permute(0, 3, 1, 2).cuda().float() / 255
            print('models loaded, not run yet : ')
            os.system('nvidia-smi | grep python')
            # win_images = [image[w[0][0]:w[1][0], w[0][1]:w[1][1]]  for w in self.windows]
            # i=0
            # for im in win_images:
            #     print(self.windows[i])
            #     print(im.shape)
            #     i+=1
            # print(image.shape)
            # print(image[320:960, 140:170].shape)
            # exit()
            if self.detect:
                self.p_full = threading.Thread(target=self.forward_pass_full_detector)
                self.p_win = threading.Thread(target=self.forward_pass_windows_detector)
                self.p_full.start()
                self.p_win.start()
                print('bvwxjcksj,nb xcn,xk,')
                self.p_full.join()
                print('JSOKJNQSDKCJNSBZDKJNSBDJSN')
                self.p_win.join()
                #detections_full = self.detector(image, **detector_kwargs)
                #detections_windows = self.detector_windows(win_images, **detector_kwargs)
                print('models run ONCE : ')
                os.system('nvidia-smi | grep python')
                torch.cuda.empty_cache()
                print('CACHE CLEARED')
                os.system('nvidia-smi | grep python')
                if self.it !=0:
                    pass
                    #exit()
                self.it+=1
                detections_windows=None
                detections_full = None
                #if len(detections_full) <= 0:
                #    detections_full=None
                #if len(detections_windows) <= 0 or detections_windows is None:
                #    detections_windows=None
                #fuse(self.pose_predictions, self.detections)
                if detections_windows is not None:
                    print(detections_windows)
                    det_win_infos = detections_windows.infos
                    filtered_det_win_infos= det_win_infos.sort_values('score',ascending = False).drop_duplicates('label').sort_index()
                    ids = filtered_det_win_infos.index
                    im_ids = [i - 1 for i in filtered_det_win_infos['batch_im_id']]
                    gaps = [[w[0][0], w[0][0], w[1][0], w[1][0]] for w in [self.windows[j] for j in im_ids]]
                    gaps = torch.as_tensor(gaps).cuda().float()
                    print(filtered_det_win_infos)
                    print(gaps)
                    det_win_tensors = detections_windows.tensors['bboxes']
                    print(det_win_tensors)
                    print(im_ids)
                    filtered_det_win_tensors = det_win_tensors[ids,:]
                    print(filtered_det_win_tensors)
                    filtered_det_win_tensors = torch.add(filtered_det_win_tensors, gaps)
                    print(filtered_det_win_tensors)
                    # kept_infos = {}
                    # for l in det_win_infos['label']:

                    self.detections = PandasTensorCollection(
                    infos=pd.DataFrame(filtered_det_win_infos),
                    bboxes=filtered_det_win_tensors,
                    )
                    self.detections.infos['batch_im_id'] = [0 for i in range(len(self.detections.infos['label']))]
                    if self.use_prior:
                        self.detect = False
                else: 
                    self.detections = None
                    self.detect =True
                # exit()
                print(self.detections)
                print('LA')
            else:
                self.detections = None
            # if self.detect:
            #     self.detections = None
            #     i = 0
            #     for w_img in win_images:
            #         win_detections = self.detector(w_img, **detector_kwargs)
            #         if len(win_detections) <= 0:
            #             win_detections=None
            #         print(win_detections)
            #         if self.detections is not None:
            #             self.detections.add(win_detections)
            #         else:
            #             self.detections = win_detections
            #         i+=1
            #         print('win : ' + str(i))
            #     self.detect = False
            #     #fuse(self.pose_predictions, self.detections)
            # else:
            #     self.detections = None

            predict = self.pose_estimation_prior is not None or self.detections is not None
            predict = False
            if predict:
                self.pose_predictions, _ = self.pose_predictor.get_predictions(
                    images=image, K=self.K,
                    data_TCO_init=self.pose_estimation_prior,
                    n_coarse_iterations=n_coarse_iterations,
                    n_refiner_iterations=self.n_refiner_iterations,
                    detections=self.detections,
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

def find_overlapping_windows(res, max_obj_size, window_size):
    width_filled = False
    height_filled = False
    img_width = res[0]
    img_height = res[1]
    windows_width = window_size[0]
    windows_height = window_size[1]
    obj_width = max_obj_size[0]
    obj_height = max_obj_size[1]
    dw = windows_width-obj_width
    dh = windows_height-obj_height
    dcoutx = img_width-windows_width
    nb_width = np.ceil(img_width/dw).astype(int)
    nb_height = 2
    print(nb_width)
    x1 = img_width - (nb_width-1)*windows_width
    y1 = img_height - (nb_height-1)*windows_height
    print(dw)
    xs = [0] + [np.floor(x1 + i*windows_width).astype(int) for i in range(nb_width-2)]+[img_width-windows_width]
    ys = [0] + [np.floor(y1 + i*windows_height).astype(int) for i in range(nb_height-2)]+[img_height-windows_height]
    xs = [0] + [int(img_width/2 - windows_width/2)]+[img_width-windows_width]
    ys = [0] + [img_height-windows_height]
    print(xs)
    print(ys)
    windows = []
    for i in range(nb_width):
        for j in range(nb_height):
            windows.append((( xs[i], ys[j]),(xs[i]+windows_width, ys[j]+windows_height)))
    return windows
    # x_corner1_left_box = 0
    # y_corner1_left_box = 0
    # x_corner2_left_box = x_corner1_left_box + windows_width
    # y_corner2_left_box = y_corner1_left_box + windows_height
    # boxes_xs = [(x_corner1_left_box, x_corner2_left_box)]
    # boxes_ys = [(y_corner1_left_box, y_corner2_left_box)]
    # x_corner1_right_box = img_width-windows_width
    # y_corner1_right_box = img_height-windows_height
    # if x_corner1_right_box <=0:
    #     width_filled = True
    # if x_corner1_right_box <=0:
    #     width_filled = True
    # while not width_filled:
    #     x=0

class KnownObjectDetector:
    def __init__(self, device, cam_setting_path = _DEFAULT_CAM_SETTINGS_PATH, cam_info_path=_DEFAUL_CAM_INTRINSICS,
     dataset = 'tless', render_txt =False, render_overlay = False, render_bboxes = True, detection_threshold=0.8):

        self.render_txt = render_txt
        self.render_overlay = render_overlay
        self.render_bboxes = render_bboxes
        self.img_ratio = 4/3
        self.pad_h = int(self.img_ratio*device.img_resolution[0] - device.img_resolution[1])
        cam_mat = device.matrix
        self.img_resolution = device.img_resolution
        print(self.img_resolution)
        # Prepare camera infos
        intrinsics = dict(
            fx=cam_mat[0,0], cx=cam_mat[0,2],
            fy=cam_mat[1,1], cy=cam_mat[1,2],
            resolution=device.img_resolution,
        )
        self.dataset = dataset
        s=200
        self.max_object_size=(s,s) #minimum size of objects in pixels
        self.window_size = (640,480)
        self.windows = find_overlapping_windows(self.img_resolution, self.max_object_size,self.window_size)
        #self.windows = [((0,0),(1152,648))]

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

        detector_kwargs = dict(one_instance_per_class=False, detection_th=detection_threshold)

        self.predictor = RigidObjectPredictor(
            object_coarse_run_id,
            object_refiner_run_id,
            object_detector_run_id,
            intrinsics,
            windows = self.windows,
            detector_kwargs=detector_kwargs
        )
        self.debug_converter = None

    def get_objects(self, image):

        # Predict poses using cosypose
        
        pose_predictions = self.predictor.predict([image, ])
        #pose_predictions = self.predictor.predict(image, detector_kwargs=detector_kwargs)
        j=0
        for w in self.windows:
            j+=1
            cv2.rectangle(image,w[0],w[1], (j*30,255-j*30,j*25),2)
        return pose_predictions
    
    def format(self, img):
        padded_img = cv2.copyMakeBorder(img, 0,self.pad_h,0,0,cv2.BORDER_CONSTANT)
        print(padded_img.shape)
        return padded_img
    
    def format_crop(self, img):
        res=(640, 480)
        croped = img[:res[0],:res[1]]
        return croped


    # def render(self, img):
    #     if self.predictor.pose_predictions is not None:
    #         if self.render_txt:
    #             self.txt(img)
    #         if self.render_overlay:
    #             self.overlay(img)
    #         if self.render_bboxes:
    #             self.bboxes(img)

    # def txt(self,img):
    #     pass
    
    # def overlay(self, img):
    #     pass

    # def bboxes(self, img):
    #     box = self.predictor.pose_predictions.boxes_rend.cpu().numpy()
    #     a = (int(box[0,0]), int(box[0,1]))
    #     b = (int(box[0,2]), int(box[0,3]))
    #     color = (255, 0, 0)
    #     thik = 2
    #     cv2.rectangle(img, a, b, color, thik )
    #     pass
    def stop(self):
        pass

def get_object_detector(type, device):

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if type == 'cosypose':
        detector = KnownObjectDetector(device)
    return detector