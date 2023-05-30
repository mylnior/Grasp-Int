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
class Object2DDetector:
    def __init__(self,
                 device,
                 detection_threshold=0.8,
                 dataset = 'tless'):

        if(dataset == "ycbv"):
            object_detector_run_id = 'detector-bop-ycbv-synt+real--292971'
        elif(dataset == "tless"):
            object_detector_run_id = 'detector-bop-tless-synt+real--452847'
        else:
            assert False

        self.img_resolution = device.img_resolution
        os.system('nvidia-smi | grep python')
        self.detector = load_detector(object_detector_run_id)
        os.system('nvidia-smi | grep python')
        # self.detector_windows = load_detector(object_detector_run_id) 
        # os.system('nvidia-smi | grep python')

        self.nb_iter_without_detection = 0
        self.threshold_nb_iter = 20
        self.detecting = True
        self.detector_kwargs = dict(one_instance_per_class=False, detection_th=detection_threshold)

        self.detections = None
        self.prediction_score_threshold = 0.8

        self.it =0


    def forward_pass_full_detector(self):
        self.detections = self.detector(self.image_full,**self.detector_kwargs)

    def forward_pass_windows_detector(self):
        self.detections_windows = self.detector_windows(self.image_full,**self.detector_kwargs)


    def detect(self, image):        

        #print(self.windows)
        # print('ITERATION NUMBER '+str(self.it+1))
        # os.system('nvidia-smi | grep python')
        scale_percent = 60 # percent of original size
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        
        # resize image
        # print(image.shape)
        # image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        # print('RESIZE')
        # print(image.shape)
        # self.images_windows = torch.as_tensor(np.stack([image[w[0][1] : w[1][1], w[0][0] : w[1][0]] for w in self.windows] )).permute(0, 3, 1, 2).cuda().float()/ 255
        
        # os.system('nvidia-smi | grep python')
        self.image_full = torch.as_tensor(np.stack([image, ])).permute(0, 3, 1, 2).cuda().float() / 255
        # print('models loaded, not run yet : ')
        # os.system('nvidia-smi | grep python')
        if self.detecting:
            self.p_full = threading.Thread(target=self.forward_pass_full_detector)
            # self.p_win = [threading.Thread(target=self.forward_pass_windows_detector)]
            self.p_full.start()
            # for p in self.p_win:
            #     p.start()
            # print('bvwxjcksj,nb xcn,xk,')
            self.p_full.join()
            # print('JSOKJNQSDKCJNSBZDKJNSBDJSN')
            # for p in self.p_win:
            #     p.join()
            # print('models run ONCE : ')
            # os.system('nvidia-smi | grep python')
            # torch.cuda.empty_cache()
            # print('CACHE CLEARED')
            # os.system('nvidia-smi | grep python')
            # if self.it !=0:
            #     pass
            #     #exit()
            # self.it+=1
            # detections_windows=None
            # detections_full = None
            # if detections_windows is not None:
            #     print(detections_windows)
            #     det_win_infos = detections_windows.infos
            #     filtered_det_win_infos= det_win_infos.sort_values('score',ascending = False).drop_duplicates('label').sort_index()
            #     ids = filtered_det_win_infos.index
            #     im_ids = [i - 1 for i in filtered_det_win_infos['batch_im_id']]
            #     gaps = [[w[0][0], w[0][0], w[1][0], w[1][0]] for w in [self.windows[j] for j in im_ids]]
            #     gaps = torch.as_tensor(gaps).cuda().float()
            #     print(filtered_det_win_infos)
            #     print(gaps)
            #     det_win_tensors = detections_windows.tensors['bboxes']
            #     print(det_win_tensors)
            #     print(im_ids)
            #     filtered_det_win_tensors = det_win_tensors[ids,:]
            #     print(filtered_det_win_tensors)
            #     filtered_det_win_tensors = torch.add(filtered_det_win_tensors, gaps)
            #     print(filtered_det_win_tensors)
            #     self.detections = PandasTensorCollection(
            #     infos=pd.DataFrame(filtered_det_win_infos),
            #     bboxes=filtered_det_win_tensors,
            #     )
            #     self.detections.infos['batch_im_id'] = [0 for i in range(len(self.detections.infos['label']))]
            #     if self.use_prior:
            #         self.detecting = False
            # else: 
            #     self.detections = None
            #     self.detecting =True
            # exit()
            if len(self.detections)<=0:
                self.detections=None
        else:
            self.detections = None
        return self.detections
    
    def stop(self):
        pass

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
def get_object_detector(type, device):

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if type == 'cosypose':
        detector = Object2DDetector(device)
    return detector