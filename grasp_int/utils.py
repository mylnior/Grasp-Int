#!/usr/bin/env python3

from grasp_int.Filters import LandmarksSmoothingFilter
import numpy as np
from scipy.spatial.transform import Rotation as R

import cv2

class Position:
    def __init__(self, vect, display='cm') -> None:
        '''Position in millimeters'''
        vect[1] = -vect[1]
        self.x = vect[0]
        self.y = vect[1]
        self.z = vect[2]
        self.v = vect
        self.display = display  
        self.ve = np.hstack((self.v,1)) 

    def __str__(self):
        if self.display == 'cm':
            return str(self.v*0.1)+' (in cm)'
        else:
            return str(self.v)+' (in mm)'

    
    def __call__(self):
        return self.v

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
    def __init__(self, tensor,position_factor=1, orientation_factor=1) -> None:
        self.min_cutoffs = {'position' : 0.001, 'orientation' : 0.0001}
        self.betas = {'position' : 100, 'orientation' : 10}
        self.derivate_cutoff = {'position' : 0.1, 'orientation' : 1}
        attributes = ('position', 'orientation')
        self.filters={}
        self.position_factor = position_factor
        self.orientation_factor = orientation_factor
        for key in attributes:
            self.filters[key] = LandmarksSmoothingFilter(min_cutoff=self.min_cutoffs[key], beta=self.betas[key], derivate_cutoff=self.derivate_cutoff[key], disable_value_scaling=True)
        self.update(tensor)

    def update(self, tensor):
        self.mat = tensor.cpu().numpy()
        self.raw_position = Position(self.mat[:3,3]*self.position_factor)
        # self.position = Position(self.filters['position'].apply(self.mat[:3,3]*self.position_factor))
        # self.position = self.raw_position
        self.position = Position(self.filters['position'].apply(self.mat[:3,3])*self.position_factor)
        # print('self.raw_position')
        # print(self.raw_position)
        # print('self.position')
        # print(self.position)
        # self.raw_orientation = Orientation(np.identity(3)*self.orientation_factor)
        mat = self.mat[:3,:3]
        # mat = np.identity(3)
        self.raw_orientation = Orientation(mat*self.orientation_factor)
        self.orientation = Orientation(self.filters['orientation'].apply(mat)*self.orientation_factor)
        # self.orientation = self.raw_orientation
    
    def __str__(self):
        out = 'position : ' + str(self.position) + ' -- orientation : ' + str(self.orientation)
        return out
    
class Bbox:

    def __init__(self, img_resolution, label, tensor) -> None:
        self.filter = LandmarksSmoothingFilter(min_cutoff=0.001, beta=0.5, derivate_cutoff=1, disable_value_scaling=True)
        
        self.label=label
        if self.label == 'obj_000028':
            self.color = (0, 0, 255)
        else:
            self.color = (255, 0, 0)
        self.thickness = 2
        self.img_resolution = img_resolution
        self.update_coordinates(tensor)

    def draw(self, img):
        cv2.rectangle(img, self.corner1, self.corner2, self.color, self.thickness)
    
    def update_coordinates(self,tensor):
        box = self.filter.apply(tensor.cpu().numpy())
        self.corner1 = (min(int(box[0]), self.img_resolution[0]-self.thickness), min(int(box[1]), self.img_resolution[1]-self.thickness))
        self.corner2 = (min(int(box[2]), self.img_resolution[0]-self.thickness), min(int(box[3]), self.img_resolution[1]-self.thickness))

    def __str__(self) -> str:
        s= 'Box <'+self.label+'> : ('+str(self.corner1)+','+str(self.corner2)+')'
        return s