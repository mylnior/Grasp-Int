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
    
    def update_display(self, color, thickness):
        self.color = color
        self.thickness = thickness

    
class DataWindow:
    def __init__(self, size:int) -> None:
        self.size = size # nb iterations or time limit ?
        self.data = list()
        self.nb_samples = 0
    
    def queue(self, new_data):
        self.data.append(new_data)
        if len(self.data)>= self.size:
            del self.data[0]
        else:
            self.nb_samples+=1



class HandConeImpactsWindow(DataWindow):
    def __init__(self, size: int) -> None:
        super().__init__(size)
        self.nb_impacts = 0

    def mean(self):
        if self.nb_samples==0:
            return None
        sum = 0
        for i in range(self.data):
            sum+=np.mean(self.data[i])
        return sum/self.nb_samples

    def queue(self, new_data):
        self.data.append(new_data)
        if len(self.data)>= self.size:
            deleted =  self.data.pop(0)
            self.nb_impacts = self.nb_impacts - len(deleted) + len(new_data)
        else:
            self.nb_impacts = self.nb_impacts + len(new_data)
            self.nb_samples+=1
    
class TargetDetector:
    def __init__(self, hand_label, window_size = 100) -> None:
        self.window_size = window_size
        self.potential_targets = {}
        self.hand_label = hand_label
        pass
    
    def new_impacts(self, obj, impacts):
        label = obj.label
        if label in self.potential_targets:
            self.potential_targets[label].update(impacts) 
        else:
            self.potential_targets[label] = Target(obj, impacts)

    def get_most_probable_target(self):
        if self.potential_targets:
            n_impacts = {}
            n_tot = 0
            to_del_keys=[]
            for lab, target in self.potential_targets.items():
                n_impacts[lab] = target.projected_collison_window.nb_impacts
                if n_impacts[lab]<=0:
                    to_del_keys.append(lab)
                n_tot +=n_impacts[lab]

            for key in to_del_keys:
                print('del key', key)
                # del self.potential_targets[key]

            if n_tot == 0:
                most_probable_target =  None                
            else:
                for lab in self.potential_targets:
                    self.potential_targets[lab].set_impact_ratio(n_impacts[lab]/n_tot)
                max_ratio_label = max(n_impacts, key = n_impacts.get)
                most_probable_target = self.potential_targets[max_ratio_label]
        else:
            most_probable_target =  None
        
        # print(self.hand_label,' most_probable_target : ',most_probable_target)
        return most_probable_target, self.potential_targets

class Target:
    def __init__(self, obj, impacts, window_size = 10) -> None:
        self.object = obj
        self.label = obj.label
        self.window_size = window_size
        self.projected_collison_window = HandConeImpactsWindow(window_size)
        self.update(impacts)
        print('POTENTIAL TARGET ')
        self.ratio=0
        

    def update(self, impacts):        
        self.projected_collison_window.queue(impacts)
    
    def set_impact_ratio(self, ratio):
        self.ratio = ratio

    def __str__(self) -> str:
        out = 'Target: '+self.object.label + ' - nb impacts: ' + str(self.projected_collison_window.nb_impacts) + ' - ratio: ' + str(self.ratio)
        return out
    def get_proba(self):
        return self.ratio

class GripSelector:
    def __init__(self) -> None:
        pass