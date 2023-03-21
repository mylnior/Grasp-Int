#!/usr/bin/env python3

import threading
import time
from scipy.interpolate import CubicSpline

import cv2
from quaternions import quaternion
import numpy as np
import trimesh as tm
import trimesh.scene.cameras as cam
import trimesh.viewer as vw
from scipy.spatial.transform import Rotation as R

from grasp_int.depthai_hand_tracker.HandTrackerRenderer import \
    HandTrackerRenderer
from grasp_int.Filters import LandmarksSmoothingFilter

_DEFAULT_RENDERING_OPTIONS=dict(write_fps = True, 
                                draw_hands=True, 
                                draw_objects=False, 
                                write_hands_pos = False, 
                                write_objects_pos = False,
                                render_objects=True)
_TLESS_MESH_PATH = '/home/emoullet/Documents/DATA/cosypose/local_data/bop_datasets/tless/models_cad/'
_YCVB_MESH_PATH = '/home/emoullet/Documents/DATA/cosypose/local_data/bop_datasets/ycbv/models/'

class Scene :
    def __init__(self, device = None, name = 'Grasping experiment', hand_detector=None, object_detector=None, rendering_options = _DEFAULT_RENDERING_OPTIONS) -> None:
        self.hands = dict()
        self.objects = dict()
        self.cam_data = dict()
        self.cam_data['res'] = device.img_resolution
        self.cam_data['K'] = device.matrix
        #self.cam['RT'] = device.extrinsics
        self.hand_detector = hand_detector
        self.object_detector = object_detector
        self.rendering_options = rendering_options
        self.time_scene = time.time()
        self.time_hands = self.time_scene
        self.time_objects = self.time_hands
        self.time_detections = self.time_hands
        self.fps_scene = 0
        self.fps_hands= 0
        self.fps_objects= 0
        self.fps_detections= 0
        self.name = name
        self.draw_mesh = True
        self.new_hand_meshes = []
        if self.draw_mesh:
            self.define_mesh_scene()

    def __str__(self) -> str:
        s = 'SCENE \n OBJECTS :\n'
        for obj in self.objects.values():
            s+=str(obj)+'\n'
        return s
    
    def display_meshes(self):
        self.mesh_scene.show(callback=self.update_meshes,callback_period=1.0/15.0)

    def update_hands_meshes(self, scene):
        for i in range(len(self.new_hand_meshes)):
            new = self.new_hand_meshes.pop(0)
            self.mesh_scene.add_geometry(new['mesh'], geom_name = new['name'])
        
        for label, hand in self.hands.items():
            hand_mesh_pos = tm.transformations.compose_matrix(translate = hand.mesh_position.v)
            scene.graph.update(label,matrix = hand_mesh_pos , geometry = label)
            vdir = hand.normed_velocity
            zaxis = np.array([0,0,1])
            vel_rot_mat = rotation_from_vectors(zaxis,vdir)
            cone_len = self.cone_lenghts_spline(hand.scalar_velocity)
            cone_diam = self.cone_diam_spline(hand.scalar_velocity)
            x_reflection_matrix = tm.transformations.reflection_matrix(np.array([0,0,0]), np.array([1,0,0]))
            z_reflection_matrix = tm.transformations.reflection_matrix(np.array([0,0,cone_len/2]), np.array([0,0,1]))
            tot_mat = hand_mesh_pos @ x_reflection_matrix @ vel_rot_mat @ z_reflection_matrix
            hand.velocity_cone = tm.creation.cone(cone_diam, cone_len)
            scene.delete_geometry(label+'vel_cone')
            
            scene.add_geometry(hand.velocity_cone, geom_name = label+'vel_cone', transform = tot_mat)

    def update_object_meshes(self, scene):
        for label, obj in self.objects.items():
            mesh_pos = obj.pose.position.v*1000*np.array([-1,1,1])
            mesh_orient_quat = obj.pose.raw_orientation.q
            mesh_orient_angles = obj.pose.raw_orientation.v*np.array([-1,-1,-1])+np.pi*np.array([1  ,1,0])
            x_reflection_matrix = tm.transformations.reflection_matrix(np.array([0,0,0]), np.array([1,0,0]))
            mesh_transform = tm.transformations.translation_matrix(mesh_pos)  @ tm.transformations.quaternion_matrix(mesh_orient_quat)
            mesh_transform = tm.transformations.translation_matrix(mesh_pos) @ tm.transformations.euler_matrix(mesh_orient_angles[0],mesh_orient_angles[1],mesh_orient_angles[2])
            scene.graph.update(label,matrix = mesh_transform)
            # for hand in self.hands.values():
            #     intersect = tm.boolean.intersection(obj.mesh,hand.velocity_cone)
            #     print(type(intersect))

    def update_meshes(self, scene):
        self.update_hands_meshes(scene)
        self.update_object_meshes(scene)

    def define_mesh_scene(self):
        self.mesh_scene= tm.Scene()
        self.mesh_scene.camera.resolution = self.cam_data['res']
        self.mesh_scene.camera.focal= (self.cam_data['K'][0,0], self.cam_data['K'][1,1])
        self.mesh_scene.camera.z_far = 3000
        X =  0
        Y =  -250
        Z = 1000
        self.mesh_scene.camera_transform = tm.transformations.rotation_matrix(np.pi, np.array([0,1,0]), np.array([0,0,0]))
        base_origin = tm.primitives.Sphere(radius = 50)
        base_origin.visual.face_colors = [0,255,0,255]
        frame_origin = np.array([[X, Y, Z],
                    [X, Y, Z],[X, Y, Z]])
        frame_axis_directions = np.array([[0, 0, 1],
                        [0, 1, 0],[1, 0, 0]])
        frame_visualize = tm.load_path(np.hstack((
        frame_origin,
        frame_origin + frame_axis_directions*100)).reshape(-1, 2, 3))
        self.mesh_scene.add_geometry(frame_visualize, geom_name='mafreme')
        plane = tm.path.creation.grid(300, count = 10, plane_origin =  np.array([X, Y, Z]), plane_normal = np.array([0,1,0]))
        cam = tm.creation.camera_marker(self.mesh_scene.camera, marker_height = 300)

        cone_max_length = 500
        cone_min_length = 100
        vmin=0.
        vmax=300
        cone_max_diam = 50
        cone_min_diam = 10
        self.cone_lenghts_spline = spline(vmin,vmax, cone_min_length, cone_max_length)
        self.cone_diam_spline = spline(vmin,vmax, cone_min_diam, cone_max_diam)
        # lambda_length = 


        self.mesh_scene.add_geometry(plane, geom_name='plane')
        self.mesh_scene.add_geometry(cam, geom_name='macamira')
        self.mesh_scene.add_geometry(base_origin, geom_name='base_origin')
        self.t_scene = threading.Thread(target=self.display_meshes)
        self.t_scene.start()

    def evaluate_grasping_intention(self):
        for obj in self.objects.values():
            for hand in self.hands.values():
                if hand.label=='right':
                    obj.is_targeted_by(hand)

    def propagate_hands(self):
        for hand in self.hands.values():
            hand.propagate()

    def clean_hands(self, hands):
        hands_label = [hand.label for hand in hands]
        for label in self.hands:
            self.hands[label].setvisible(label in hands_label)

    def update_hands(self, hands_predictions):
        self.update_hands_fps()
        # print('hands_predictions')
        # print(hands_predictions)
        for hand_pred in hands_predictions:
            # print('hand_pred.label')
            # print(hand_pred.label)
            if hand_pred.label not in self.hands:
                self.new_hand(hand_pred)
            else : 
                self.hands[hand_pred.label].update(hand_pred)
        self.clean_hands(hands_predictions)
        self.propagate_hands()
        # self.evaluate_grasping_intention()
    
    def new_hand(self, pred):
        self.hands[pred.label] = Hand(pred, self.hand_detector.device)
        self.new_hand_meshes.append({'mesh' : self.hands[pred.label].mesh_origin, 'name': pred.label})

        
    def update_objects(self, objects_predictions):
        if objects_predictions is not None:
            for n in range(len(objects_predictions)):
                label=objects_predictions.infos.loc[n, 'label']
                pose=objects_predictions.poses[n]
                render_box=objects_predictions.boxes_rend[n]
                render_box_crop=objects_predictions.boxes_crop[n]
                if label in self.objects:
                    self.objects[label].update(pose, render_box)
                    #self.objects[label].mesh.apply_transform(tm.transformations.random_rotation_matrix())
                else:
                    self.objects[label] = RigidObject(img_resolution= self.cam_data['res'], label=label,
                        pose = pose,
                        image_id=objects_predictions.infos.loc[n, 'batch_im_id'],
                        score=objects_predictions.infos['score'][n],
                        render_box=render_box)
                    
                    self.mesh_scene.add_geometry(self.objects[label].mesh, geom_name=label)
        self.update_objects_fps()
        self.clean_objects()

    def update_hands_fps(self):
        now = time.time()
        self.elapsed_hands= now - self.time_hands 
        for hand in self.hands.values():
            hand.set_elapsed(self.elapsed_hands)
        self.fps_hands = 1 / self.elapsed_hands
        self.time_hands = now

    def update_scene_fps(self):
        now = time.time()
        self.elapsed_scene= now - self.time_scene
        self.fps_scene = 1 / self.elapsed_scene
        self.time_scene = now

    def update_objects_fps(self):
        now = time.time()
        self.elapsed_objects = now - self.time_objects
        self.fps_objects = 1 /self.elapsed_objects
        self.time_objects = now
    
    def update_detections_fps(self):
        now = time.time()
        self.elapsed_detections = now - self.time_detections
        self.fps_detections = 1 /self.elapsed_detections
        self.time_detections = now


    def clean_objects(self):
        todel=[]
        for label in self.objects.keys():
            if self.objects[label].nb_updates <=0:
                todel.append(label)
            self.objects[label].nb_updates-=1
        for key in todel:
            del self.objects[key]
            print('object '+key+' forgotten')

    def compute_distances(self):
        for obj in self.objects.values():
            for hand in self.hands.values():
                obj.distance_to(hand)


    def render(self, img):
        self.compute_distances()
        self.update_scene_fps()
        #if self.rendering_options['draw_hands']:
        #    self.hand_detector.draw_landmarks(img)
        #if self.rendering_options['write_hands_pos']:
        #    self.hand_detector.write_pos(img)
        for hand in self.hands.values():
            hand.render(img)
        for obj in self.objects.values():
            obj.render(img)
        if self.rendering_options['write_fps']:
            self.write_fps(img)

        cv2.imshow(self.name, img)

        return cv2.waitKey(5) & 0xFF == 27

    def write_fps(self, img):
        cv2.putText(img,'fps scene: {:.0f}'.format(self.fps_scene),(10,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
        cv2.putText(img,'fps hands: {:.0f}'.format(self.fps_hands),(10,65),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
        cv2.putText(img,'fps detections: {:.0f}'.format(self.fps_detections),(10,90),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
        cv2.putText(img,'fps objects: {:.0f}'.format(self.fps_objects),(10,115),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

    def stop(self):
        self.t_scene.join()
        # self.mesh_scene.close()

class Position:
    def __init__(self, vect) -> None:
        vect[1] = -vect[1]
        self.x = vect[0]
        self.y = vect[1]
        self.z = vect[2]
        self.v = vect

    def __str__(self):
        return str(self.v)
    
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
    def __init__(self, tensor) -> None:
        self.min_cutoffs = {'position' : 0.000001, 'orientation' : 0.001}
        self.betas = {'position' : 5, 'orientation' : 0.5}
        self.derivate_cutoff = {'position' : 1, 'orientation' : 1}
        attributes = ('position', 'orientation')
        self.filters={}
        for key in attributes:
            self.filters[key] = LandmarksSmoothingFilter(min_cutoff=self.min_cutoffs[key], beta=self.betas[key], derivate_cutoff=self.derivate_cutoff[key], disable_value_scaling=True)
        self.update(tensor)

    def update(self, tensor):
        self.mat = tensor.cpu().numpy()
        self.raw_position = Position(self.mat[:3,3])
        self.raw_orientation = Orientation(self.mat[:3,:3])
        # self.position = Position(self.filters['position'].apply(self.mat[:3,3]))
        self.position = self.raw_position
        # print(self.raw_position)
        # print(self.position)
        self.orientation = Orientation(self.filters['orientation'].apply(self.mat[:3,:3]))
    
    def __str__(self):
        out = 'position : ' + str(self.position) + ' -- orientation : ' + str(self.orientation)
        return out
    
class Bbox:

    def __init__(self, img_resolution, label, tensor) -> None:
        self.filter = LandmarksSmoothingFilter(min_cutoff=0.001, beta=0.5, derivate_cutoff=1, disable_value_scaling=True)
        
        self.color = (255, 0, 0)
        self.thickness = 2
        self.label=label
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

class Entity:
    def __init__(self) -> None:
        self.visible = True
        self.lost = False
        self.invisible_time = 0
        self.max_invisible_time = 0.3
        self.raw = {}
        self.refined = {}
        self.derived = {}
        self.derived_refined = {}
        self.refinable_keys = {}
        self.filters={}
        self.derived_filters={}
        self.elapsed=0
        self.velocity = 0
        self.normed_velocity = self.velocity
        self.new= True

    def setvisible(self, bool):
        if not bool:
            self.invisible_time += self.elapsed
        elif self.visible:
            self.invisible_time = 0
        self.visible = bool
        self.lost = not (self.invisible_time < self.max_invisible_time)

    def set_elapsed(self, elapsed):
        self.elapsed = elapsed

    def pose(self):
        return self.pose
    
    def position(self):
        return self.pose.position.v
    
    def velocity(self):
        return self.velocity

class RigidObject(Entity):
    def __init__(self, img_resolution = (1280, 720), image_id = None, label = None, pose=None, score = None, render_box=None) -> None:
        super().__init__()
        self.image_id = image_id
        self.label = label
        self.color = (255, 0, 0)
        self.pose = Pose(pose)
        self.score = score
        self.render_box = Bbox(img_resolution, label, render_box)
        self.distances={}
        print('object '+self.label+ ' discovered')
        self.nb_updates = 10
        self.target_metric = 0
        self.appearing_radius = 0.2
        self.load_mesh()

    def __str__(self):
        out = 'label: ' + str(self.label) + '\n pose: {' +str(self.pose)+'} \n nb_updates: '+str(self.nb_updates)
        return out
    
    def update(self, new_pose, new_render_box):
        self.pose.update(new_pose)
        self.render_box.update_coordinates(new_render_box)
        if self.nb_updates <=15:
            self.nb_updates+=2

    def load_mesh(self):
        try :
            self.mesh = tm.load_mesh(_TLESS_MESH_PATH+self.label+'.ply')
            print('MESH LOADED : ' +self.label+'.ply')
        except:
            self.mesh = None
            print('MESH LOADING FAILED')

    def write(self, img):
        text = self.label 
        x = self.render_box.corner1[0]
        y = self.render_box.corner1[1]-70
        dy = 15
        cv2.rectangle(img, (x,y-10), (self.render_box.corner2[0],self.render_box.corner1[1]), (200,200,200), -1)
        cv2.putText(img, text , (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color)    
        y+=dy
        text = 'x : ' + str(int(self.pose.position.x*100))+' cm'
        cv2.putText(img, text , (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, self.color)  
        y+=dy
        text = 'y : ' + str(int(self.pose.position.y*100))+' cm'
        cv2.putText(img, text , (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, self.color)  
        y+=dy
        text = 'z : ' + str(int(self.pose.position.z*100))+' cm'
        cv2.putText(img, text , (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, self.color)  
        y+=dy
        text = 'score : ' + str(int(self.score*100))+' cm'
        cv2.putText(img, text , (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, self.color)  

    def write_dist(self, img):
        text = self.label
        x = self.render_box.corner1[0]
        y = self.render_box.corner2[1]
        dy = 20
        cv2.rectangle(img, (x,y), (self.render_box.corner2[0]+30,y+50), (200,200,200), -1)
        for k, d in self.distances.items():
            cv2.putText(img, 'd-'+k+' : '+str(int(d)) +' cm' , (x,y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.color)    
            y+=dy

    def render(self, img, bbox = True, txt=True, dist=True, overlay=False):
        if txt:
            self.write(img)
        if bbox:
            self.render_box.draw(img)
        if dist:
            self.write_dist(img)
        if overlay:
            pass
    
    def distance_to(self, hand):
        self.distances[hand.label] = np.linalg.norm(100*self.pose.position.v - 0.1*hand.xyz)

    def is_targeted_by(self, hand):
        hpos = hand.position()
        opos = self.position()
        print('opos: ' + str(opos))
        print('hpos: ' + str(hpos))
        # v = hand.normed_velocity
        v = np.array([0,0,1])
        proj_point = hpos + np.dot(opos- hpos, v) * v #projection of the hand on the plan orthogonal 
        #to the directionnal vector and the passing by the object
        cone_appearing_radius = np.tan(hand.cone_angle) * np.linalg.norm(proj_point - hpos) 
        #self.target_metric = np.linalg.norm(opos - proj_point) / (cone_appearing_radius + self.appearing_radius)
        self.target_metric = np.linalg.norm(opos - proj_point) / (self.appearing_radius)
        print(self.target_metric)
        return self.target_metric


class Hand(Entity):
    def __init__(self, depthai_hand, detector) -> None:
        self.__dict__= depthai_hand.__dict__
        super().__init__()
        self.visible = True
        self.invisible_time = 0
        self.max_invisible_time = 0.3
        self.renderer = HandTrackerRenderer(detector)
        self.raw = {}
        self.refined = {}
        self.derived = {}
        self.derived_refined = {}
        self.refinable_keys = {}
        self.filters={}
        self.derived_filters={}
        self.elapsed=0
        self.velocity = 0*self.xyz
        self.scalar_velocity = 0
        self.normed_velocity = self.velocity
        self.new= True
        self.min_cutoffs = {'landmarks' : 0.001, 'world_landmarks' : 0.001, 'xyz': 0.00001}
        self.betas = {'landmarks' : 0.5, 'world_landmarks' : 0.5, 'xyz': 0.01}
        self.derivate_cutoff = {'landmarks' : 1, 'world_landmarks' : 1, 'xyz': 1}

        self.derivative_min_cutoffs = {'landmarks' : 0.001, 'world_landmarks' : 0.001, 'xyz': 0.00001}
        self.derivative_betas = {'landmarks' : 0.5, 'world_landmarks' : 0.5, 'xyz': 0.01}
        self.derivative_derivate_cutoff = {'landmarks' : 1, 'world_landmarks' : 1, 'xyz': 1}
        self.cone_angle = np.pi/8
        self.position = None
        self.mesh_position= None

        # labs = ['landmarks', 'world_landmarks', 'xyz']
        labs = ['landmarks', 'xyz']
        for key in labs:
            self.refinable_keys[key] = range(len(depthai_hand.__dict__[key]))
            self.raw[key]=depthai_hand.__dict__[key]
            self.refined[key]=depthai_hand.__dict__[key]
            self.derived[key]=0*depthai_hand.__dict__[key]
            self.derived_refined[key]=0*depthai_hand.__dict__[key]
            self.filters[key] = LandmarksSmoothingFilter(min_cutoff=self.min_cutoffs[key], beta=self.betas[key], derivate_cutoff=self.derivate_cutoff[key], disable_value_scaling=True)
            self.derived_filters[key] = LandmarksSmoothingFilter(min_cutoff=self.derivative_min_cutoffs[key], beta=self.derivative_betas[key], derivate_cutoff=self.derivative_derivate_cutoff[key], disable_value_scaling=True)

        self.position = Position(self.xyz*np.array([1,-1,1])/1000)
        self.mesh_position = Position(self.xyz*np.array([1,1,1]))
        self.mesh_origin = tm.primitives.Sphere(radius = 30)
        if self.label == 'right':
            self.mesh_origin.visual.face_colors = [255,0,0,255]
        else:  
            self.mesh_origin.visual.face_colors = [0,0,100,255]

    def update(self, depthai_hand):
        self.new = False
        for key, val in depthai_hand.__dict__.items():
            self.__dict__[key] = val
        for key, key_indexes in self.refinable_keys.items():
            #self.derived[key]=(self.raw[key] - depthai_hand.__dict__[key])/self.elapsed 
            self.raw[key]=depthai_hand.__dict__[key]

    def propagate1(self):
        if not self.visible:
            for key, key_indexes in self.refinable_keys.items():
                self.refined[key]=[np.round(self.raw[key][i] + self.elapsed*self.derived[key][i]).astype(int) for i in key_indexes]
        else: 
            self.refined = self.raw
        for key in self.refinable_keys:
            self.__dict__[key] = self.refined[key]
        self.raw = self.refined
        pass

    def propagate(self):
        if not self.lost:
            self.position = Position(self.xyz*np.array([1,-1,1])/1000)
            self.mesh_position = Position(self.xyz*np.array([-1,-1,1]))
            if not self.new:
                for key, key_indexes in self.refinable_keys.items():
                    new_refined=np.rint(self.filters[key].apply(self.raw[key])).astype(int)
                    self.derived[key]=(new_refined - self.refined[key])/self.elapsed 
                    self.refined[key] = new_refined
                    self.derived_refined[key]=np.rint(self.derived_filters[key].apply(self.derived[key])).astype(int)
                for key in self.refinable_keys:
                    self.__dict__[key] = self.refined[key]
                self.velocity = self.derived_refined['xyz']
                self.scalar_velocity = np.linalg.norm(self.velocity)
                if self.scalar_velocity != 0:
                    self.normed_velocity = self.velocity/self.scalar_velocity
                else:
                    self.normed_velocity = np.array([0,0,0])
        else:
            self.velocity =  np.array([0,0,0])
            self.normed_velocity =  np.array([0,0,0])

    def render(self, img):
        if not self.lost:
            self.renderer.frame=img
            self.renderer.draw_hand(self)



def rotation_from_vectors(v1, v2):

    # Calcul de l'axe et de l'angle de rotation
    axis = np.cross(v1, v2)
    if np.linalg.norm(v1) * np.linalg.norm(v2) == 0:
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    else:
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    # # Création de la rotation à partir de l'axe et de l'angle
    # r = R.from_rotvec(angle * axis)

    # # Obtention du quaternion correspondant à la rotation
    # q = r.as_quat()
    q = tm.transformations.rotation_matrix(angle, axis)
    return q

def spline(x0,x1, y0, y1):

    x=np.array([x0,x1])
    y=np.array([y0, y1])
    cs = CubicSpline(x,y, bc_type='clamped')
    def eval(x):
        if x<=x0:
            return y0
        elif x>=x1:
            return y1
        else:
            return cs(x)
    return eval