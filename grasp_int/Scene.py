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
import pyfqmr
from scipy.spatial.transform import Rotation as R

from grasp_int.depthai_hand_tracker.HandTrackerRenderer import \
    HandTrackerRenderer
from grasp_int.Filters import LandmarksSmoothingFilter
from grasp_int.utils import *

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
        self.hand_detector = hand_detector
        self.object_detector = object_detector
        if device is None:
            self.cam_data['res'] = self.hand_detector.get_res()
            self.cam_data['K'] = self.hand_detector.matrix
        else:
            self.cam_data['res'] = device.img_resolution
            self.cam_data['K'] = device.matrix
        #self.cam['RT'] = device.extrinsics
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
        self.new_object_meshes = []
        self.objects_collider = tm.collision.CollisionManager()
        if self.draw_mesh:
            self.define_mesh_scene()
        self.velocity_cone_mode = 'rays'
        # self.velocity_cone_mode = 'cone'

    def __str__(self) -> str:
        s = 'SCENE \n OBJECTS :\n'
        objs = self.objects.copy().items()
        for obj in self.objs:
            s+=str(obj)+'\n'
        return s
    
    def display_meshes(self):
        self.mesh_scene.show(callback=self.update_meshes,callback_period=1.0/30.0,line_settings={'point_size':20})

    def update_hands_meshes(self, scene):
        for i in range(len(self.new_hand_meshes)):
            new = self.new_hand_meshes.pop(0)
            scene.add_geometry(new['mesh'], geom_name = new['name'])
        
        hands = self.hands.copy().items()
        for label, hand in hands:
            hand.mesh_pos = tm.transformations.compose_matrix(translate = hand.mesh_position.v)
            scene.graph.update(label,matrix = hand.mesh_pos , geometry = label)
            if self.velocity_cone_mode == 'cone':
                vdir = hand.normed_velocity
                vdir = np.array([0,0.0001,-1])                
                zaxis = np.array([0,0,1])
                vel_rot_mat = rotation_from_vectors(zaxis,vdir)
                cone_len = self.cone_lenghts_spline(hand.scalar_velocity)
                cone_len = 2000
                cone_diam = self.cone_diam_spline(hand.scalar_velocity)
                x_reflection_matrix = tm.transformations.reflection_matrix(np.array([0,0,0]), np.array([1,0,0]))
                z_reflection_matrix = tm.transformations.reflection_matrix(np.array([0,0,cone_len/2]), np.array([0,0,1]))
                tot_mat = hand.mesh_pos @ x_reflection_matrix @ vel_rot_mat @ z_reflection_matrix
                # tot_mat = hand.mesh_pos 
                hand.velocity_cone = tm.creation.cone(cone_diam, cone_len)
                scene.delete_geometry(label+'vel_cone')
                scene.add_geometry(hand.velocity_cone, geom_name = label+'vel_cone', transform = tot_mat)
                self.check_collisions(hand.velocity_cone, tot_mat)
            else:
                scene.delete_geometry(hand.label+'vel_cone')
                hand.make_rays()
                scene.add_geometry(hand.ray_visualize, geom_name=hand.label+'vel_cone')
                #self.check_rays(scene,hand)

    def check_all_targets(self, scene):
        targets = {}
        hands = self.hands.copy()
        objs = self.objects.copy()
        for hlabel, hand in hands.items(): 
            for olabel, obj in objs.items():
                if olabel in scene.geometry:
                    mesh = scene.geometry[olabel]
                    mesh_frame_locations, world_locations, _, _ = hand.check_rays(obj, mesh)
                    print(mesh_frame_locations)
                    scene.delete_geometry(hand.label+obj.label+'ray_impacts')
                    hand.set_object_cone_impact(obj, mesh_frame_locations, world_locations)
                    if len(mesh_frame_locations)>0:
                        impacts = tm.points.PointCloud(mesh_frame_locations, colors=hand.color)
                        # scene.add_geometry(impacts, geom_name=hand.label+'ray_impacts')
                        scene.add_geometry(impacts, geom_name=hand.label+obj.label+'ray_impacts',transform = obj.mesh_transform)
                else:
                    print('Mesh '+olabel+' has not been loaded yet')
            targets[hlabel]= hand.fetch_targets()
            
        for olabel in objs:
            target_info = (False, None, None)
            for hlabel in hands:
                if olabel in targets[hlabel]:
                    target_info=(True, self.hands[hlabel], targets[hlabel][olabel])
                self.objects[olabel].set_target_info(target_info)


    def update_object_meshes(self, scene):
        for i in range(len(self.new_object_meshes)):
            new = self.new_object_meshes.pop(0)
            scene.add_geometry(new['mesh'], geom_name = new['name'])
            # self.objects_collider.add_object(new['name'], new['mesh'])            
            print('add here')
            print(scene.geometry)
            print('add there')

        objs = self.objects.copy().items()
        for label, obj in objs:
            obj.mesh_pos = obj.pose.position.v*np.array([-1,1,1])
            mesh_orient_quat = [obj.pose.orientation.q[i] for i in range(4)]
            mesh_orient_angles = obj.pose.orientation.v*np.array([-1,-1,-1])+np.pi*np.array([1  ,1,0])
            mesh_orient_angles = obj.pose.orientation.v*np.array([1,1,1])+np.pi*np.array([0 ,0,1])
            x_reflection_matrix = tm.transformations.reflection_matrix(np.array([0,0,0]), np.array([1,0,0]))
            #mesh_transform = tm.transformations.translation_matrix(mesh_pos)  @ tm.transformations.quaternion_matrix(mesh_orient_quat)
            rot_mat = tm.transformations.euler_matrix(mesh_orient_angles[0],mesh_orient_angles[1],mesh_orient_angles[2])
            mesh_transform = tm.transformations.translation_matrix(obj.mesh_pos) @ rot_mat
            # scene.geometry[label].apply_transform(mesh_transform)
            # self.objects_collider.set_transform(label,mesh_transform)           
            scene.graph.update(label, matrix=mesh_transform)
            obj.set_mesh_transform(mesh_transform)

    def check_collisions(self,cone, tf):
        collision, name, data = self.objects_collider.in_collision_single(cone, transform=tf, return_data=True, return_names=True)
        if collision:
            print(str(name)+' : ')
            for dat in data:
                print('normal : '+str(dat.normal))
                print('point : '+str(dat.point))
                print('depth : '+str(dat.depth))



    def update_meshes(self, scene):
        self.update_hands_meshes(scene)
        self.update_object_meshes(scene)
        self.check_all_targets(scene)

    def define_mesh_scene(self):
        self.mesh_scene= tm.Scene()
        self.mesh_scene.camera.resolution = self.cam_data['res']
        self.mesh_scene.camera.focal= (self.cam_data['K'][0,0], self.cam_data['K'][1,1])
        self.mesh_scene.camera.z_far = 3000
        print('self.cam_data', self.cam_data)
        X =  0
        Y =  -250
        Z = 1000
        self.test_cone = tm.creation.cone(50,500)
        # self.mesh_scene.add_geometry(self.test_cone, geom_name='test_cone')
        self.mesh_scene.camera_transform = tm.transformations.rotation_matrix(np.pi, np.array([0,1,0]), np.array([0,0,0]))
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

        # lambda_length = 


        self.mesh_scene.add_geometry(plane, geom_name='plane')
        # self.mesh_scene.add_geometry(cam, geom_name='macamira')
        # base_origin = tm.primitives.Sphere(radius = 50)
        # base_origin.visual.face_colors = [0,255,0,255]
        # self.mesh_scene.add_geometry(base_origin, geom_name='base_origin')
        # cyl = tm.creation.cylinder(50,500)
        # self.objects_collider.add_object('cyl', cyl)     
        # self.mesh_scene.add_geometry(cyl, geom_name='cyl')
        self.t_scene = threading.Thread(target=self.display_meshes)
        self.t_scene.start()

    def evaluate_grasping_intention(self):
        hands = self.hands.copy().values
        objs = self.objects.values()
        for obj in objs():
            for hand in hands:
                if hand.label=='right':
                    obj.is_targeted_by(hand)

    def propagate_hands(self):
        hands = self.hands.copy().values()
        for hand in hands:
            hand.propagate()

    def clean_hands(self, newhands):
        hands_label = [hand.label for hand in newhands]
        hands = self.hands.copy()
        for label in hands:
            self.hands[label].setvisible(label in hands_label)

    def update_hands(self, hands_predictions):
        self.update_hands_fps()
        # print('hands_predictions')
        # print(hands_predictions)
        hands = self.hands.copy()
        for hand_pred in hands_predictions:
            if hand_pred.label not in hands:
                self.new_hand(hand_pred)
            else : 
                self.hands[hand_pred.label].update(hand_pred)
        self.clean_hands(hands_predictions)
        self.propagate_hands()
        # self.evaluate_grasping_intention()
    
    def new_hand(self, pred):
        self.hands[pred.label] = Hand(pred, self.hand_detector)
        self.new_hand_meshes.append({'mesh' : self.hands[pred.label].mesh_origin, 'name': pred.label})

    def new_object(self,pred, label, n):
        pose=pred.poses[n]
        render_box=pred.boxes_rend[n]
        self.objects[label] = RigidObject(img_resolution= self.cam_data['res'], label=label,
            pose = pose,
            image_id=pred.infos.loc[n, 'batch_im_id'],
            score=pred.infos['score'][n],
            render_box=render_box)
        self.new_object_meshes.append({'mesh' : self.objects[label].mesh, 'name': label})

        
    def update_objects(self, objects_predictions):
        hands = self.hands.copy()
        objs = self.objects.copy()
        if objects_predictions is not None:
            for n in range(len(objects_predictions)):
                label=objects_predictions.infos.loc[n, 'label']
                #render_box_crop=objects_predictions.boxes_crop[n]
                if label in objs:
                    pose=objects_predictions.poses[n]
                    render_box=objects_predictions.boxes_rend[n]
                    self.objects[label].update(pose, render_box)
                    #self.objects[label].mesh.apply_transform(tm.transformations.random_rotation_matrix())
                else:                    
                    self.new_object(objects_predictions,label,n)
        self.update_objects_fps()
        self.clean_objects()

    def update_hands_fps(self):
        now = time.time()
        self.elapsed_hands= now - self.time_hands 
        hands = self.hands.copy().values()
        for hand in hands:
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
        objs = self.objects.copy()
        for label in objs.keys():
            if self.objects[label].nb_updates <=0:
                todel.append(label)
            self.objects[label].nb_updates-=1
        for key in todel:
            del self.objects[key]
            print('object '+key+' forgotten')

    def compute_distances(self):
        hands = self.hands.copy().values()
        objs = self.objects.copy().values()
        for obj in objs:
            for hand in hands:
                obj.distance_to(hand)


    def render(self, img):
        self.compute_distances()
        self.update_scene_fps()
        hands = self.hands.copy().values()
        objs = self.objects.copy().values()
        #if self.rendering_options['draw_hands']:
        #    self.hand_detector.draw_landmarks(img)
        #if self.rendering_options['write_hands_pos']:
        #    self.hand_detector.write_pos(img)
        for hand in hands:
            hand.render(img)
        for obj in objs:
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

    def check_rays(self,scene,hand):
        objs = self.objects.copy().items()
        for label, obj in objs:
            vdir = hand.normed_velocity
            n_layers = 5
            # vecteur aléatoire
            random_vector = np.random.rand(len(vdir))

            # projection
            projection = np.dot(random_vector, vdir)

            # vecteur orthogonal
            orthogonal_vector = random_vector - projection * vdir

            # vecteur unitaire orthogonal

            #ray_directions = np.vstack([ vdir*cone_len + (-cone_diam/2+i*cone_diam/(n_layers-1)) * orthogonal_unit_vector for i in range(n_layers) for j in range(n_layers) ])    
            ray_directions_list = [ np.array([0,0,1000] + np.array([-200+i*100,-200+j*100,0]) ) for i in range(5) for j in range(5) ]
            inv_trans = np.linalg.inv(obj.mesh_transform)
            translation = inv_trans[:3,3]
            rot = inv_trans[:3,:3]

            ray_origins = np.vstack([hand.mesh_position.v for i in range(n_layers) for j in range(n_layers) ])
            ray_directions = np.vstack(ray_directions_list)
            ray_visualize = tm.load_path(np.hstack((
                ray_origins,
                ray_origins + ray_directions)).reshape(-1, 2, 3))
            extended = np.hstack((hand.mesh_position.v,1))
            ray_origins_obj_frame = np.vstack([ (inv_trans@extended)[:3] for i in range(n_layers) for j in range(n_layers) ])
            ray_directions_obj_frame = np.vstack([rot @ ray_dir for ray_dir in ray_directions_list])    
            ray_visualize_obj_frame = tm.load_path(np.hstack((
                ray_origins_obj_frame,
                ray_origins_obj_frame + ray_directions_obj_frame)).reshape(-1, 2, 3))
            # mesh = tm.primitives.Cylinder(radius=60, height=1000)
            # scene.delete_geometry('cyl')
            # scene.add_geometry(mesh, geom_name='cyl')

            scene.delete_geometry(hand.label+'vel_cone')
            scene.add_geometry(ray_visualize, geom_name=hand.label+'vel_cone')
            # scene.delete_geometry(hand.label+'vel_cone_of')
            # scene.add_geometry(ray_visualize_obj_frame, geom_name=hand.label+'vel_cone_of')
            if label in scene.geometry:
                oobj = scene.geometry[label]
                locations, index_ray, index_tri = oobj.ray.intersects_location(ray_origins=ray_origins_obj_frame,
                                                                                ray_directions=ray_directions_obj_frame,
                                                                                multiple_hits=False)
                print(locations)
                scene.delete_geometry(hand.label+'ray_impacts')
                if len(locations)>0:
                    impacts = tm.points.PointCloud(locations, colors=hand.color)
                    # scene.add_geometry(impacts, geom_name=hand.label+'ray_impacts')
                    scene.add_geometry(impacts, geom_name=hand.label+'ray_impacts',transform = obj.mesh_transform)
            else:
                print('Mesh '+label+' has not been loaded yet')

class Entity:
    def __init__(self) -> None:
        self.visible = True
        self.lost = False
        self.invisible_time = 0
        self.max_invisible_time = 0.1
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
        self.default_color = (0, 255, 0)
        self.pose = Pose(pose, position_factor=1000)
        self.score = score
        self.render_box = Bbox(img_resolution, label, render_box)
        self.distances={}
        print('object '+self.label+ ' discovered')
        self.nb_updates = 10
        self.target_metric = 0
        self.appearing_radius = 0.2
        self.load_mesh()
        self.mesh_pos = np.array([0,0,0])
        self.mesh_transform = np.identity(4)
        self.is_targeted = False
        self.targeter = None
        self.target_info = None
        self.update_display()

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
            mesh_simplifier = pyfqmr.Simplify()
            mesh_simplifier.setMesh(self.mesh.vertices,self.mesh.faces)
            mesh_simplifier.simplify_mesh(target_count = 1000, aggressiveness=7, preserve_border=True, verbose=10)
            v, f, n = mesh_simplifier.getMesh()
            self.mesh = tm.Trimesh(vertices=v, faces=f, face_normals=n)
        except:
            self.mesh = None
            print('MESH LOADING FAILED')

    def write(self, img):
        text = self.label 
        x = self.render_box.corner1[0]
        y = self.render_box.corner1[1]-60
        dy = 15
        cv2.rectangle(img, (x,y-20), (self.render_box.corner2[0],self.render_box.corner1[1]), (200,200,200), -1)
        cv2.putText(img, text , (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color)    
        if self.is_targeted:
            text ='Trgt by : ' + self.targeter.label 
            y+=dy
            cv2.putText(img, text , (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color)  
            text = 'tbi : '+str(self.target_info.get_time_of_impact()) + 'ms'
            y+=dy
            cv2.putText(img, text , (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color)  
            text ='GRIP : '+self.target_info.get_grip()
            y+=dy
            cv2.putText(img, text , (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color)  
        # y+=dy
        # text = 'x : ' + str(int(self.pose.position.x*0.1))+' cm'
        # cv2.putText(img, text , (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, self.color)  
        # y+=dy
        # text = 'y : ' + str(int(self.pose.position.y*0.1))+' cm'
        # cv2.putText(img, text , (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, self.color)  
        # y+=dy
        # text = 'z : ' + str(int(self.pose.position.z*0.1))+' cm'
        # cv2.putText(img, text , (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, self.color)  
        # y+=dy
        # text = 'score : ' + str(int(self.score*100))+' cm'
        # cv2.putText(img, text , (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, self.color)  

    def write_dist(self, img):
        text = self.label
        x = self.render_box.corner1[0]
        y = self.render_box.corner2[1]
        dy = 20
        cv2.rectangle(img, (x,y), (self.render_box.corner2[0]+30,y+50), (200,200,200), -1)
        for k, d in self.distances.items():
            cv2.putText(img, 'd-'+k+' : '+str(int(d)) +' cm' , (x,y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.color)    
            y+=dy

    def render(self, img, bbox = True, txt=True, dist=False, overlay=False):
        self.update_display()
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

    def set_mesh_transform(self, tf):
        self.mesh_transform=tf
        self.inv_mesh_transform = np.linalg.inv(tf)

    def set_target_info(self, info):
        self.is_targeted = info[0]
        self.targeter = info[1]
        self.target_info = info[2]
        #if self.is_targeted:
        #    print(self.label, 'is targeted by ', self.targeter.label, 'in', self.target_info.get_time_of_impact())

    def update_display(self):
        if self.is_targeted:
            self.color = self.targeter.text_color
            thickness = 4
        else:
            self.color = self.default_color
            thickness = 2
        self.render_box.update_display(self.color, thickness)
        # print('DISPLAY UPDATED', time.time())

class Hand(Entity):
    def __init__(self, depthai_hand, detector) -> None:
        self.__dict__= depthai_hand.__dict__
        super().__init__()
        self.visible = True
        self.invisible_time = 0
        self.max_invisible_time = 0.3
        if detector.type != 'HybridOAKMediapipeDetector':
            self.renderer = HandTrackerRenderer(detector.device)
        else:
            self.draw = depthai_hand.draw
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

        self.derivative_min_cutoffs = {'landmarks' : 0.001, 'world_landmarks' : 0.001, 'xyz': 0.000000001}
        self.derivative_betas = {'landmarks' : 1.5, 'world_landmarks' : 0.5, 'xyz': 0.001}
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
        self.mesh_origin = tm.primitives.Sphere(radius = 20)
        if self.label == 'right':
            self.mesh_origin.visual.face_colors = self.color = [255,0,0,255]
            self.text_color = [0,0,255]
        else:  
            self.mesh_origin.visual.face_colors = self.color = [0,0,100,255]
            self.text_color=[100,0,0]
        self.define_velocity_cones()
        self.target_detector = TargetDetector(self.label)

    def define_velocity_cones(self,cone_max_length = 500, cone_min_length = 100, vmin=0., vmax=300, cone_max_diam = 100,cone_min_diam = 20, n_layers=5):

        self.n_layers = n_layers
        self.cone_lenghts_spline = spline(vmin,vmax, cone_min_length, cone_max_length)
        self.cone_diam_spline = spline(vmin,vmax, cone_min_diam, cone_max_diam)

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
            #self.mesh_position = Position(self.xyz*np.array([-1,-1,1]))
            if not self.new:
                for key, key_indexes in self.refinable_keys.items():
                    new_refined=np.rint(self.filters[key].apply(self.raw[key])).astype(int)
                    self.derived[key]=(new_refined - self.refined[key])/self.elapsed 
                    self.refined[key] = new_refined
                    self.derived_refined[key]=np.rint(self.derived_filters[key].apply(self.derived[key])).astype(int)
                for key in self.refinable_keys:
                    self.__dict__[key] = self.refined[key]
                self.velocity = self.derived_refined['xyz']*np.array([-1,1,1])
                self.scalar_velocity = np.linalg.norm(self.velocity)
                if self.scalar_velocity != 0:
                    self.normed_velocity = self.velocity/self.scalar_velocity
                else:
                    self.normed_velocity = np.array([0,0,0])            
            self.mesh_position = Position(self.refined['xyz']*np.array([-1,-1,1]))

        else:
            self.velocity =  np.array([0,0,0])
            self.normed_velocity =  np.array([0,0,0])

    def render(self, img):
        if not self.lost:
            if 'renderer' in self.__dict__:
                self.renderer_render(img)
            else:
                self.draw(img)

    def renderer_render(self,img):
        self.renderer.frame=img
        self.renderer.draw_hand(self)

    def make_rays(self): 
        vdir = self.normed_velocity
        cone_len = self.cone_lenghts_spline(self.scalar_velocity)
        cone_diam = self.cone_diam_spline(self.scalar_velocity)
        # vecteur aléatoire
        random_vector = np.random.rand(len(vdir))

        # projection
        projection = np.dot(random_vector, vdir)

        # vecteur orthogonal
        orthogonal_vector = random_vector - projection * vdir

        # vecteur unitaire orthogonal
        orthogonal_unit_vector = orthogonal_vector / np.linalg.norm(orthogonal_vector)
        orthogonal_unit_vector2 = np.cross(vdir, orthogonal_unit_vector)

        self.ray_directions_list = [ vdir*cone_len + (-cone_diam/2+i*cone_diam/(self.n_layers-1)) * orthogonal_unit_vector+ (-cone_diam/2+j*cone_diam/(self.n_layers-1)) * orthogonal_unit_vector2 for i in range(self.n_layers) for j in range(self.n_layers) ] 

        self.ray_origins = np.vstack([self.mesh_position.v for i in range(self.n_layers) for j in range(self.n_layers) ])
        self.ray_directions = np.vstack(self.ray_directions_list)
        self.ray_visualize = tm.load_path(np.hstack((
            self.ray_origins,
            self.ray_origins + self.ray_directions)).reshape(-1, 2, 3))     

    def check_rays(self,obj, mesh):
        # ray_directions_list = [ np.array([0,-1000,0]) + np.array([-200+i*100,0,-200+j*100] ) for i in range(5) for j in range(5) ]
        inv_trans = obj.inv_mesh_transform
        rot = inv_trans[:3,:3]
        ray_origins_obj_frame = np.vstack([ (inv_trans@self.mesh_position.ve)[:3] for i in range(self.n_layers) for j in range(self.n_layers) ])
        ray_directions_obj_frame = np.vstack([rot @ ray_dir for ray_dir in self.ray_directions_list])    
        try:
            mesh_frame_locations, index_ray, index_tri = mesh.ray.intersects_location(ray_origins=ray_origins_obj_frame,
                                                                            ray_directions=ray_directions_obj_frame,
                                                                            multiple_hits=False)
            world_locations = mesh_frame_locations + obj.mesh_pos
        except:
            mesh_frame_locations, world_locations, index_ray, index_tri = (), (), (), ()
        return mesh_frame_locations, world_locations, index_ray, index_tri

    
    def set_object_cone_impact(self, object, mesh_frame_locations, world_locations):
        self.target_detector.new_impacts(object, mesh_frame_locations, self.mesh_position, self.scalar_velocity)

    def fetch_targets(self):
        self.most_probable_target, targets = self.target_detector.get_most_probable_target()
        return targets
        
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