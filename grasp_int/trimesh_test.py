"""
view_callback.py
------------------

Show how to pass a callback to the scene viewer for
easy visualizations.
"""

import time
import trimesh
import numpy as np
import pickle

def sinwave(scene):
    """
    A callback passed to a scene viewer which will update
    transforms in the viewer periodically.

    Parameters
    -------------
    scene : trimesh.Scene
      Scene containing geometry

    """
    # create an empty homogeneous transformation
    matrix = np.eye(4)
    # set Y as cos of time
    matrix[1][3] = np.cos(time.time()) * 2
    # set Z as sin of time
    matrix[2][3] = np.sin(time.time()) * 3

    # take one of the two spheres arbitrarily
    node = s.graph.nodes_geometry[0]
    # apply the transform to the node
    # scene.graph.update(node, matrix=matrix)
    prev = np.array([10,0,0])
    s.graph.update('first', matrix = trimesh.transformations.compose_matrix(translate = prev ), geometry = 'first')
    prev = np.array([10,10,0])
    s.graph.update('second', matrix = trimesh.transformations.compose_matrix(translate = prev ), geometry = 'second')


if __name__ == '__main__':
    # create some spheres
    a = trimesh.primitives.Sphere()
    b = trimesh.primitives.Sphere()

    # set some colors for the balls
    a.visual.face_colors = [255, 0, 0, 255]
    b.visual.face_colors = [0, 0, 100, 255]

    # create a scene with the two balls
    s = trimesh.Scene()
    s.add_geometry(a, geom_name='first')
    s.add_geometry(b, geom_name='second')
    # s.show()
    # with open('my_graph', 'rb') as f:
    #   g = pickle.load(f)
    #   g.show()
    # s.graph.show()# open the scene viewer and move a ball around
    s.show(callback=sinwave)
