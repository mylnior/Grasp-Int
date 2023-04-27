import trimesh as tm
import numpy as np


def arrow(radius=0.03,
             height=0.2, cone_radius=0.05, cone_height=0.1,
             sections=None,
             segment=None,
             transform=None,
             **kwargs):
    """
    Create a mesh of a cylinder along Z centered at the origin.

    Parameters
    ----------
    radius : float
      The radius of the cylinder
    height : float or None
      The height of the cylinder
    sections : int or None
      How many pie wedges should the cylinder have
    segment : (2, 3) float
      Endpoints of axis, overrides transform and height
    transform : (4, 4) float
      Transform to apply
    **kwargs:
        passed to Trimesh to create cylinder

    Returns
    ----------
    cylinder: trimesh.Trimesh
      Resulting mesh of a cylinder
    """

    if segment is not None:
        # override transform and height with the segment
        transform, height = tm.creation._segment_to_cylinder(segment=segment)

    if height is None:
        raise ValueError('either `height` or `segment` must be passed!')

    half = abs(float(height+cone_height)) / 2.0
    # create a profile to revolve
    linestring = [[0, -half],
                  [radius, -half],
                  [radius, half-cone_height],
                  [cone_radius, half-cone_height],
                  [0, half]]
    if 'metadata' not in kwargs:
        kwargs['metadata'] = dict()
    kwargs['metadata'].update(
        {'shape': 'cylinder',
         'height': height,
         'radius': radius})
    # generate cylinder through simple revolution
    return tm.creation.revolve(linestring=linestring,
                   sections=sections,
                   transform=transform,
                   **kwargs)


pill = tm.creation.capsule(radius = 0.5)
palmar_arrows = []
x=tm.transformations.translation_matrix(np.array([0,-0.4,0]))@tm.transformations.euler_matrix(-np.pi/2,np.pi/8,0)
pill.apply_transform(x)
for i in range(4):
    for j in range(6):
        arrow1 = arrow()
        rot=tm.transformations.euler_matrix(np.pi/2,0,0)
        rot2=tm.transformations.euler_matrix(0,0,i*np.pi/2)
        if j !=5:
            trans=tm.transformations.translation_matrix(np.array([0,0.65,0.25*j]))
        else:
            trans=tm.transformations.translation_matrix(np.array([0,0.58,0.25*j]))

        arrow1.apply_transform(x@rot2@trans@rot)
        arrow1.visual.face_colors = [0,255,0,255]
        palmar_arrows.append(arrow1)
pinch_arrows = []
for i in range(4):
    for j in range(2):
        arrow1 = arrow()
        rot=tm.transformations.euler_matrix(0,np.pi,0)
        if j !=0:
            rot2=tm.transformations.euler_matrix(0,0,i*np.pi/2)
            trans=tm.transformations.translation_matrix(np.array([0,0.43,1.4]))
        else:
            rot2=tm.transformations.euler_matrix(0,0,i*np.pi/2+np.pi/4)
            trans=tm.transformations.translation_matrix(np.array([0,0.3,1.55]))

        arrow1.apply_transform(x@rot2@trans@rot)
        arrow1.visual.face_colors = [255,0,255,255]
        pinch_arrows.append(arrow1)
scene = tm.Scene([pill]+palmar_arrows+pinch_arrows)
scene.show()