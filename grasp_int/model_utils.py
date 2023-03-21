import yaml
from pathlib import Path
import numpy as np
import torch

from cosypose.config import EXP_DIR, LOCAL_DATA_DIR
from cosypose.datasets.datasets_cfg import make_object_dataset

from cosypose.rendering.bullet_batch_renderer import BulletBatchRenderer

# Rigid
from cosypose.integrated.pose_predictor import CoarseRefinePosePredictor as RigidPosePredictor
from cosypose.training.pose_models_cfg import create_model_pose as create_model_rigid
from cosypose.lib3d.rigid_mesh_database import MeshDataBase as RigidMeshDataBase

# Detector
from cosypose.training.detector_models_cfg import create_model_detector
from cosypose.integrated.detector import Detector


def load_torch_model(run_id, *args, model_type='rigid'):
    run_dir = EXP_DIR / run_id
    cfg = yaml.load((run_dir / 'config.yaml').read_text(), Loader=yaml.UnsafeLoader)
    if model_type == 'rigid':
        create_model_fn = create_model_rigid
    elif model_type == 'detector':
        create_model_fn = create_model_detector
        assert len(args) == 0
        args = [len(cfg.label_to_category_id)]
    else:
        raise ValueError('Unknown model type', model_type)
    model = create_model_fn(cfg, *args)
    ckpt = torch.load(run_dir / 'checkpoint.pth.tar')
    ckpt = ckpt['state_dict']
    model.load_state_dict(ckpt)
    model = model.cuda().eval()
    model.cfg = cfg
    model.config = cfg
    model_type = 'rigid' # if 'ycbv' in coarse_run_id else 'articulated'
    return model


def load_pose_predictor(coarse_run_id, refiner_run_id, n_workers=1, preload_cache=False):
    run_dir = EXP_DIR
    coarse_run_dir = EXP_DIR / coarse_run_id
    coarse_cfg = yaml.load((coarse_run_dir / 'config.yaml').read_text(), Loader=yaml.UnsafeLoader)

    renderer = BulletBatchRenderer(coarse_cfg.urdf_ds_name, preload_cache=preload_cache, n_workers=n_workers)
    object_ds = make_object_dataset(coarse_cfg.object_ds_name)
    mesh_db = RigidMeshDataBase.from_object_ds(object_ds).batched().cuda()

    coarse_model = load_torch_model(coarse_run_id, renderer, mesh_db, model_type='rigid')
    refiner_model = load_torch_model(refiner_run_id, renderer, mesh_db, model_type='rigid')

    return RigidPosePredictor(coarse_model, refiner_model)


def load_detector(detector_run_id):
    detector_model = load_torch_model(detector_run_id, model_type='detector')
    model = Detector(detector_model)
    return model


# def load_friendly_names(
#         friendly_names_path=LOCAL_DATA_DIR / 'bop_datasets/ycbv/ycbv_friendly_names.txt'):
#     friendly_names = Path(friendly_names_path).read_text()
#     d = dict()
#     for l in friendly_names.split('\n')[:-1]:
#         n = int(l.split(' ')[0])
#         v = l.split(' ')[1]
#         k = f'obj_{n:06d}'
#         v = v[4:]
#         d[k] = v
#     return d
