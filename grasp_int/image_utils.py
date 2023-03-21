import torch
import pandas as pd
import numpy as np
import cosypose.utils.tensor_collection as tc

def parse_encoding(encoding):
    if encoding.startswith('32F'):
        dtype = np.float32
    elif encoding.startswith('16U'):
        dtype = np.uint16
    elif encoding.startswith('16S'):
        dtype = np.int16
    else:
        return None
    return dtype, int(encoding[-1])

def make_cameras(list_dict_intrinsics):
    infos = []
    K = []
    for n, dict_intrinsics in enumerate(list_dict_intrinsics):
        cx, cy, fx, fy = [dict_intrinsics[k] for k in ('cx', 'cy', 'fx', 'fy')]
        this_K = torch.tensor([
            [fx, 0,  cx],
            [0,  fy, cy],
            [0,  0,  1.],
        ])
        K.append(this_K)
        infos.append(dict(batch_im_id=n, resolution=dict_intrinsics['resolution']))
    cameras = tc.PandasTensorCollection(
        infos=pd.DataFrame(infos),
        K=torch.stack(K)
    )
    return cameras
