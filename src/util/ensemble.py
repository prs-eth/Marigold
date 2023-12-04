# Test align depth images
# Author: Bingxin Ke
# Last modified: 2023-12-04

import os
import numpy as np
import torch

from scipy.optimize import minimize

def inter_distances(arrays):
    """
    To calculate the distance between each two depth maps.
    """
    distances = []
    for i, j in torch.combinations(torch.arange(arrays.shape[0])):
        arr1 = arrays[i:i+1]
        arr2 = arrays[j:j+1]
        distances.append(arr1 - arr2)
    if isinstance(arrays, torch.Tensor):
        dist = torch.concatenate(distances, dim=0)
    elif isinstance(arrays, np.ndarray):
        dist = np.concatenate(distances, axis=0)
    return dist


def ensemble_depths(input_images, regularizer_strength=0.02, max_iter=2, tol=1e-3, reduction='median', max_res=None, disp=False, device='cuda'):
    """ 
    To ensemble multiple affine-invariant depth images (up to scale and shift),
        by aligning estimating the sacle and shift
    """
    original_input = input_images.copy()
    n_img = input_images.shape[0]
    ori_shape = input_images.shape
            
    if max_res is not None:
        scale_factor = np.min(max_res / np.array(ori_shape[-2:]))
        if scale_factor < 1:
            downscaler = torch.nn.Upsample(scale_factor=scale_factor, mode='nearest')
            input_images = downscaler(torch.from_numpy(input_images)).numpy()

    # init guess
    _min = np.min(input_images.reshape((n_img, -1)), axis=1)
    _max = np.max(input_images.reshape((n_img, -1)), axis=1)
    s_init = 1.0 / (_max - _min).reshape((-1, 1, 1))
    t_init = (-1 * s_init.flatten() * _min.flatten()).reshape((-1, 1, 1))
    
    x = np.concatenate([s_init, t_init]).reshape(-1)
    input_images = torch.from_numpy(input_images).to(device)
    
    # objective function
    def closure(x):
        l = len(x)
        s = x[:int(l/2)]
        t = x[int(l/2):]
        s = torch.from_numpy(s).to(device)
        t = torch.from_numpy(t).to(device)
        
        transformed_arrays = input_images * s.view((-1, 1, 1)) + t.view((-1, 1, 1))
        dists = inter_distances(transformed_arrays)
        sqrt_dist = torch.sqrt(torch.mean(dists**2))
        
        if 'mean' == reduction:
            pred = torch.mean(transformed_arrays, dim=0)
        elif 'median' == reduction:
            pred = torch.median(transformed_arrays, dim=0).values
        else:
            raise ValueError
        
        near_err = torch.sqrt((0 - torch.min(pred))**2)
        far_err = torch.sqrt((1 - torch.max(pred))**2)
        
        err = sqrt_dist + (near_err + far_err) * regularizer_strength
        err = err.detach().cpu().numpy()
        return err
    
    res = minimize(closure, x, method='BFGS', tol=tol, options={'maxiter': max_iter, 'disp': disp})
    x = res.x
    l = len(x)
    s = x[:int(l/2)]
    t = x[int(l/2):]
    
    # Prediction
    transformed_arrays = original_input * s[:, np.newaxis, np.newaxis] + t[:, np.newaxis, np.newaxis]
    if 'mean' == reduction:
        aligned_images = np.mean(transformed_arrays, axis=0)
        std = np.std(transformed_arrays, axis=0)
        uncertainty = std
    elif 'median' == reduction:
        aligned_images = np.median(transformed_arrays, axis=0)
        # MAD (median absolute deviation) as uncertainty indicator
        abs_dev = np.abs(transformed_arrays - aligned_images)
        mad = np.median(abs_dev, axis=0)
        uncertainty = mad
    else:
        raise ValueError
    
    # Scale and shift to [0, 1]
    _min = np.min(aligned_images)
    _max = np.max(aligned_images)
    aligned_images = (aligned_images - _min) / (_max - _min)
    uncertainty /= (_max - _min)
    
    return aligned_images, uncertainty

