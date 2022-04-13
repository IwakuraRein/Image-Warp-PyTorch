import torch
import imageio
import numpy as np
import os

from warp_utils import image_warp_accum

device = torch.device('cuda:0')

def tone_mapping(input_image):
    tone_mapped_color = torch.clamp(
        torch.pow(torch.max(torch.zeros_like(input_image), input_image), 0.454545), 0., 1.)
    return tone_mapped_color

img0  = torch.Tensor(np.expand_dims(imageio.imread(r'./Images/img0.exr'),0)).to(device)
img1  = torch.Tensor(np.expand_dims(imageio.imread(r'./Images/img1.exr'),0)).to(device)
img2  = torch.Tensor(np.expand_dims(imageio.imread(r'./Images/img2.exr'),0)).to(device)
img3  = torch.Tensor(np.expand_dims(imageio.imread(r'./Images/img3.exr'),0)).to(device)
img4  = torch.Tensor(np.expand_dims(imageio.imread(r'./Images/img4.exr'),0)).to(device)
img0 = img0.permute([0,3,1,2]).contiguous()
img1 = img1.permute([0,3,1,2]).contiguous()
img2 = img2.permute([0,3,1,2]).contiguous()
img3 = img3.permute([0,3,1,2]).contiguous()
img4 = img4.permute([0,3,1,2]).contiguous()
motion_vector1 = np.expand_dims(np.load("./motion_vectors/0--1.npy").astype(np.float32),0)
motion_vector2 = np.expand_dims(np.load("./motion_vectors/1--2.npy").astype(np.float32),0)
motion_vector3 = np.expand_dims(np.load("./motion_vectors/2--3.npy").astype(np.float32),0)
motion_vector4 = np.expand_dims(np.load("./motion_vectors/3--4.npy").astype(np.float32),0)
motion_vector1 = torch.Tensor(motion_vector1).to(device)
motion_vector2 = torch.Tensor(motion_vector2).to(device)
motion_vector3 = torch.Tensor(motion_vector3).to(device)
motion_vector4 = torch.Tensor(motion_vector4).to(device)

imgbd1 = image_warp_accum.apply(img0, img1, motion_vector1, 0.5)
imgbd2 = image_warp_accum.apply(imgbd1, img2, motion_vector2, 0.5)
imgbd3 = image_warp_accum.apply(imgbd2, img3, motion_vector3, 0.5)
imgbd4 = image_warp_accum.apply(imgbd3, img4, motion_vector4, 0.5)

imgOut = tone_mapping(imgbd4).permute([0, 2, 3, 1]).cpu().numpy()[0]
imageio.imwrite('imgOut.png', imgOut)
imageio.imwrite('imgOut.exr', imgOut)
