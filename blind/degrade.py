#the degradation process
import glob
import cv2
import sys 
from os.path import join, exists
import numpy as np
import torch
from utils_blindsr import bicubic_degradation,srmd_degradation,dpsr_degradation,classical_degradation,add_Gaussian_noise
sys.path.append("..") 
from util import cv2_imread,cv2_imsave,DUF_downsample,automkdir
#对video的degradation

