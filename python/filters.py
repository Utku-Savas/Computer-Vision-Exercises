from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2
import math


def gaussian_helper(x, y, sigma=1.0, mu = 0):
    
    poe = ( -1 * ( (x - mu)**2 + (y - mu)**2 ) / (2 * sigma**2)) 
    coe = 1 / (2 * math.pi * sigma**2) 
    
    return coe * math.e**poe


def gaussian_kernel(size, sigma=1.0, mu=0):
    kernel = []
    for i in range(-(size//2), size//2+1):
        row = []
        for k in range(-(size//2), size//2+1):
            row.append(gaussian_helper(i, k, sigma, mu))
        kernel.append(row)
        
    kernel = np.array(kernel)
    
    return kernel


def mean_kernel(size):
    kernel = [1/(size*size) for i in range(size*size)]
    
    return np.array(kernel).reshape(size, size)


def sobelX_kernel():
    return np.array(([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]))


def sobelY_kernel():
    return np.array(([1, 2, 1], [0, 0, 0], [-1, -2, -1]))


def sharpen_kernel():
    return np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])


def convolve(image, kernel):
        
    (iH, iW) = image.shape[:2]
    (kH, kW) = kernel.shape[:2]
    
    pad = (kW - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype="float32")

    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
            k = (roi * kernel).sum()
            output[y - pad, x - pad] = k
            
    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")

    return output


def convolve_rgb(image, kernel):
    (B, G, R) = cv2.split(image)
    
    conv_image_r = convolve(R, kernel)
    conv_image_g = convolve(G, kernel)
    conv_image_b = convolve(B, kernel)
    
    output = cv2.merge([conv_image_b, conv_image_g, conv_image_r])
    return output







def median_operation(roi):
    l = sorted(roi.flatten().tolist())
    return l[len(l) // 2]


def conservative_smoothing_operation(roi):
    l = roi.flatten().tolist()
    
    center = l[len(l) // 2]
    maxp   = max(l)
    minp   = min(l)
    
    if center >= maxp:
        return maxp
    elif center <= minp:
        return minp
    else:
        return center


def apply_operation(image, kernel_size, operation):
    (iH, iW) = image.shape[:2]
    
    pad = (kernel_size - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype="float32")
    
    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
            output[y - pad, x - pad] = operation(roi)
            
    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")

    return output