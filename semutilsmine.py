import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.ndimage import generic_filter, convolve
from scipy.stats import mode

def generate_azimuth_angles(num_horizontal_points):
    # Generate azimuth angles evenly spaced from 0 to 360 degrees
    return np.linspace(0, 360, num_horizontal_points, endpoint=False)

def convert_range_image_to_point_cloud(range_image, beam_inclination, horizontal_fov=360.0):
    # Get the shape of the range image
    num_beams, num_horizontal_points = range_image.shape
    
    # Generate azimuth angles
    azimuth_angles = generate_azimuth_angles(num_horizontal_points)
    
    # Convert angles from degrees to radians
    azimuth_rad = np.radians(azimuth_angles)
    #elevation_rad = np.radians(beam_inclination)
    elevation_rad = beam_inclination
    # Initialize arrays for point cloud
    x_coords = np.zeros_like(range_image)
    y_coords = np.zeros_like(range_image)
    z_coords = np.zeros_like(range_image)
    
    # Compute Cartesian coordinates for each point
    for i in range(num_beams):
        x_coords[i, :] = range_image[i, :] * np.cos(elevation_rad[i]) * np.cos(azimuth_rad)
        y_coords[i, :] = range_image[i, :] * np.cos(elevation_rad[i]) * np.sin(azimuth_rad)
        z_coords[i, :] = range_image[i, :] * np.sin(elevation_rad[i])
    
    # Stack the coordinates to form a point cloud
    point_cloud = np.stack([x_coords, y_coords, z_coords], axis=-1)
    
    return point_cloud


#point_cloud = convert_range_image_to_point_cloud(range_image, beam_inclination)

def generate_azimuth_angles_torch(num_horizontal_points):
    return torch.linspace(-180, 0, num_horizontal_points, endpoint=False)

def convert_range_image_to_point_cloud_torch(range_image, beam_inclination, horizontal_fov=360.0):
    # Get the shape of the range image
    num_beams, num_horizontal_points = range_image.shape
    
    # Generate azimuth angles
    azimuth_angles = generate_azimuth_angles_torch(num_horizontal_points)
    
    # Convert angles from degrees to radians
    azimuth_rad = torch.radians(torch.tensor(azimuth_angles))
    elevation_rad = beam_inclination  # Assuming beam_inclination is already a tensor in radians
    
    # Initialize tensors for point cloud
    x_coords = torch.zeros_like(range_image)
    y_coords = torch.zeros_like(range_image)
    z_coords = torch.zeros_like(range_image)
    
    # Compute Cartesian coordinates for each point
    for i in range(num_beams):
        x_coords[i, :] = range_image[i, :] * torch.cos(elevation_rad[i]) * torch.cos(azimuth_rad)
        y_coords[i, :] = range_image[i, :] * torch.cos(elevation_rad[i]) * torch.sin(azimuth_rad)
        z_coords[i, :] = range_image[i, :] * torch.sin(elevation_rad[i])
    
    # Stack the coordinates to form a point cloud
    point_cloud = torch.stack([x_coords, y_coords, z_coords], axis=-1)
    
    return point_cloud


def interpolate_pixel3(values):
    center = values[4]  # The central pixel
    if center != -1:
        return center

    valid_values = values[values != -1]
    if len(valid_values) >= 3:
        return np.mean(valid_values)
    else:
        return center

def interpolate_nan_pixels3(image):
    return generic_filter(image, interpolate_pixel3, size=3, mode='constant', cval=-1)

def interpolate_pixel5(values):
    center = values[4]  # The central pixel
    if center != -1:
        return center

    valid_values = values[values != -1]
    if len(valid_values) >= 3:
        return np.mean(valid_values)
    else:
        return center

def interpolate_nan_pixels5(image):
    return generic_filter(image, interpolate_pixel5, size=3, mode='constant', cval=-1)


def interpolate_pixel_with_majority_check5(values):
    center = values[4] 
    
    if center != 0:
        return center
    
    valid_values = values[values != 0]
    
    if len(valid_values) >= 5:
        return mode(valid_values)[0]
    else:
        return center

def interpolate_nan_pixels_with_majority5(image):
    return generic_filter(image, interpolate_pixel_with_majority_check5, size=3, mode='constant', cval=0)

def interpolate_pixel_with_majority_check3(values):
    center = values[4] 
    
    if center != 0:
        return center
    
    valid_values = values[values != 0]
    
    if len(valid_values) >= 5:
        return mode(valid_values)[0]
    else:
        return center

def interpolate_nan_pixels_with_majority3(image):
    return generic_filter(image, interpolate_pixel_with_majority_check3, size=3, mode='constant', cval=0)

def fast_interpolate(arr, thresholds=[5, 3]):
    mask = (arr == -1)
    
    # Define a 3x3 kernel for counting neighbors
    kernel = np.ones((3, 3))
    kernel[1, 1] = 0  # Do not count the center pixel itself
    
    for threshold in thresholds:
        # Update the mask for -1 pixels
        mask = (arr == -1)
        mmask = (~mask).astype(np.int8)
        
        # Count valid neighbors by applying convolution
        valid_neighbor_count = convolve(mmask, kernel, mode='constant', cval=0)

        # Compute the sum of valid neighbors
        neighbor_sum = convolve(np.where(mask, 0, arr), kernel, mode='constant', cval=0)

        # Find pixels that should be interpolated
        interpolate_mask = (valid_neighbor_count >= threshold) & mask

        # Interpolate those pixels by dividing the sum by the number of valid neighbors
        arr[interpolate_mask] = neighbor_sum[interpolate_mask] / valid_neighbor_count[interpolate_mask]

    return arr
