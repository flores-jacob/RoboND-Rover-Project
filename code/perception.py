import numpy as np
import cv2

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

def color_thresh_range(img, rgb_lower, rgb_higher):
    """
    takes an image input, and returns an image composed of colors within range
    :param img:
    :param rgb_lower: lower range
    :param rgb_higher: higher range
    :return:
    """
    color_select = np.zeros_like(img[:, :, 0])
    above_thresh = (img[:, :, 0] > rgb_lower[0]) \
                   & (img[:, :, 1] > rgb_lower[1]) \
                   & (img[:, :, 2] > rgb_lower[2])
    below_thresh = (img[:, :, 0] < rgb_higher[0]) \
                   & (img[:, :, 1] < rgb_higher[1]) \
                   & (img[:, :, 2] < rgb_higher[2])

    color_select[above_thresh & below_thresh] = 1
    return color_select



# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
                            
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result  
    return xpix_rotated, ypix_rotated

def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result  
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    
    return warped


# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # NOTE: camera image is coming to you in Rover.img
    image = Rover.img

    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    # navigable_terrain = color_thresh(warped, rgb_thresh=(160, 160, 160))

    # threshold the image before warping, to avoid artifacts in the resulting image
    nav_lower_range = (180, 180, 180)
    nav_upper_range = (256, 256, 256)
    navigable_terrain = color_thresh_range(image, nav_lower_range, nav_upper_range)

    obstacle_lower_range = (0, 0, 0)
    obstacle_upper_range = (179, 179, 179)
    obstacle_terrain = color_thresh_range(image, obstacle_lower_range, obstacle_upper_range)

    rock_sample_thresh = color_thresh_range(image, (110, 110, 0), (255, 255, 50))

    # source points
    rows, cols, ch = image.shape
    # lower 13, 140 and 303,140
    # higher 118, 95 and 200, 95
    dst_size = 5
    # Set a bottom offset to account for the fact that the bottom of the image
    # is not the position of the rover but a bit in front of it
    # this is just a rough guess, feel free to change it!
    bottom_offset = 6
    source = np.float32([[14, 140], [301, 140], [200, 96], [118, 96]])
    destination = np.float32([[image.shape[1] / 2 - dst_size, image.shape[0] - bottom_offset],
                              [image.shape[1] / 2 + dst_size, image.shape[0] - bottom_offset],
                              [image.shape[1] / 2 + dst_size, image.shape[0] - 2 * dst_size - bottom_offset],
                              [image.shape[1] / 2 - dst_size, image.shape[0] - 2 * dst_size - bottom_offset],
                              ])
    # 2) Apply perspective transform
    warped_navigable_terrain = perspect_transform(navigable_terrain, source, destination)
    warped_obstacle_terrain = perspect_transform(obstacle_terrain, source, destination)
    warped_rock_sample_thresh = perspect_transform(rock_sample_thresh, source, destination)

    # zero out the values of any obstacles found beneath the offset
    # bottom_offset = 20
    # obstacle_terrain[warped.shape[0] - bottom_offset:warped.shape[0], :] = obstacle_terrain[
    #                                                                        warped.shape[0] - bottom_offset:warped.shape[
    #                                                                            0], :] * 0



    # remove the upper part of the image. this is because what's seen further away is normally
    # inaccurate.
    bottom_offset = 100
    warped_obstacle_terrain[0:bottom_offset, :] = warped_obstacle_terrain[0:bottom_offset, :] * 0
    warped_navigable_terrain[0:bottom_offset, :] = warped_navigable_terrain[0:bottom_offset, :] * 0
    warped_rock_sample_thresh[0:bottom_offset, :] = warped_rock_sample_thresh[0:bottom_offset, :] * 0


    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
    # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
    #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
    #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image

    Rover.vision_image[:, :, 0] = warped_obstacle_terrain * 255
    Rover.vision_image[:, :, 1] = warped_rock_sample_thresh * 255
    Rover.vision_image[:, :, 2] = warped_navigable_terrain * 255

    # 5) Convert map image pixel values to rover-centric coords
    obstacle_xpix, obstacle_ypix = rover_coords(warped_obstacle_terrain)
    navigable_xpix, navigable_ypix = rover_coords(warped_navigable_terrain)
    rock_sample_xpix, rock_sample_ypix = rover_coords(warped_rock_sample_thresh)

    # 6) Convert rover-centric pixel values to world coordinates
    scale = 10
    obstacle_x_world, obstacle_y_world = pix_to_world(obstacle_xpix, obstacle_ypix, Rover.pos[0],
                                                      Rover.pos[1], Rover.yaw,
                                                      Rover.worldmap.shape[0], scale)

    rock_sample_x_world, rock_sample_y_world = pix_to_world(rock_sample_xpix, rock_sample_ypix, Rover.pos[0],
                                                            Rover.pos[1], Rover.yaw,
                                                            Rover.worldmap.shape[0], scale)

    x_world, y_world = pix_to_world(navigable_xpix, navigable_ypix, Rover.pos[0],
                                    Rover.pos[1], Rover.yaw,
                                    Rover.worldmap.shape[0], scale)

    # 7) Update Rover worldmap (to be displayed on right side of screen)
    # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
    #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
    #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1

    if ((Rover.roll > 359.5) or (Rover.roll < 0.5)) and ((Rover.pitch > 359.5) or (Rover.pitch < 0.5)):
        Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] = 255
        Rover.worldmap[rock_sample_y_world, rock_sample_x_world, 1] = 255
        Rover.worldmap[y_world, x_world, 2] = 255

    # 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
    # Rover.nav_dists = rover_centric_pixel_distances
    # Rover.nav_angles = rover_centric_angles

    obstacle_distances, obstacle_angles = to_polar_coords(obstacle_xpix, obstacle_ypix)
    rock_sample_distances, rock_sample_angles = to_polar_coords(rock_sample_xpix, rock_sample_xpix)
    distances, angles = to_polar_coords(navigable_xpix, navigable_ypix)  # Convert to polar coords

    # avg_angle = np.mean(angles)
    # avg_angle_degrees = avg_angle * 180 / np.pi
    # steering = np.clip(avg_angle_to_obstacles, -15, 15)

    Rover.nav_dists = distances
    Rover.nav_angles = angles

    Rover.obstacle_angles = obstacle_angles

    return Rover