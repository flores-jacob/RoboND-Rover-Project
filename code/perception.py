import numpy as np
import cv2


# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:, :, 0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:, :, 0] > rgb_thresh[0]) \
                   & (img[:, :, 1] > rgb_thresh[1]) \
                   & (img[:, :, 2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select


# Define a function to convert to rover-centric coordinates
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = np.absolute(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[0]).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel ** 2 + y_pixel ** 2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles


# Define a function to apply a rotation to pixel positions
def rotate_pix(xpix, ypix, yaw):
    # TODO:
    # Convert yaw to radians
    yaw_rad = yaw * (np.pi / 180)
    cos_yaw = np.cos(yaw_rad)
    sin_yaw = np.sin(yaw_rad)

    # Apply a rotation
    xpix_rotated = (xpix * cos_yaw) - (ypix * sin_yaw)
    ypix_rotated = (xpix * sin_yaw) + (ypix * cos_yaw)
    # Return the result  
    return xpix_rotated, ypix_rotated


# Define a function to perform a translation
def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale):
    # TODO:
    # Apply a scaling and a translation
    x_world = np.int_(xpos + (xpix_rot / scale))
    y_world = np.int_(ypos + (ypix_rot / scale))

    xpix_translated = x_world
    ypix_translated = y_world
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
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))  # keep same size as input image

    return warped


# def get_wall_outlines(overhead_map):
#     """
#     This function takes an overhead map in the form of a numpy array, with elements with positive values as
#     traversable areas.  It returns an array with simply the wall outlines of the map.
#     :param overhead_map:
#     :return:
#     """
#     pass


def get_wall_outlines(rover_coords_x, rover_coords_y, look_forward=0):
    """
    This function takes an x and y coordinates of the rover in the forms of numpy arrays, with elements with positive
    values as traversable areas.  It returns arrays of the walls on the left and the right of the rover
    :param rover_coords_x:
    :param rover_coords_y:
    :return:
    """
    # get max y values for left side wall
    # get min y values for right side wall
    # problem here would be outlier specks which will distort the wall outline


    # I'm not yet that experienced with matrix manipulation, as such, I'll have to make do with a for loop
    # To speed the code up, it may be best to simply process the first few wall points closest to the rover
    # instead of processing the wall for the entire image

    xy_coords = np.stack((rover_coords_x, rover_coords_y), axis=1)

    # xy_coords is ordered starting from the highest x values, to the lowest. We flip it to order it from
    # lowest to highest, to make working with it more intuitive, solution taken from
    # https://stackoverflow.com/a/24813184

    xy_coords = np.flipud(xy_coords)

    # get all the unique values of X
    x_uniques = np.unique(rover_coords_x)

    # for each x value, get the max and min y value
    # save these values onto 2 separate arrays, or maybe we could save these onto a single array

    # initialize the array for the x coords and their respective maximum and minimum values
    xy_coords_with_max_y = np.zeros((len(x_uniques), 2))
    xy_coords_with_min_y = np.zeros((len(x_uniques), 2))

    print(len(x_uniques))

    if look_forward == 0:
        wall_predict = len(x_uniques)
    else:
        wall_predict = look_forward
    # for each x value, get all points with that x value and its respective y values
    # maybe we can also just use the nearest 10 or 20 points to speed up processing
    for x_index in range(wall_predict):
        x_value = x_uniques[x_index]
        # filter all coords, take only the current x_value in question
        ys_of_x = xy_coords[:, 0] == x_value
        ys_of_x_values = xy_coords[ys_of_x]

        # get max y value for the particular x value
        row_max = np.argmax(ys_of_x_values[:, 1])
        x_with_max_y = ys_of_x_values[row_max, :]

        # get min y value for the particular x value
        row_min = np.argmin(ys_of_x_values[:, 1])
        x_with_min_y = ys_of_x_values[row_min, :]

        # insert this set of coords on the array of coords with max y values
        xy_coords_with_max_y[x_index] = x_with_max_y
        xy_coords_with_min_y[x_index] = x_with_min_y
    # print(ys_of_x_values)
    #         print(x_with_max_y)
    #         print(x_with_min_y)

    #         savgol = savgol_filter(xy_coords_with_max_y[:, 1], 5, 2)
    #         plt.plot(xy_coords_with_max_y[:, 0], savgol, '.')

    #         x_new = np.linspace(xy_coords_with_max_y[:, 0].min(), xy_coords_with_max_y[:, 0].max(), 100)
    #         power_smooth = spline(xy_coords_with_max_y[:, 0], xy_coords_with_max_y[:, 1], x_new)
    #         plt.plot(x_new, power_smooth)

    #         tck = interpolate.splrep(xy_coords_with_max_y[:, 0], xy_coords_with_max_y[:, 1], k=5, s=1000)
    #         plt.plot(tck)
    #         f = interpolate.interp1d(xy_coords_with_max_y[:, 0], xy_coords_with_max_y[:, 1])
    #         ynew = f(xy_coords_with_max_y[:, 0])   # use interpolation function returned by `interp1d`
    #         plt.plot(xy_coords_with_max_y[:, 0], ynew, '-')

    left_wall_coords = xy_coords_with_max_y
    right_wall_coords = xy_coords_with_min_y

    return left_wall_coords, right_wall_coords


# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # TODO: 
    # NOTE: camera image is coming to you in Rover.img
    # 1) Define source and destination points for perspective transform

    image = Rover.img
    # source points
    rows, cols, ch = image.shape
    # lower 13, 140 and 303,140
    # higher 118, 95 and 200, 95
    base_y_src = 140
    upper_y_src = 95

    # source points
    lower_left_src = [13, base_y_src]
    lower_right_src = [303, base_y_src]
    upper_left_src = [118, upper_y_src]
    upper_right_src = [200, upper_y_src]

    # destination points
    base_y_dst = 140
    upper_y_dst = base_y_dst - 10
    image_vert_midline = int(cols / 2)
    x_left_dst = image_vert_midline - 5
    x_right_dst = image_vert_midline + 5

    lower_left_dst = [x_left_dst, base_y_dst]
    lower_right_dst = [x_right_dst, base_y_dst]
    upper_left_dst = [x_left_dst, upper_y_dst]
    upper_right_dst = [x_right_dst, upper_y_dst]

    source = np.float32([lower_left_src, lower_right_src, upper_right_src, upper_left_src])
    destination = np.float32([lower_left_dst, lower_right_dst, upper_right_dst, upper_left_dst])

    # 2) Apply perspective transform
    warped = perspect_transform(image, source, destination)
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    threshed = color_thresh(warped, rgb_thresh=(160, 160, 160))

    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
    # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
    #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
    #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image

    Rover.vision_image[:, :, 0] = threshed

    # 5) Convert map image pixel values to rover-centric coords
    xpix, ypix = rover_coords(threshed)

    # 6) Convert rover-centric pixel values to world coordinates
    scale = 10
    x_world, y_world = pix_to_world(xpix, ypix, Rover.pos[0],
                                    Rover.pos[1], Rover.yaw,
                                    Rover.worldmap.shape[0], scale)

    # 7) Update Rover worldmap (to be displayed on right side of screen)
    # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
    #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
    #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1

    Rover.worldmap[y_world, x_world] = 255

    # 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
    # Rover.nav_dists = rover_centric_pixel_distances
    # Rover.nav_angles = rover_centric_angles

    distances, angles = to_polar_coords(xpix, ypix)  # Convert to polar coords
    # avg_angle = np.mean(angles)

    # avg_angle_degrees = avg_angle * 180 / np.pi
    # steering = np.clip(avg_angle_degrees, -15, 15)

    Rover.nav_dists = distances
    Rover.nav_angles = angles

    # Rover.steer = steering

    return Rover
