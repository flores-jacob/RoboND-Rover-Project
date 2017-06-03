import numpy as np
import cv2

import path_generation_helpers


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


def polar_coords_with_origin(origin_x, origin_y, x_pixels, y_pixels):
    pass


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


def get_side_wall_outlines(navigable_rover_x_coords, navigable_rover_y_coords, look_forward=0):
    """
    This function takes an x and y coordinates of the rover in the forms of numpy arrays, with elements with positive
    values as traversable areas.  It returns arrays of the walls on the left and the right of the rover
    :param navigable_rover_x_coords:
    :param navigable_rover_y_coords:
    :return:
    """
    # get max y values for left side wall
    # get min y values for right side wall
    # problem here would be outlier specks which will distort the wall outline

    # I'm not yet that experienced with matrix manipulation, as such, I'll have to make do with a for loop
    # To speed the code up, it may be best to simply process the first few wall points closest to the rover
    # instead of processing the wall for the entire image

    xy_coords = np.stack((navigable_rover_x_coords, navigable_rover_y_coords), axis=1)

    # xy_coords is ordered starting from the highest x values, to the lowest. We flip it to order it from
    # lowest to highest, to make working with it more intuitive, solution taken from
    # https://stackoverflow.com/a/24813184

    xy_coords = np.flipud(xy_coords)

    # get all the unique values of X
    x_uniques = np.unique(navigable_rover_x_coords)

    # for each x value, get the max and min y value
    # save these values onto 2 separate arrays, or maybe we could save these onto a single array

    # initialize the array for the x coords and their respective maximum and minimum values
    xy_coords_with_max_y = np.zeros((len(x_uniques), 2))
    xy_coords_with_min_y = np.zeros((len(x_uniques), 2))

    if (look_forward == 0) or (len(x_uniques) < look_forward):
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

    left_wall_coords = xy_coords_with_max_y
    right_wall_coords = xy_coords_with_min_y

    return left_wall_coords, right_wall_coords


# unused function, superseded by front_vicinity sampler
def front_obstacle_coords(rover_coords_x, rover_coords_y, look_forward=25):
    """
    Return coordinates of obstacles within 25 pixels in front of rover (x axis), with y equivalent
      to 0 or 1
    :param rover_coords_x:
    :param rover_coords_y:
    :param look_forward: int indicating how many pixels from the front of the rover would be considered obstacles
    :return:
    """
    xy_coords = np.stack((rover_coords_x, rover_coords_y), axis=1)
    xy_coords = np.flipud(xy_coords)
    obstacle_coords_indices = (xy_coords[:, 1] == (0 | 1 | -1)) & (
        (xy_coords[:, 0] <= look_forward) & (xy_coords[:, 0] > 15))
    obstacle_coords = xy_coords[obstacle_coords_indices]
    # print("obstacle coords ", obstacle_coords)

    return obstacle_coords


# unused function
def get_surrounding_pixel_types(rover_x_pos, rover_y_pos, memory_map_single_layer):
    # offset is the number of pixels * 10 from the origin
    offset = 2 * 10

    # note that the x,y coordinate order is inversed.  This is because the first value accesses the
    # row values or m for the y values, and the x values are in the columns
    north_pixel = memory_map_single_layer[rover_y_pos + offset, rover_x_pos]
    south_pixel = memory_map_single_layer[rover_y_pos - offset, rover_x_pos]
    east_pixel = memory_map_single_layer[rover_y_pos, rover_x_pos + offset]
    west_pixel = memory_map_single_layer[rover_y_pos, rover_x_pos - offset]

    northwest_pixel = memory_map_single_layer[rover_y_pos + offset, rover_x_pos - offset]
    northeast_pixel = memory_map_single_layer[rover_y_pos + offset, rover_x_pos + offset]
    southwest_pixel = memory_map_single_layer[rover_y_pos - offset, rover_x_pos - offset]
    southeast_pixel = memory_map_single_layer[rover_y_pos - offset, rover_x_pos + offset]

    origin = memory_map_single_layer[rover_y_pos, rover_x_pos]

    surrounding_pixels = np.asarray([[northwest_pixel, north_pixel, northeast_pixel],
                                     [west_pixel, origin, east_pixel],
                                     [southwest_pixel, south_pixel, southeast_pixel]])

    return surrounding_pixels


# unused function
def identify_surrounding_pixels(rover_x_pos, rover_y_pos, memory_map):
    surrounding_obstacle_pixels = get_surrounding_pixel_types(rover_x_pos, rover_y_pos, memory_map[:, :, 0])
    surrounding_rock_sample_pixels = get_surrounding_pixel_types(rover_x_pos, rover_y_pos, memory_map[:, :, 1])
    surrounding_navigable_pixels = get_surrounding_pixel_types(rover_x_pos, rover_y_pos, memory_map[:, :, 2])

    surrounding_pixels = np.zeros([3, 3], dtype=np.float)

    for i in range(0, 3):
        for j in range(0, 3):
            # Use 5, 6, 7 to avoid confusion with the use of 0, 1, 2, and 3 in other parts of code
            if surrounding_obstacle_pixels[i][j] > 0:
                surrounding_pixels[i][j] = 5  # fives are obstacle pixels
            if surrounding_rock_sample_pixels[i][j] > 0:
                surrounding_pixels[i][j] = 6  # sixs are rock sample pixels
            elif surrounding_navigable_pixels[i][j] > 0:
                surrounding_pixels[i][j] = 7  # sevens are navigable pixels
                # else:
                #     surrounding_pixels[i][j] = 3  # threes are unexplored pixels

    return surrounding_pixels


def front_vicinity_sampler(memory_xpos, memory_ypos, yaw, memory_map):
    """
    get a box of memory pixel points from the memory map
    :param memory_xpos:
    :param memory_ypos:
    :param yaw:
    :param memory_map: Entire memory map with all its layers
    :return: returns multilayered memory pixels from memory_map
    """
    # offset is the number of memory_pixel from the side/front of the rover. 10 memory pixels = 1 worldmap pixel
    offset = 8

    # box size is how large we want our vicinity sample to be
    box_size = 15

    x = memory_xpos
    y = memory_ypos

    leftmost_x_adjust = offset
    bottom_y_adjust = -box_size / 2
    rotated_leftmost_x_adjust, rotated_bottom_y_adjust = rotate_pix(leftmost_x_adjust, bottom_y_adjust, yaw)

    topmost_y_adjust = box_size / 2
    rightmost_x_adjust = offset + box_size
    rotated_righmost_x_adjust, rotated_topmost_y_adjust = rotate_pix(rightmost_x_adjust, topmost_y_adjust, yaw)

    # print("bottomy", rotated_bottom_y_adjust)
    # print("topy   ", rotated_topmost_y_adjust)
    # print("lefx   ", rotated_leftmost_x_adjust)
    # print("righx  ", rotated_righmost_x_adjust)

    lower_y = min([y + rotated_bottom_y_adjust, y + rotated_topmost_y_adjust])
    higher_y = max([y + rotated_bottom_y_adjust, y + rotated_topmost_y_adjust])

    lower_x = min([x + rotated_leftmost_x_adjust, x + rotated_righmost_x_adjust])
    higher_x = max([x + rotated_leftmost_x_adjust, x + rotated_righmost_x_adjust])

    # print("range ", lower_y, ":", higher_y)
    # print("range ", lower_x, ":", higher_x)

    # slice the array of pixels from the memory_map
    front_vicinity_array = memory_map[
                           int(round(lower_y)):int(round(higher_y)),
                           int(round(lower_x)): int(round(higher_x)),
                           :
                           ]
    # slice the array of pixels from the memory_map
    # vicinity_array = memory_map_layer[
    #                  bottom_y_adjust:topmost_y_adjust,
    #                  leftmost_x_adjust: rightmost_x_adjust
    #                  ]
    # print("shape ", vicinity_array.shape)

    return front_vicinity_array


def left_vicinity_sampler(memory_xpos, memory_ypos, yaw, memory_map):
    """
    get a box of memory pixel points from the memory map
    :param memory_xpos:
    :param memory_ypos:
    :param yaw:
    :param memory_map: Entire memory map with all its layers
    :return: returns multilayered memory pixels from memory_map
    """
    # offset is the number of memory_pixel from the side/front of the rover. 10 memory pixels = 1 worldmap pixel
    offset = 3
    rover_width = 6

    # box size is how large we want our vicinity sample to be
    box_size = 15

    x = memory_xpos
    y = memory_ypos

    leftmost_x_adjust = -box_size
    bottom_y_adjust = (rover_width / 2) + offset
    rotated_leftmost_x_adjust, rotated_bottom_y_adjust = rotate_pix(leftmost_x_adjust, bottom_y_adjust, yaw)

    topmost_y_adjust = (rover_width / 2) + offset + box_size
    rightmost_x_adjust = 0
    rotated_righmost_x_adjust, rotated_topmost_y_adjust = rotate_pix(rightmost_x_adjust, topmost_y_adjust, yaw)

    lower_y = min([y + rotated_bottom_y_adjust, y + rotated_topmost_y_adjust])
    higher_y = max([y + rotated_bottom_y_adjust, y + rotated_topmost_y_adjust])

    lower_x = min([x + rotated_leftmost_x_adjust, x + rotated_righmost_x_adjust])
    higher_x = max([x + rotated_leftmost_x_adjust, x + rotated_righmost_x_adjust])

    # slice the array of pixels from the memory_map
    left_vicinity_array = memory_map[
                          int(round(lower_y)):int(round(higher_y)),
                          int(round(lower_x)): int(round(higher_x)),
                          :]

    return left_vicinity_array


def right_vicinity_sampler(memory_xpos, memory_ypos, yaw, memory_map):
    """
    get a box of memory pixel points from the memory map
    :param memory_xpos:
    :param memory_ypos:
    :param yaw:
    :param memory_map: Entire memory map with all its layers
    :return: returns multilayered memory pixels from memory_map
    """
    # offset is the number of memory_pixel from the side/front of the rover. 10 memory pixels = 1 worldmap pixel
    offset = 3
    rover_width = 6

    # box size is how large we want our vicinity sample to be
    box_size = 15

    x = memory_xpos
    y = memory_ypos

    leftmost_x_adjust = - box_size
    bottom_y_adjust = (-rover_width / 2) - offset - box_size
    rotated_leftmost_x_adjust, rotated_bottom_y_adjust = rotate_pix(leftmost_x_adjust, bottom_y_adjust, yaw)

    topmost_y_adjust = (-rover_width / 2) + offset
    rightmost_x_adjust = 0
    rotated_righmost_x_adjust, rotated_topmost_y_adjust = rotate_pix(rightmost_x_adjust, topmost_y_adjust, yaw)

    lower_y = min([y + rotated_bottom_y_adjust, y + rotated_topmost_y_adjust])
    higher_y = max([y + rotated_bottom_y_adjust, y + rotated_topmost_y_adjust])

    lower_x = min([x + rotated_leftmost_x_adjust, x + rotated_righmost_x_adjust])
    higher_x = max([x + rotated_leftmost_x_adjust, x + rotated_righmost_x_adjust])

    # slice the array of pixels from the memory_map
    left_vicinity_array = memory_map[
                          int(round(lower_y)):int(round(higher_y)),
                          int(round(lower_x)): int(round(higher_x)),
                          :]

    return left_vicinity_array


def upper_left_vicinity_sampler(memory_xpos, memory_ypos, yaw, memory_map):
    """
    get a box of memory pixel points from the memory map
    :param memory_xpos:
    :param memory_ypos:
    :param yaw:
    :param memory_map: Entire memory map with all its layers
    :return: returns multilayered memory pixels from memory_map
    """
    # offset is the number of memory_pixel from the side/front of the rover. 10 memory pixels = 1 worldmap pixel
    y_offset = 0
    x_offset = 3
    rover_width = 6

    # box size is how large we want our vicinity sample to be
    box_size = 10

    x = memory_xpos
    y = memory_ypos

    leftmost_x_adjust = x_offset
    bottom_y_adjust = (rover_width / 2) + y_offset
    rotated_leftmost_x_adjust, rotated_bottom_y_adjust = rotate_pix(leftmost_x_adjust, bottom_y_adjust, yaw)

    topmost_y_adjust = y_offset + (rover_width / 2) + box_size
    rightmost_x_adjust = x_offset + box_size
    rotated_righmost_x_adjust, rotated_topmost_y_adjust = rotate_pix(rightmost_x_adjust, topmost_y_adjust, yaw)

    lower_y = min([y + rotated_bottom_y_adjust, y + rotated_topmost_y_adjust])
    higher_y = max([y + rotated_bottom_y_adjust, y + rotated_topmost_y_adjust])

    lower_x = min([x + rotated_leftmost_x_adjust, x + rotated_righmost_x_adjust])
    higher_x = max([x + rotated_leftmost_x_adjust, x + rotated_righmost_x_adjust])

    print("x ", lower_x, higher_x)
    print("y ", lower_y, higher_x)

    # slice the array of pixels from the memory_map
    upper_left_vicinity_array = memory_map[
                                int(round(lower_y)):int(round(higher_y)),
                                int(round(lower_x)): int(round(higher_x)),
                                :]

    return upper_left_vicinity_array


def upper_right_vicinity_sampler(memory_xpos, memory_ypos, yaw, memory_map):
    """
    get a box of memory pixel points from the memory map
    :param memory_xpos:
    :param memory_ypos:
    :param yaw:
    :param memory_map: Entire memory map with all its layers
    :return: returns multilayered memory pixels from memory_map
    """
    # offset is the number of memory_pixel from the side/front of the rover. 10 memory pixels = 1 worldmap pixel
    y_offset = 0
    x_offset = 3
    rover_width = 6

    # box size is how large we want our vicinity sample to be
    box_size = 10

    x = memory_xpos
    y = memory_ypos

    leftmost_x_adjust = x_offset
    bottom_y_adjust = (-rover_width / 2) - y_offset - box_size
    rotated_leftmost_x_adjust, rotated_bottom_y_adjust = rotate_pix(leftmost_x_adjust, bottom_y_adjust, yaw)

    topmost_y_adjust = -y_offset - (rover_width / 2)
    rightmost_x_adjust = x_offset + box_size
    rotated_righmost_x_adjust, rotated_topmost_y_adjust = rotate_pix(rightmost_x_adjust, topmost_y_adjust, yaw)

    lower_y = min([y + rotated_bottom_y_adjust, y + rotated_topmost_y_adjust])
    higher_y = max([y + rotated_bottom_y_adjust, y + rotated_topmost_y_adjust])

    lower_x = min([x + rotated_leftmost_x_adjust, x + rotated_righmost_x_adjust])
    higher_x = max([x + rotated_leftmost_x_adjust, x + rotated_righmost_x_adjust])

    print("x ", lower_x, higher_x)
    print("y ", lower_y, higher_x)

    # slice the array of pixels from the memory_map
    upper_left_vicinity_array = memory_map[
                                int(round(lower_y)):int(round(higher_y)),
                                int(round(lower_x)): int(round(higher_x)),
                                :]

    return upper_left_vicinity_array


def obstacle_detected(vicinity_array):
    obstacle_count = np.count_nonzero(vicinity_array[:, :, 0])
    # rock_sample_count = np.count_nonzero(vicinity_array[:,:,1])
    navigable_terrain_count = np.count_nonzero(vicinity_array[:, :, 2])

    if obstacle_count and ((float(obstacle_count) / vicinity_array[:, :, 0].size) > .3) and (
                obstacle_count > navigable_terrain_count):
        # print("obstacles ", obstacle_count / vicinity_array.size)
        obstacles_present = True
    else:
        obstacles_present = False

    return obstacles_present


def navigable_area(vicinity_array):
    obstacle_count = np.count_nonzero(vicinity_array[:, :, 0])
    navigable_terrain_count = np.count_nonzero(vicinity_array[:, :, 2])

    # It's not enough that there are more navigable pixels than obstacles. The area should also be passable
    if navigable_terrain_count and ((float(navigable_terrain_count) / vicinity_array[:, :, 2].size) > .3) and (
                navigable_terrain_count > obstacle_count):
        is_navigable = True
    else:
        is_navigable = False

    return is_navigable


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
    # navigable_terrain = color_thresh(warped, rgb_thresh=(160, 160, 160))

    nav_lower_range = (160, 160, 160)
    nav_upper_range = (256, 256, 256)
    navigable_terrain = color_thresh_range(warped, nav_lower_range, nav_upper_range)

    obstacle_lower_range = (0, 0, 0)
    obstacle_upper_range = (100, 100, 100)
    obstacle_terrain = color_thresh_range(warped, obstacle_lower_range, obstacle_upper_range)
    rock_sample_thresh = color_thresh_range(warped, (100, 100, 0), (255, 255, 75))

    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
    # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
    #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
    #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image

    Rover.vision_image[:, :, 0] = obstacle_terrain * 255
    Rover.vision_image[:, :, 1] = rock_sample_thresh * 255
    Rover.vision_image[:, :, 2] = navigable_terrain * 255

    # 5) Convert map image pixel values to rover-centric coords
    obstacle_xpix, obstacle_ypix = rover_coords(obstacle_terrain)
    navigable_xpix, navigable_ypix = rover_coords(navigable_terrain)
    rock_sample_xpix, rock_sample_ypix = rover_coords(rock_sample_thresh)

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

    # 6b) convert rover-centric pixels to coordinates for a 2000 x 2000 memory map
    # same scale as the world map. size difference between the rover centric map and the objective map
    memory_scale = 10

    # make sure that the resultant points are multiplied according by 10 (different from the "scale" value above).
    # This is so that these points occupy the entire 2000 x 2000 map which is 10 times larger than the orginal worldmap
    obstacle_x_memory, obstacle_y_memory = pix_to_world(obstacle_xpix, obstacle_ypix, Rover.pos[0],
                                                        Rover.pos[1], Rover.yaw,
                                                        Rover.memory_map.shape[0], memory_scale)

    rock_sample_x_memory, rock_sample_y_memory = pix_to_world(rock_sample_xpix, rock_sample_ypix,
                                                              Rover.pos[0],
                                                              Rover.pos[1], Rover.yaw,
                                                              Rover.memory_map.shape[0], memory_scale)

    navigable_x_memory, navigable_y_memory = pix_to_world(navigable_xpix, navigable_ypix, Rover.pos[0],
                                                          Rover.pos[1], Rover.yaw,
                                                          Rover.memory_map.shape[0], memory_scale)

    # 7) Update Rover worldmap (to be displayed on right side of screen)
    # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
    #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
    #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1

    if ((Rover.roll > 359) or (Rover.roll < 1)) and ((Rover.pitch > 359) or (Rover.pitch < 1)):
        Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] = 255
        Rover.worldmap[rock_sample_y_world, rock_sample_x_world, 1] = 255
        Rover.worldmap[y_world, x_world, 2] = 255

        # 7b) Update memory map to save coordinates of the seen pixels

        # Rover.memory_map[obstacle_y_memory, obstacle_x_memory, 0] = 255
        # Rover.memory_map[rock_sample_y_memory, rock_sample_x_memory, 1] = 255
        # Rover.memory_map[navigable_y_memory, navigable_x_memory, 2] = 255

        Rover.memory_map[obstacle_y_memory, obstacle_x_memory] = (255, 0, 0, 5)
        Rover.memory_map[rock_sample_y_memory, rock_sample_x_memory] = (0, 255, 0, 6)
        Rover.memory_map[navigable_y_memory, navigable_x_memory] = (0, 0, 255, 7)

        # 7c) where nav_terrain = 255 and obstacle_terrain = 255 set obstacle_terrain to zero
        # this is to prevent shadows of obstacle terrain from conflicting with navigable terrain
        navigable_terrain_indices = np.where(Rover.memory_map[:, :, 2] == 255)
        Rover.memory_map[:, :, 0][navigable_terrain_indices] = 0

    # 7d) note 3 areas of interest based on previously mapped data(Rover.memory_map), front of the rover
    # front_box = front_vicinity_sampler(int(round(Rover.pos[0] * 10)), int(round(Rover.pos[1] * 10)), Rover.yaw,
    #                                    Rover.memory_map)
    # left_box = left_vicinity_sampler(int(round(Rover.pos[0] * 10)), int(round(Rover.pos[1] * 10)), Rover.yaw,
    #                                  Rover.memory_map)
    # right_box = right_vicinity_sampler(int(round(Rover.pos[0] * 10)), int(round(Rover.pos[1] * 10)), Rover.yaw,
    #                                    Rover.memory_map)
    # upper_left_box = upper_left_vicinity_sampler(int(round(Rover.pos[0] * 10)), int(round(Rover.pos[1] * 10)),
    #                                              Rover.yaw,
    #                                              Rover.memory_map)
    # upper_right_box = upper_right_vicinity_sampler(int(round(Rover.pos[0] * 10)), int(round(Rover.pos[1] * 10)),
    #                                                Rover.yaw,
    #                                                Rover.memory_map)
    # # Check if the front is navigable and make a note of it
    # Rover.front_navigable = navigable_area(front_box)
    # Rover.front_obstacle = obstacle_detected(front_box)
    # # Check if there's a wall/obstacle on the left to follow
    # Rover.left_obstacle = obstacle_detected(left_box)
    # Rover.right_obstacle = obstacle_detected(right_box)
    # # Check if the upper left hand corner still has a wall to follow
    # Rover.upper_left_obstacle = obstacle_detected(upper_left_box)
    # Rover.upper_right_obstacle = obstacle_detected(upper_right_box)

    # 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
    # Rover.nav_dists = rover_centric_pixel_distances
    # Rover.nav_angles = rover_centric_angles

    obstacle_distances, obstacle_angles = to_polar_coords(obstacle_xpix, obstacle_ypix)
    rock_sample_distances, rock_sample_angles = to_polar_coords(rock_sample_xpix, rock_sample_xpix)
    distances, angles = to_polar_coords(navigable_xpix, navigable_ypix)  # Convert to polar coords

    # Rover.find_wall_angle = (np.mean(-obstacle_angles) / 2) * (180 / np.pi)

    # avg_angle = np.mean(angles)
    # avg_angle_degrees = avg_angle * 180 / np.pi
    # steering = np.clip(avg_angle_to_obstacles, -15, 15)

    Rover.nav_dists = distances
    Rover.nav_angles = angles

    if Rover.destination_point is None:
        destination_point, destination_distance, destination_angle = path_generation_helpers.choose_destination(
            Rover.pos[0], Rover.pos[1], Rover.memory_map[:, :, 3])
        Rover.destination_point = destination_point
        Rover.destination_distance = destination_distance
        Rover.destination_angle = destination_angle

        print(Rover.destination_angle)



    # for destination_pixel, angle, and distance to work, the rover has to be fully stopped. Otherwise calculations
    # won't be accurate. Also, obstacle avoidance wouldn't work either, since if the rover has moved significantly from
    # the original calculated origin point, the plotted path to the destination may no longer be valid




    # print("front box ", front_box)
    # print("left_box", Rover.left_obstacle)
    # print("upper_left_box", Rover.upper_left_obstacle)

    # 9) construct a path to follow based on mapped wall to the left. Take the average. Shift the obstacle points from
    # the upper left hand corner into the front of the rover. Get the average position of obstacle pixels in the upper
    # left hand corner, and project it in front of the rover to follow.

    # next_point = np.mean()

    # left_wall, right_wall = get_side_wall_outlines(navigable_xpix, navigable_ypix)

    # left_wall = np.concatenate((left_wall, right_wall), axis=0)

    # if left_wall.size > 10:
    #     follow_path = left_wall[:, :] - np.asarray([[0, 2.5]])
    #     next_follow_point = follow_path[0, :]
    #
    #     if next_follow_point[0] <= 25:
    #         Rover.left_wall_continues = True
    #         radians_to_next_follow_point = np.arctan2(next_follow_point[1], next_follow_point[0])
    #         Rover.angle_to_next_follow_point = radians_to_next_follow_point * 180 / np.pi
    #
    #     elif (Rover.front_navigable is True) and (Rover.left_obstacle is False) and (Rover.upper_left_obstacle is False):
    #         Rover.angle_to_next_follow_point = 15
    #         Rover.left_wall_continues = False
    #
    #     else:
    #         Rover.angle_to_next_follow_point = np.mean(angles) * (180/np.pi)
    #         Rover.left_wall_continues = False
    # else:
    # Rover.angle_to_next_follow_point = np.mean(angles) * (180 / np.pi)
    # Rover.angle_to_next_follow_point = (np.mean(-obstacle_angles) / 2) * (180 / np.pi)

    # print("next follow point", next_follow_point)
    # nearest_wall_point = left_wall[10, :]
    # next_wall_point = left_wall[1, :]

    # if Rover.wall_crawl_status == "wall_follow":
    #     pass

    # angle_to_nearest_wall_point = np.arctan(nearest_wall_point[1] - next_wall_point[1])/(nearest_wall_point[0] - next_wall_point[0])
    # angle_to_nearest_wall_point = (np.arctan(nearest_wall_point[1])/(nearest_wall_point[0]))

    # print("nearest angle", str(angle_to_nearest_wall_point))
    #
    # steering = np.clip(angle_to_nearest_wall_point, -15, 15)
    #
    # Rover.steer = angle_to_nearest_wall_point # steering


    # elif Rover.wall_crawl_status == "wall_follow" and (not left_wall.size):
    #     print("not working")
    #     # Rover.brake = 1
    #     Rover.throttle = 0
    #     Rover.steer = 15
    # elif not left_wall.size:
    #     Rover.wall_available = False
    # print("left wall size", left_wall.size)
    # print("front is navigable ", Rover.front_navigable)
    # print("front is obstacle ", Rover.front_navigable)
    # print("left wall obstacle ", Rover.left_obstacle)
    # print("upper left obstacle", Rover.upper_left_obstacle)
    # print("upper right obstacle", Rover.upper_right_obstacle)
    #
    # print("Rover mode ", Rover.mode)

    # if rock_sample_xpix.size and rock_sample_ypix.size:
    #     angle_to_rock = path_to_visible_course((rock_sample_xpix, rock_sample_ypix))
    #     Rover.angle_to_rock = angle_to_rock
    # else:
    #     Rover.angle_to_rock = 0



    return Rover
