import numpy as np
from collections import namedtuple


def to_polar_coords_with_origin(origin_x, origin_y, x_pixels, y_pixels):
    y_diffs = y_pixels - float(origin_y)
    x_diffs = x_pixels - float(origin_x)
    # Calculate the distances between the origin and the pixels
    dist = np.sqrt((y_diffs ** 2) + (x_diffs ** 2))
    angles = np.arctan2(y_diffs, x_diffs)
    return dist, angles


def compute_distances(origin_x, origin_y, x_points, y_points):
    y_diffs = y_points - origin_y
    x_diffs = x_points - origin_x
    # Calculate the distances between the origin and the pixels
    distances = np.sqrt((y_diffs ** 2) + (x_diffs ** 2))

    return distances


# adapted from https://stackoverflow.com/a/7869457
def compute_misalignment(destination_angle, yaw):

    angle_difference = destination_angle - yaw
    misalignment = (angle_difference + 180) % 360 - 180

    return misalignment


def choose_destination(origin_xpos, origin_ypos, map_data, minimum_distance=0):
    """
    This function returns a memory_map coordinate. The initial destination is normally chosen to be
    the nearest unexplored (untagged) point on the memory_map

    This may trigger the rover to first rotate 360 degrees upon initialization to survey what's in sight.
    TODO allow the rover to choose as a destination any of the (navigable) areas it has already explored

    :param origin_xpos:
    :param origin_ypos:
    :param map_data:
    :param minimum_distance: Specify a minimum distance. This is to prevent the rover from constantly
    stopping every 1 or 2 pixels after achieving its target destination
    :return: returns an (x,y) tuple of 2 integers. These is the destination's xy coordinates in the
    memory _ap.  Rover.memory_map is different from Rover.worldmap.  Rover.memory_map is 10 times
    larger (2000 x 2000 )and has more layers
    """

    assert map_data.ndim == 2, " map does not have 2 dimensions "

    unexplored_point_indices = np.where(map_data == 0)

    # The first array (index 0) are the y values.  The second one (index 1) are the x values
    x_points = unexplored_point_indices[1]
    y_points = unexplored_point_indices[0]

    distances, angles = to_polar_coords_with_origin(origin_xpos, origin_ypos, x_points, y_points)

    # Get the argmin values given a condition
    # https://seanlaw.github.io/2015/09/10/numpy-argmin-with-a-condition/
    mask = (distances >= minimum_distance)
    subset_idx = np.argmin(distances[mask])
    parent_idx = np.arange(distances.shape[0])[mask][subset_idx]

    distance_min_idx = parent_idx
    min_distance = distances[distance_min_idx]
    accompanying_angle = angles[distance_min_idx]
    x_point = x_points[distance_min_idx]
    y_point = y_points[distance_min_idx]

    chosen_destination_coords = (int(x_point), int(y_point))
    chosen_destination_distance = min_distance
    chosen_destination_angle = accompanying_angle

    x_diff = x_point - float(origin_xpos)
    y_diff = y_point - float(origin_ypos)

    # logging.debug("min distance " + str(min_distance))
    # logging.debug("x_diff " + str(x_diff))
    # logging.debug("y-diff " + str(y_diff))
    # logging.debug("dist ** 2 " + str((min_distance ** 2)))
    # logging.debug("x_diff ** 2 + y_diff **2 " + str((x_diff ** 2) + (y_diff ** 2)))
    # logging.debug("accompanying_ angle " + str(accompanying_angle))
    # logging.debug("np.arctan2(float(y_diff), x_diff) " + str(np.arctan2(float(y_diff), x_diff)))

    c_squared = min_distance ** 2
    a_squared = (x_diff ** 2)
    b_squared = (y_diff ** 2)
    assert np.isclose(c_squared, a_squared + b_squared, rtol=1e-05, atol=1e-08, equal_nan=False)
    assert np.isclose(accompanying_angle, np.arctan2(float(y_diff), x_diff))

    return chosen_destination_coords, chosen_destination_distance, chosen_destination_angle


def obstacle_crossed_by_line(origin_x, origin_y, destination_x, destination_y, map_data, flag_list, granularity=1,
                             line_width=0, return_all=False):
    """
    x_points: should be divisible by the granularity value, otherwise, this function won't detect it. This function
    can only detect coordinate's whose x values are divisible by the granularity value
    map_data: should be a 2 dimensional array indicating which areas are obstacles and not
    line_width: TODO find all points traversed by a line with thickness of line_width
    return: list of (x,y) tuples
    """
    # "draw" the line by getting its different elements
    x_diff = destination_x - float(origin_x)  # convert one of the numbers into float so that we can have more
    y_diff = destination_y - float(origin_y)  # accurate computations, with no rounding off

    slope = y_diff / x_diff

    y_intercept = origin_y - (slope * origin_x)

    distance, angle = to_polar_coords_with_origin(origin_x, origin_y, destination_x, destination_y)

    # set the range to begin from the lowest x value to the highest
    range_start = min(origin_x, destination_x)
    range_end = max(origin_x, destination_x)

    if (0 <= abs(angle) <= np.pi / 2) or (((3 * np.pi) / (np.pi * 2)) < abs(angle) < (np.pi * 2)):
        # check from left to right
        range_to_iterate_over = np.arange(range_start, range_end, granularity)
    elif (np.pi / 2 < abs(angle) <= np.pi) or (np.pi < abs(angle) <= (3 * np.pi) / (np.pi * 2)):
        # if the angle is more than 90 degrees, x should be from right to left
        range_to_iterate_over = np.arange(range_start, range_end, granularity)[::-1]

    # if x y coords are given, check each x, y coordinate pairs from x_points y_points if they are on the line
    if return_all is True:  # run the function until all flagged coordinates that cross the line are returned
        crossed_flagged_coords = []
        for x in range_to_iterate_over:
            y = (slope * x) + y_intercept

            # round up and down because numpy only accepts integers when accessing array values
            # speaking of which, it may not be possible to have a granularity that is less than one
            y_up = np.ceil(y)
            y_down = np.floor(y)
            for flag in flag_list:
                if map_data[int(y_up), int(x)] == flag:
                    crossed_flagged_coords.append((int(x), int(y)))
                elif map_data[int(y_down), int(x)] == flag:
                    crossed_flagged_coords.append((int(x), int(y)))
        # no need to proceed with the rest of the function
        return crossed_flagged_coords
    else:  # just return the first obstacle that
        # for each flag in the list, check if they are found on the line
        for x in range_to_iterate_over:
            y = (slope * x) + y_intercept

            # round up and down because numpy only accepts integers when accessing array values
            # speaking of which, it may not be possible to have a granularity that is less than one
            y_up = np.ceil(y)
            y_down = np.floor(y)
            for flag in flag_list:
                if map_data[int(y_up), int(x)] == flag:
                    return [(x, int(y_up))]
                elif map_data[int(y_down), int(x)] == flag:
                    return [(x, int(y_down))]

    return False


def sidestep_obstacle(origin_x, origin_y, destination_x, destination_y, map_data, navigable_flag, obstacle_flag):
    assert np.ndim(map_data) == 2

    # compute the distance between origin and all navigable points, and all navigable points to destinaion point
    navigable_points = np.where(map_data == navigable_flag)

    distance_origin_to_midpoint = compute_distances(origin_x, origin_y, navigable_points[1], navigable_points[0])
    distance_midpoint_to_destination = compute_distances(navigable_points[1], navigable_points[0], destination_x,
                                                         destination_y)

    combined_distance = distance_origin_to_midpoint + distance_midpoint_to_destination

    # find the shortest distance, but remember its original index

    original_indices = np.argsort(combined_distance)

    for original_index in original_indices:
        midpoint_x = np.array(navigable_points)[1, original_index]
        midpoint_y = np.array(navigable_points)[0, original_index]
        # check if either of the two paths are blocked
        obstacle_crossed_part_1 = obstacle_crossed_by_line(origin_x, origin_y, midpoint_x, midpoint_y, map_data,
                                                           [obstacle_flag])
        if obstacle_crossed_part_1 is not False:
            # if the path was blocked
            continue
        obstacle_crossed_part_2 = obstacle_crossed_by_line(midpoint_x, midpoint_y, destination_x, destination_y,
                                                           map_data, [obstacle_flag])
        if obstacle_crossed_part_2 is not False:
            continue

        # if not blocked, compute the polar coordinates to be sent to the rover
        if (obstacle_crossed_part_1 is False) and (obstacle_crossed_part_2 is False):
            Path_guide = namedtuple("Path_guide",
                                    ["x", "y", "midpoint_distance", "midpoint_angle", "destination_distance",
                                     "destination_angle"])
            midpoint_distance, midpoint_angle = to_polar_coords_with_origin(origin_x, origin_y, midpoint_x, midpoint_y)
            destination_distance, destination_angle = to_polar_coords_with_origin(midpoint_x, midpoint_y, destination_x,
                                                                                  destination_y)
            path_guide = Path_guide(midpoint_x, midpoint_y, midpoint_distance, midpoint_angle, destination_distance,
                                    destination_angle)
            # if both are clear, then return the path guide
            return path_guide
    # if there are no clear paths, then return False
    return False
