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
    """

    :param destination_angle: value in degree
    :param yaw: value in degrees
    :return: misalignment in degrees
    """

    angle_difference = destination_angle - yaw
    misalignment = (angle_difference + 180) % 360 - 180

    return misalignment


def get_surrounding_values(x_pixel, y_pixel, map_data, radius=1):
    """
    Identify and return a pixel's value along with the values of its surrounding pixels
    :param x_pixel:
    :param y_pixel:
    :param map_data: 2 dimensional map that holds the pixel and the values
    :param radius: how many pixels from teh central pixel are we going to fetch the values of
    :return: 2 dimensional array with (x_pixel, y_pixel) at the middle
    """
    assert np.ndim(map_data) == 2
    surrounding_pixels = map_data[(y_pixel - radius):(y_pixel + radius + 1), (x_pixel - radius): (x_pixel + radius + 1)]
    return surrounding_pixels


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

    x_points = unexplored_point_indices[0]
    y_points = unexplored_point_indices[1]

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


def get_range_to_iterate_over(origin_x, origin_y, destination_x, destination_y, angle, granularity):
    """
    angle: in radians
    """
    # set the range to begin from the lowest x value to the highest
    range_start_x = min(origin_x, destination_x)
    range_end_x = max(origin_x, destination_x)

    if (0 <= abs(angle) <= (np.pi / 2)) or (((3 * np.pi) / 4) < abs(angle) < (np.pi * 2)):
        # print(" quadrant I or IV")
        # check from left to right
        range_to_iterate_over_x = np.arange(range_start_x, range_end_x, granularity)
    elif (np.pi / 2) < abs(angle) <= (
        (3 * np.pi) / 4):  # or (((3 * np.pi)/ (np.pi * 2)) < abs(angle) <= (3 * np.pi)/ (np.pi * 4)):
        # print("quadrant II or III")
        # if the angle is more than 90 degrees, x should be from right to left
        range_to_iterate_over_x = np.arange(range_start_x, range_end_x, granularity)[::-1]

    # do the same for y values
    range_start_y = min(origin_y, destination_y)
    range_end_y = max(origin_y, destination_y)

    # if the angle is in the 2 upper quadrants of the cartesian plane
    if (0 <= angle <= np.pi) or (-np.pi <= angle <= -np.pi * 2):
        # print("quadrant I or II")
        # check from bottom to top
        range_to_iterate_over_y = np.arange(range_start_y, range_end_y, granularity)
    # if the anlge is in the 2 bottom quadrants of the cartesian plane
    elif (np.pi < angle < np.pi * 2) or (-np.pi < angle < 0):
        # print("quadrant III or IV")
        # check from top to bottom
        range_to_iterate_over_y = np.arange(range_start_y, range_end_y, granularity)[::-1]

    return range_to_iterate_over_x, range_to_iterate_over_y


def obstacle_crossed_by_line(origin_x, origin_y, destination_x, destination_y, map_data, flag_list, granularity=1,
                             line_width=0, return_all=False):
    """
    x_points: should be divisible by the granularity value, otherwise, this function won't detect it. This function
    can only detect coordinate's whose x values are divisible by the granularity value
    map_data: should be a 2 dimensional array indicating which areas are obstacles and not
    line_width: TODO find all points traversed by a line with thickness of line_width
    return:
    :param origin_x:
    :param origin_y:
    :param destination_x:
    :param destination_y:
    :param map_data:
    :param flag_list: List of integers that flag obstacles (or areas to avoid) in the map_data
    :param granularity:
    :param line_width:
    :param return_all: if True, all crossed obstacle coords are returned, otherwise, just the first one is returned
    :return: list of (x,y) tuples representing crossed obstacle coords
    """

    assert np.ndim(map_data) == 2, "map_data is not 2 dimensional"

    # "draw" the line by getting its different elements

    # we add a small amount to the x and y diffs to avoid operations on zero values
    x_diff = destination_x - float(origin_x) + .000001  # convert one of the numbers into float so that we can have more
    y_diff = destination_y - float(origin_y) + .000001  # accurate computations, with no rounding off

    slope = y_diff / x_diff

    y_intercept = origin_y - (slope * origin_x)

    distance, angle = to_polar_coords_with_origin(origin_x, origin_y, destination_x, destination_y)

    range_to_iterate_over_x, range_to_iterate_over_y = get_range_to_iterate_over(origin_x, origin_y, destination_x,
                                                                                 destination_y, angle, granularity)

    # if x y coords are given, check each x, y coordinate pairs from x_points y_points if they are on the line
    if return_all is True:  # run the function until all flagged coordinates that cross the line are returned
        crossed_flagged_coords_x = []
        crossed_flagged_coords_y = []

        # check if any of the x values between the origin and destination have y_values that are obstacles
        for x in range_to_iterate_over_x:
            y = (slope * x) + y_intercept

            # round up and down because numpy only accepts integers when accessing array values
            # speaking of which, it may not be possible to have a granularity that is less than one
            y_up = np.ceil(y)
            y_down = np.floor(y)
            for flag in flag_list:
                if (map_data[int(y_up), int(x)] == flag):
                    crossed_flagged_coords_x.append((int(x), int(y)))
                elif (map_data[int(y_down), int(x)] == flag):
                    crossed_flagged_coords_x.append((int(x), int(y)))
        # do the same thing for y
        for y in range_to_iterate_over_y:
            x = (y - y_intercept) / slope
            x_left = np.floor(x)
            x_right = np.ceil(x)
            for flag in flag_list:
                if map_data[int(y), int(x_left)] == flag:
                    crossed_flagged_coords_y.append((int(x_left), int(y)))
                if map_data[int(y), int(x_right)] == flag:
                    crossed_flagged_coords_y.append((int(x_right), int(y)))

        # combine the crossed x and y coords with each other
        # https://stackoverflow.com/a/1319353
        crossed_flagged_coords = crossed_flagged_coords_x + list(
            set(crossed_flagged_coords_y) - set(crossed_flagged_coords_x))
        # no need to proceed with the rest of the function
        return crossed_flagged_coords
    else:  # just return the first x or y obstacle that is encountered
        # for each flag in the list, check if they are found on the line
        first_obstacle_x = None
        first_obstacle_y = None
        for x in range_to_iterate_over_x:
            # if we already have a first obstacle, do not prceed with the loop
            if first_obstacle_x is not None:
                break
            y = (slope * x) + y_intercept
            # round up and down because numpy only accepts integers when accessing array values
            # speaking of which, it may not be possible to have a granularity that is less than one
            y_up = np.ceil(y)
            y_down = np.floor(y)
            for flag in flag_list:
                if (map_data[int(y_up), int(x)] == flag):
                    first_obstacle_x = (x, int(y_up))
                    break
                elif (map_data[int(y_down), int(x)] == flag):
                    first_obstacle_x = (x, int(y_down))
                    break

        # do the same thing for y
        for y in range_to_iterate_over_y:
            # if first obstacle y is not none, break the loop
            if first_obstacle_y is not None:
                break
            x = (y - y_intercept) / slope
            x_left = np.floor(x)
            x_right = np.ceil(x)
            for flag in flag_list:
                if map_data[int(y), int(x_left)] == flag:
                    first_obstacle_y = (int(x_left), int(y))
                    break
                if map_data[int(y), int(x_right)] == flag:
                    first_obstacle_y = (int(x_right), int(y))
                    break
        if first_obstacle_x and first_obstacle_y:
            # compute which obstacle is closest
            first_x_distance = compute_distances(origin_x, origin_y, first_obstacle_x[0], first_obstacle_x[1])
            first_y_distance = compute_distances(origin_x, origin_y, first_obstacle_y[0], first_obstacle_y[1])
            # return the closest obstacle as a list
            if first_x_distance >= first_y_distance:
                return [first_obstacle_y]
            elif first_y_distance > first_x_distance:
                return [first_obstacle_x]
        elif first_obstacle_x:
            return [first_obstacle_x]
        elif first_obstacle_y:
            return [first_obstacle_y]

    return False


def sidestep_obstacle(origin_x, origin_y, destination_x, destination_y, map_data, navigable_flag, obstacle_flag):
    """
    This function takes origin coordinates and destination coordinates, and plots a path to the destination in two steps
    or two lines uninterrupted by obstacles based on the map_data
    :param origin_x:
    :param origin_y:
    :param destination_x:
    :param destination_y:
    :param map_data: 2 dimensional array flagged with indices representing nature of their coordinates
    :param navigable_flag: int used when on map_data when flagging navigable pixels
    :param obstacle_flag: int used when on map_data when flagging obstacle pixels
    :return: a Path_guide named tuple that contains the following values:
        ["midpoint_x", "midpoint_y", "midpoint_distance", "midpoint_angle", "destination_distance","destination_angle"]
        midpoint_x: x coordinate of the midpoint
        midpoint_y: y coordinate of the midpoint
        midpoint_distance: distance from the origin to the midpoint
        midpoint_angle: angle from teh origin to the midpoint
        destination_distance: distance from the midpoint to the destination
        destination_angle: angle from the midoint to the destination
    """
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
                                    ["midpoint_x", "midpoint_y", "midpoint_distance", "midpoint_angle",
                                     "destination_distance",
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


def find_waypoint(origin_x, origin_y, destination_x, destination_y, map_data, navigable_flag, obstacle_flag):
    assert np.ndim(map_data) == 2

    # compute the distance between origin and all navigable points, and all navigable points to destination point
    navigable_points = np.where(map_data == navigable_flag)

    distance_origin_to_midpoint = compute_distances(origin_x, origin_y, navigable_points[1], navigable_points[0])
    distance_midpoint_to_destination = compute_distances(navigable_points[1], navigable_points[0], destination_x,
                                                         destination_y)

    combined_distance = distance_origin_to_midpoint + distance_midpoint_to_destination

    # find the shortest distance, but remember its original index
    original_indices = np.argsort(combined_distance)

    complete_path = []
    midpoint_path_only = []

    Path_guide = namedtuple("Path_guide",
                            ["midpoint_x", "midpoint_y", "midpoint_distance", "midpoint_angle",
                             "destination_distance",
                             "destination_angle"])

    for original_index in original_indices:
        midpoint_x = np.array(navigable_points)[1, original_index]
        midpoint_y = np.array(navigable_points)[0, original_index]
        # check if either of the two paths are blocked
        obstacle_crossed_part_1 = obstacle_crossed_by_line(origin_x, origin_y, midpoint_x, midpoint_y, map_data,
                                                           [obstacle_flag])
        obstacle_crossed_part_2 = obstacle_crossed_by_line(midpoint_x, midpoint_y, destination_x, destination_y,
                                                           map_data, [obstacle_flag])

        if (obstacle_crossed_part_1 is False) and (obstacle_crossed_part_2 is False):

            midpoint_distance, midpoint_angle = to_polar_coords_with_origin(origin_x, origin_y, midpoint_x, midpoint_y)
            destination_distance, destination_angle = to_polar_coords_with_origin(midpoint_x, midpoint_y, destination_x,
                                                                                  destination_y)
            path_guide = Path_guide(midpoint_x, midpoint_y, midpoint_distance, midpoint_angle, destination_distance,
                                    destination_angle)

            complete_path.append(path_guide)
        elif obstacle_crossed_part_1 is False:
            midpoint_distance, midpoint_angle = to_polar_coords_with_origin(origin_x, origin_y, midpoint_x, midpoint_y)
            path_guide = Path_guide(midpoint_x, midpoint_y, midpoint_distance, midpoint_angle, None,
                                    None)
            midpoint_path_only.append(path_guide)

    if complete_path:
        # TODO sort the paths instead of just returning the first one
        path_guide = complete_path[0]
    elif midpoint_path_only:
        path_guide = midpoint_path_only[0]
    else:
        path_guide = None
    return path_guide