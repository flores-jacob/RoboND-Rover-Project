import numpy as np
from collections import namedtuple

from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder


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


def coordinates_reached(current_coordinates, target_coordinates):
    # If the x and y values of the current position rounded up or down are equivalent to
    # the destination point:

    x_range = range(round(current_coordinates[0]) - 1, round(current_coordinates[0]) + 2)
    y_range = range(round(current_coordinates[1]) - 1, round(current_coordinates[1]) + 2)

    dest_x_reached = target_coordinates[0] in x_range
    dest_y_reached = target_coordinates[1] in y_range

    if dest_x_reached and dest_y_reached:
        print("destinaton reached ")
        return True
    else:
        return False


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


def choose_closest_flag(origin_x, origin_y, map_data, minimum_distance=0, flag=0, x_lower_bound=None,
                        x_upper_bound=None, y_lower_bound=None, y_upper_bound=None):
    """
    This function returns a memory_map coordinate. The initial destination is normally chosen to be
    the nearest unexplored (untagged) point on the memory_map

    This may trigger the rover to first rotate 360 degrees upon initialization to survey what's in sight.

    :return: returns an (x,y) tuple of 2 integers. These is the destination's xy coordinates in the
    memory _ap.  Rover.memory_map is different from Rover.worldmap.  Rover.memory_map is 10 times
    larger (2000 x 2000 )and has more layers

    TODO allow the rover to choose as a destination any of the (navigable) areas it has already explored

    """

    assert map_data.ndim == 2, " map does not have 2 dimensions "

    if x_lower_bound is None:
        x_lower_bound = 0
    if x_upper_bound is None:
        x_upper_bound = map_data.shape[1]

    if y_lower_bound is None:
        y_lower_bound = 0
    if y_upper_bound is None:
        y_upper_bound = map_data.shape[0]

    flag_point_indices = np.where(map_data[y_lower_bound:y_upper_bound, x_lower_bound:x_upper_bound] == flag)

    x_points = flag_point_indices[1] + x_lower_bound
    y_points = flag_point_indices[0] + y_lower_bound

    distances, angles = to_polar_coords_with_origin(origin_x, origin_y, x_points, y_points)

    # Get the argmin values given a condition
    # https://seanlaw.github.io/2015/09/10/numpy-argmin-with-a-condition/
    mask = (distances >= minimum_distance)
    subset_idx = np.argmin(distances[mask])
    parent_idx = np.arange(distances.shape[0])[mask][subset_idx]

    # distance_min_idx = np.argmin(distances)
    distance_min_idx = parent_idx
    min_distance = distances[distance_min_idx]
    accompanying_angle = angles[distance_min_idx]

    x_point = x_points[distance_min_idx]
    y_point = y_points[distance_min_idx]

    chosen_destination_coords = (int(x_point), int(y_point))
    chosen_destination_distance = min_distance
    chosen_destination_angle = accompanying_angle

    x_diff = x_point - float(origin_x)
    y_diff = y_point - float(origin_y)

    # logging.debug("min distance " + str(min_distance))
    # logging.debug("x_diff " + str(x_diff))
    # logging.debug("y-diff " + str(y_diff))
    # logging.debug("dist ** 2 " + str((min_distance ** 2)))
    # logging.debug("x_diff ** 2 + y_diff **2 " + str((x_diff ** 2) + (y_diff ** 2)))
    # logging.debug("accompanying_ angle " + str(accompanying_angle))
    # logging.debug("np.arctan2(float(y_diff), x_diff) " + str(np.arctan2(float(y_diff), x_diff)))

    #     assert (float(min_distance ** 2) == float((x_diff ** 2) + (y_diff ** 2)))
    c_squared = min_distance ** 2
    a_squared = (x_diff ** 2)
    b_squared = (y_diff ** 2)
    assert np.isclose(c_squared, a_squared + b_squared, rtol=1e-05, atol=1e-08, equal_nan=False)
    assert np.isclose((accompanying_angle), np.arctan2(float(y_diff), x_diff))

    return chosen_destination_coords, chosen_destination_distance, chosen_destination_angle


def choose_farthest_flag(origin_x, origin_y, map_data, maximum_distance=None, flag=0, x_lower_bound=None,
                         x_upper_bound=None, y_lower_bound=None, y_upper_bound=None):
    assert map_data.ndim == 2, " map does not have 2 dimensions "

    if x_lower_bound is None:
        x_lower_bound = 0
    if x_upper_bound is None:
        x_upper_bound = map_data.shape[1]

    if y_lower_bound is None:
        y_lower_bound = 0
    if y_upper_bound is None:
        y_upper_bound = map_data.shape[0]

    flag_point_indices = np.where(map_data[y_lower_bound:y_upper_bound, x_lower_bound:x_upper_bound] == flag)

    x_points = flag_point_indices[1] + x_lower_bound
    y_points = flag_point_indices[0] + y_lower_bound

    distances, angles = to_polar_coords_with_origin(origin_x, origin_y, x_points, y_points)

    # if there were no points found, return None
    if not distances.size:
        return None, None, None

    if maximum_distance:
        # Get the argmin values given a condition
        # https://seanlaw.github.io/2015/09/10/numpy-argmin-with-a-condition/
        mask = (distances <= maximum_distance)
        subset_idx = np.argmax(distances[mask])
        parent_idx = np.arange(distances.shape[0])[mask][subset_idx]
    else:
        subset_idx = np.argmax(distances)
        parent_idx = np.arange(distances.shape[0])[subset_idx]

    # distance_min_idx = np.argmin(distances)
    distance_max_idx = parent_idx
    max_distance = distances[distance_max_idx]
    accompanying_angle = angles[distance_max_idx]

    x_point = x_points[distance_max_idx]
    y_point = y_points[distance_max_idx]

    chosen_destination_coords = (int(x_point), int(y_point))
    chosen_destination_distance = max_distance
    chosen_destination_angle = accompanying_angle

    x_diff = x_point - float(origin_x)
    y_diff = y_point - float(origin_y)

    c_squared = max_distance ** 2
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
    assert (np.ndim(map_data) == 2)

    #     print("map_data_size: ", map_data.size)

    #     print("destination x", destination_x)
    #     print("desitnation y", destination_y)
    #     print("this is the map data originally\n", map_data)
    #     print("this is the x y value for map_data", map_data[destination_y, destination_x])

    #     assert(map_data[destination_y, destination_x] == 0, "tag is " + str(map_data[destination_y, destination_x]) + " instead")

    # "draw" the line by getting its different elements

    # we add a small amount to the x and y diffs to avoid operations on zero values
    x_diff = destination_x - float(origin_x) + .000001  # convert one of the numbers into float so that we can have more
    y_diff = destination_y - float(origin_y) + .000001  # accurate computations, with no rounding off

    slope = y_diff / x_diff

    y_intercept = origin_y - (slope * origin_x)

    distance, angle = to_polar_coords_with_origin(origin_x, origin_y, destination_x, destination_y)

    range_to_iterate_over_x, range_to_iterate_over_y = get_range_to_iterate_over(int(origin_x), int(origin_y),
                                                                                 destination_x,
                                                                                 destination_y, angle, granularity)

    # if x y coords are given, check each x, y coordinate pairs from x_points y_points if they are on the line
    #     print("outside ")
    if return_all is True:  # run the function until all flagged coordinates that cross the line are returned
        #         print("return all is true ")
        crossed_flagged_coords_x = []
        crossed_flagged_coords_y = []

        # check if any of the x values between the origin and destination have y_values that are obstacles
        for x in range_to_iterate_over_x:
            #             ("itrating over ranges ")
            y = (slope * x) + y_intercept

            # round up and down because numpy only accepts integers when accessing array values
            # speaking of which, it may not be possible to have a granularity that is less than one
            y_up = min(np.ceil(y), map_data.shape[0])
            y_down = max(np.floor(y), 0)
            for flag in flag_list:
                if (map_data[int(y_up), int(x)] == flag):
                    crossed_flagged_coords_x.append((int(x), int(y)))
                elif (map_data[int(y_down), int(x)] == flag):
                    crossed_flagged_coords_x.append((int(x), int(y)))
        # do the same thing for y
        for y in range_to_iterate_over_y:
            x = (y - y_intercept) / slope
            x_left = max(np.floor(x), 0)
            x_right = min(np.ceil(x), map_data.shape[1])
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
        #         print(crossed_flagged_coords)
        return crossed_flagged_coords
    else:  # just return the first x or y obstacle that is encountered
        # for each flag in the list, check if they are found on the line
        #         print("entering ")

        first_obstacle_x = None
        first_obstacle_y = None
        #         print("range_to_iterate_over_y", range_to_iterate_over_y)
        #         print("range_to_iterate_over_x", range_to_iterate_over_x)
        for x in range_to_iterate_over_x:
            #             print("iterating x ")
            # if we already have a first obstacle, do not prceed with the loop
            if first_obstacle_x is not None:
                break
                #             "reaching this space x"
                #             print("x is ", x)
                #             print("slope", slope)
                #             print("intercept ", y_intercept)
            y = (slope * x) + y_intercept
            #             print("y_result ", y)

            # round up and down because numpy only accepts integers when accessing array values
            # speaking of which, it may not be possible to have a granularity that is less than one
            y_up = min(np.ceil(y), map_data.shape[0])
            y_down = max(np.floor(y), 0)
            for flag in flag_list:
                if (map_data[int(y_up), int(x)] == flag):
                    first_obstacle_x = (x, int(y_up))
                    break
                elif (map_data[int(y_down), int(x)] == flag):
                    first_obstacle_x = (x, int(y_down))
                    break

        # do the same thing for y
        for y in range_to_iterate_over_y:
            #             print("iterating y", range_to_iterate_over_y)
            # if first obstacle y is not none, break the loop
            if first_obstacle_y is not None:
                break
                #             print("reaching this space y")
            x = (y - y_intercept) / slope

            #             print("y source ", y)
            #             print("y_inrercept ", y_intercept)
            #             print("slope ", slope)
            #             print("x result ", x)
            x_left = max(np.floor(x), 0)
            x_right = min(np.ceil(x), map_data.shape[1])
            for flag in flag_list:
                if (map_data[int(y), int(x_left)] == flag):
                    first_obstacle_y = (int(x_left), int(y))
                    break
                elif (map_data[int(y), int(x_right)].astype(int) == flag):
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

            #         print("first obstacle x ", first_obstacle_x)
            #         print("first obstacle y ", first_obstacle_y)
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


def get_optimal_midpoint(origin_x, origin_y, destination_x, destination_y, map_data, navigable_flag, obstacle_flag):
    # compute the distance between origin and all navigable points, and all navigable points to destination point
    navigable_points = np.where(map_data == navigable_flag)
    print("navigable len ", len(navigable_points))

    distance_origin_to_midpoint = compute_distances(origin_x, origin_y, navigable_points[1], navigable_points[0])
    print("len distance origin to midpoint ", len(distance_origin_to_midpoint))
    distance_midpoint_to_destination = compute_distances(navigable_points[1], navigable_points[0], destination_x,
                                                         destination_y)
    print("len distance midpoint to destination", len(distance_midpoint_to_destination))

    combined_distance = distance_origin_to_midpoint + distance_midpoint_to_destination

    print("combined distance ", len(combined_distance))

    # find the shortest distance, but remember its original index
    index_with_shortest_distance = np.argmin(combined_distance)
    if index_with_shortest_distance:
        midpoint_x = np.array(navigable_points)[1, index_with_shortest_distance]
        midpoint_y = np.array(navigable_points)[0, index_with_shortest_distance]
        return (midpoint_x, midpoint_y)
    else:
        return None


def find_waypoint(origin_x, origin_y, destination_x, destination_y, map_data, navigable_flag, obstacle_flag):
    assert np.ndim(map_data) == 2

    # compute the distance between origin and all navigable points, and all navigable points to destination point
    navigable_points = np.where(map_data == navigable_flag)
    # print("navigable len ", len(navigable_points))

    distance_origin_to_midpoint = compute_distances(origin_x, origin_y, navigable_points[1], navigable_points[0])
    # print("len distance origin to midpoint ", len(distance_origin_to_midpoint))
    distance_midpoint_to_destination = compute_distances(navigable_points[1], navigable_points[0], destination_x,
                                                         destination_y)
    # print("len distance midpoint to destination", len(distance_midpoint_to_destination))

    combined_distance = distance_origin_to_midpoint + distance_midpoint_to_destination

    # print("combined distance ", len(combined_distance))

    # find the shortest distance, but remember its original index
    original_indices = np.argsort(combined_distance)

    complete_path = []
    midpoint_path_only = []

    Path_guide = namedtuple("Path_guide",
                            ["midpoint_x", "midpoint_y", "midpoint_distance", "midpoint_angle",
                             "destination_distance",
                             "destination_angle"])

    index = 0
    for original_index in original_indices:
        index += 1
        # print("iterating through index ", index)
        midpoint_x = np.array(navigable_points)[1, original_index]
        midpoint_y = np.array(navigable_points)[0, original_index]
        # check if either of the two paths are blocked
        obstacle_crossed_part_1 = obstacle_crossed_by_line(origin_x, origin_y, midpoint_x, midpoint_y, map_data,
                                                           [obstacle_flag])
        obstacle_crossed_part_2 = obstacle_crossed_by_line(midpoint_x, midpoint_y, destination_x, destination_y,
                                                           map_data, [obstacle_flag])

        if (obstacle_crossed_part_1 is False) and (obstacle_crossed_part_2 is False):
            # print("success1 and success2")
            midpoint_distance, midpoint_angle = to_polar_coords_with_origin(origin_x, origin_y, midpoint_x, midpoint_y)
            destination_distance, destination_angle = to_polar_coords_with_origin(midpoint_x, midpoint_y, destination_x,
                                                                                  destination_y)
            path_guide = Path_guide(midpoint_x, midpoint_y, midpoint_distance, midpoint_angle, destination_distance,
                                    destination_angle)

            complete_path.append(path_guide)
        elif obstacle_crossed_part_1 is False:
            # print("success1 only")
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


def choose_closest_unobstructed_point(origin_x, origin_y, map_data, flag_target=0, flag_obstruction=5,
                                      minimum_distance=0, x_lower_bound=None, x_upper_bound=None, y_lower_bound=None,
                                      y_upper_bound=None):
    assert map_data.ndim == 2, " map does not have 2 dimensions "

    if x_lower_bound is None:
        x_lower_bound = 0
    if x_upper_bound is None:
        x_upper_bound = map_data.shape[1]

    if y_lower_bound is None:
        y_lower_bound = 0
    if y_upper_bound is None:
        y_upper_bound = map_data.shape[0]

    # get all distances to flagged areas

    # get all the nav points in the desired area
    flag_point_indices = np.where(map_data[y_lower_bound:y_upper_bound, x_lower_bound:x_upper_bound] == flag_target)

    x_points = flag_point_indices[1] + x_lower_bound
    y_points = flag_point_indices[0] + y_lower_bound

    #     for index in x_points:
    #         print("should be zero ", map_data[y_points[index], x_points[index]])

    #     print("x_points ", len(x_points))
    #     print("y_points ", len(y_points))

    # compute the distances to them
    distances = compute_distances(origin_x, origin_y, x_points, y_points)

    # from lowest to highest distance, check if the path is obstructed or not

    #     result = obstacle_crossed_by_line(origin_x, origin_y, x_points, y_points, map_data, [flag_obstruction])

    #     distances = np.asarray([9,8,7,6,5,4,3,2,1])

    # Get the argmin values given a condition
    # https://seanlaw.github.io/2015/09/10/numpy-argmin-with-a-condition/
    mask = (distances >= minimum_distance)
    subset_idx = np.argsort(distances[mask])
    parent_idx = np.arange(distances.shape[0])[mask][subset_idx]

    closest_unobstructed_point_index = None
    # once we have sorted them by their distances, we check if each point is obstructed
    for index in parent_idx:
        #         print("this is the index ", index)
        #         print("x value ", x_points[index])
        #         print("y value ", y_points[index])
        #         print("map_data value ", map_data[y_points[index], x_points[index]])

        obstruction_present = obstacle_crossed_by_line(origin_x, origin_y, x_points[index], y_points[index], map_data,
                                                       [flag_obstruction], return_all=False)
        if obstruction_present:
            # if path is obstructed, do nothing
            #             print("this is the obstruction ", obstruction_present)
            pass
        # if there are no obstructions, use the current index
        elif obstruction_present is False:
            closest_unobstructed_point_index = index
            break

    # use the obtained index of the unobstructed point to get its x and y coordinates
    if closest_unobstructed_point_index:
        closest_unobstructed_x_point = x_points[closest_unobstructed_point_index]
        closest_unobstructed_y_point = y_points[closest_unobstructed_point_index]
        closest_unobstructed_point = (closest_unobstructed_x_point, closest_unobstructed_y_point)
    else:
        closest_unobstructed_point = None

    return closest_unobstructed_point


def get_closest_accessible_navigable_point_to_destination(origin_x, origin_y, destination_x, destination_y, map_data,
                                                          navigable_flag=7, obstacle_flag=5, minimum_distance=0):
    assert np.ndim(map_data) == 2, "map data does not have 2 dimensions"

    navigable_points = np.where(map_data[:, :] == navigable_flag)

    x_points = navigable_points[1]
    y_points = navigable_points[0]

    # compute the distances of the navigable_points to the destination_point
    distances = compute_distances(destination_x, destination_y, x_points, y_points)

    # from lowest to highest distance, check if the path is obstructed from the origin or not
    indices = np.argsort(distances)

    closest_unobstructed_point_index = None
    current_closest_distance = None
    # once we have sorted them by their distances, we check if each point is obstructed
    for index in indices:
        obstruction_present = obstacle_crossed_by_line(origin_x, origin_y, x_points[index], y_points[index], map_data,
                                                       [obstacle_flag], return_all=False)
        if obstruction_present:
            # if path is obstructed, do nothing
            #             print("this is the obstruction ", obstruction_present)
            pass
        # if there are no obstructions, use the current index
        elif obstruction_present is False:
            if current_closest_distance is None:
                current_closest_distance = distances[index]
                closest_unobstructed_point_index = index
            elif current_closest_distance > distances[index]:
                current_closest_distance = distances[index]
                closest_unobstructed_point_index = index

    # use the obtained index of the unobstructed point to get its x and y coordinates
    if closest_unobstructed_point_index:
        closest_unobstructed_x_point = x_points[closest_unobstructed_point_index]
        closest_unobstructed_y_point = y_points[closest_unobstructed_point_index]
        closest_unobstructed_point = (closest_unobstructed_x_point, closest_unobstructed_y_point)
    else:
        closest_unobstructed_point = None

    return closest_unobstructed_point


def get_nav_points_besides_unexplored_area(map_data, x_lower_bound=None, x_upper_bound=None, y_lower_bound=None,
                                           y_upper_bound=None):
    assert map_data.ndim == 2, " map does not have 2 dimensions "

    nav_flag = 7
    unexplored_flag = 0

    if x_lower_bound is None:
        x_lower_bound = 0
    if x_upper_bound is None:
        x_upper_bound = map_data.shape[1]

    if y_lower_bound is None:
        y_lower_bound = 0
    if y_upper_bound is None:
        y_upper_bound = map_data.shape[0]

    # get all the nav points in the desired area
    flag_point_indices = np.where(map_data[y_lower_bound:y_upper_bound, x_lower_bound:x_upper_bound] == nav_flag)

    x_points = flag_point_indices[1] + x_lower_bound
    y_points = flag_point_indices[0] + y_lower_bound

    unexplored_points = []

    # now that we have all the nav points, let's check each one if they are beside an unexplored pixel
    for index in range(x_points.size):
        x_coordinate = x_points[index]
        y_coordinate = y_points[index]

        top = map_data[y_coordinate + 1, x_coordinate]
        bottom = map_data[y_coordinate - 1, x_coordinate]
        left = map_data[y_coordinate, x_coordinate - 1]
        right = map_data[y_coordinate, x_coordinate + 1]

        # surrounding_pixels = map_data[y_coordinate - 1: y_coordinate + 2, x_coordinate - 1: x_coordinate + 2]
        if unexplored_flag in [top, bottom, left, right]:
            # if np.any(surrounding_pixels[:, :] == unexplored_flag):
            unexplored_points.append((x_coordinate, y_coordinate))
    # if there are no unexplored points beside nav points, return None
    return unexplored_points


def get_unexplored_points_besides_navigable_areas(map_data, x_lower_bound=None, x_upper_bound=None, y_lower_bound=None,
                                                  y_upper_bound=None, return_value="nav"):
    assert map_data.ndim == 2, " map does not have 2 dimensions "

    unexplored_flag = 0
    nav_flag = 7

    if x_lower_bound is None:
        x_lower_bound = 0
    if x_upper_bound is None:
        x_upper_bound = map_data.shape[1]

    if y_lower_bound is None:
        y_lower_bound = 0
    if y_upper_bound is None:
        y_upper_bound = map_data.shape[0]

    # get all the nav points in the desired area
    flag_point_indices = np.where(map_data[y_lower_bound:y_upper_bound, x_lower_bound:x_upper_bound] == unexplored_flag)

    x_points = flag_point_indices[1] + x_lower_bound
    y_points = flag_point_indices[0] + y_lower_bound

    unexplored_points = []
    nav_points = []

    # now that we have all the nav points, let's check each one if they are beside an unexplored pixel
    for index in range(x_points.size):
        x_coordinate = x_points[index]
        y_coordinate = y_points[index]

        top_x = x_coordinate
        top_y = min(y_coordinate + 1, y_upper_bound)
        top = map_data[top_y, top_x]

        bottom_x = x_coordinate
        bottom_y = max(y_coordinate - 1, y_lower_bound)
        bottom = map_data[bottom_y, bottom_x]

        left_x = max(x_coordinate - 1, x_lower_bound)
        left_y = y_coordinate
        left = map_data[left_y, left_x]

        right_x = min(x_coordinate + 1, x_upper_bound)
        right_y = y_coordinate
        right = map_data[right_y, right_x]

        if top == nav_flag:
            unexplored_points.append((x_coordinate, y_coordinate))
            nav_points.append((top_x, top_y))
        elif bottom == nav_flag:
            unexplored_points.append((x_coordinate, y_coordinate))
            nav_points.append((bottom_x, bottom_y))
        elif left == nav_flag:
            unexplored_points.append((x_coordinate, y_coordinate))
            nav_points.append((left_x, left_y))
        elif right == nav_flag:
            unexplored_points.append((x_coordinate, y_coordinate))
            nav_points.append((right_x, right_y))

            # if nav_flag in [top, bottom, left, right]:
            #     unexplored_points.append((x_coordinate, y_coordinate))

    # if there are no unexplored points beside nav points, return []
    if return_value == "nav":
        return nav_points
    elif return_value == "unexplored":
        return unexplored_points
    else:
        return nav_points, unexplored_points


def determine_quadrant(origin_x, origin_y, map_data):
    half_x = map_data.shape[1] / 2
    half_y = map_data.shape[0] / 2

    if (origin_x >= half_x) and (origin_y >= half_y):
        quadrant = 1
    elif (origin_x < half_x) and (origin_y >= half_y):
        quadrant = 2
    elif (origin_x < half_x) and (origin_y < half_y):
        quadrant = 3
    elif (origin_x >= half_x) and (origin_y < half_y):
        quadrant = 4
    else:
        raise Exception("unable to determine quadrant of coordinates")
    return quadrant


def get_coordinate_lower_and_upper_bounds(quadrant_number, map_data):
    half_x = int(map_data.shape[1] / 2)
    half_y = int(map_data.shape[0] / 2)

    full_x = map_data.shape[1]
    full_y = map_data.shape[0]
    if quadrant_number == 1:
        x_lower_bound = half_x
        x_upper_bound = full_x
        y_lower_bound = half_y
        y_upper_bound = full_y
    elif quadrant_number == 2:
        x_lower_bound = 0
        x_upper_bound = half_x
        y_lower_bound = half_y
        y_upper_bound = full_y
    elif quadrant_number == 3:
        x_lower_bound = 0
        x_upper_bound = half_x
        y_lower_bound = 0
        y_upper_bound = half_y
    elif quadrant_number == 4:
        x_lower_bound = half_x
        x_upper_bound = full_x
        y_lower_bound = 0
        y_upper_bound = half_y
    else:
        raise Exception("inappropriate quadrant input")

    return (x_lower_bound, x_upper_bound, y_lower_bound, y_upper_bound)


def get_new_target(rover):
    print("computing new target")

    # code to get new target point
    # if new target generated then continue
    # if new target fails to generate, assign start point as target. return
    # to the middle of the map such that we can start exploring other quadrants

    # if the start position has never been assigned, assign it
    if not rover.start_pos:
        rover.start_pos = (int(rover.pos[0]), int(rover.pos[1]))

    # initialize the first target quadrant to be the current quadrant the rover is in
    if not rover.target_quadrant:
        rover.target_quadrant = determine_quadrant(rover.pos[0], rover.pos[1],
                                                   rover.memory_map[:, :, 3])

    # get the x and y bounds for the current target quadrant
    rover_x_lower, rover_x_upper, rover_y_lower, rover_y_upper = get_coordinate_lower_and_upper_bounds(
        rover.target_quadrant, rover.memory_map[:, :, 3])

    # get all the unexplored points that are next to a nav point
    # unexplored_points_beside_nav_points = get_unexplored_points_besides_navigable_areas(rover.memory_map[:, :, 3],
    #                                                                                     x_lower_bound=rover_x_lower,
    #                                                                                     x_upper_bound=rover_x_upper,
    #                                                                                     y_lower_bound=rover_y_lower,
    #                                                                                     y_upper_bound=rover_y_upper, return_value="nav")
    #
    # rover_xpos = round(int(rover.pos[0]))
    # rover_ypos = round(int(rover.pos[1]))

    # # if available, get the closest one to the rover
    # if unexplored_points_beside_nav_points:
    #     print("found unexplored points beside nav points ", unexplored_points_beside_nav_points)
    #     array_format = np.asarray(unexplored_points_beside_nav_points)
    #     x_points = array_format[:, 1]
    #     y_points = array_format[:, 0]
    #     distances_from_rover = compute_distances(rover_xpos, rover_ypos, x_points, y_points)
    #     closest_point_index = np.argmin(distances_from_rover)
    #     new_target_x = x_points[closest_point_index]
    #     new_target_y = y_points[closest_point_index]
    # # if none, choose the nearest accessible unexplored point, and travel to it
    # else:
    #     # print("unexplored points beside nav points not found, looking for unobstructed unexplored points intead")
    #     # closest_unobstructed_point = choose_closest_unobstructed_point(rover_xpos, rover_ypos,
    #     #                                                                rover.memory_map[:, :, 3], flag_target=0,
    #     #                                                                flag_obstruction=5, minimum_distance=0,
    #     #                                                                x_lower_bound=rover_x_lower,
    #     #                                                                x_upper_bound=rover_x_upper,
    #     #                                                                y_lower_bound=rover_y_lower,
    #     #                                                                y_upper_bound=rover_y_upper)
    #     # if closest_unobstructed_point:
    #     #     print("found closest unobstructed point ", closest_unobstructed_point)
    #     #     new_target_x = closest_unobstructed_point[0]
    #     #     new_target_y = closest_unobstructed_point[1]
    #     # else:
    #     print("no unobstructed points found")

    new_target_x = None
    new_target_y = None
    if rover.return_home:
        new_coords = choose_closest_flag(rover.start_pos[0], rover.start_pos[1], rover.memory_map[:, :, 3])[0]

        if new_coords:
            new_target_x = new_coords[0]
            new_target_y = new_coords[1]
        # once we've reached the starting position:
        if coordinates_reached(rover.pos, rover.start_pos):
            rover.return_home = False
            # assign the next quadrant as the target
            rover.target_quadrant = (rover.target_quadrant + 1) % 4

    else:
        new_coords = choose_farthest_flag(rover.start_pos[0], rover.start_pos[1],
                                          rover.memory_map[:, :, 3],
                                          flag=7, x_lower_bound=rover_x_lower,
                                          x_upper_bound=rover_x_upper,
                                          y_lower_bound=rover_y_lower,
                                          y_upper_bound=rover_y_upper)[0]
        if new_coords:
            new_target_x = new_coords[0]
            new_target_y = new_coords[1]

    print("found new coords instead ", new_coords)

    if new_target_x and new_target_y:
        return (new_target_x, new_target_y)
    else:
        print("no target found ")
        return None


def compute_destination_points(rover):
    print("entering loop")
    # if we don't have any, then we get new ones
    # 1. recheck if we can travel to destination in a straight line
    # if there are no obstacles blocking the way to the target, then assign target as destination point
    flag_list = [5]

    path = []

    obstacles = obstacle_crossed_by_line(rover.pos[0], rover.pos[1],
                                         rover.target[0],
                                         rover.target[1],
                                         rover.memory_map[:, :, 3], flag_list)

    if not obstacles:
        # destination_point = rover.target
        path = [rover.target]
    # 2. if there are obstacles, then let's check if we can sidestep these obstacles:
    else:
        path_guide = sidestep_obstacle(rover.pos[0], rover.pos[1],
                                       rover.target[0],
                                       rover.target[1],
                                       rover.memory_map[:, :, 3],
                                       7, 5)
        # if we are successful in finding a path that can sidestep, we assign the nearer point as the
        # destination, and we queue the Rover.target point in Rover.path for later use upon reaching
        # Rover.destination
        if path_guide:
            # destination_point = (path_guide.x, path_guide.y)
            path = [rover.target, (path_guide.x, path_guide.y)]
        # 3. if we were unable to sidestep, then we plot a path using A *
        else:

            print("attempting A *")

            obstaclevalues = [5, 0]
            matrix = np.in1d(rover.memory_map[:, :, 3].ravel(), obstaclevalues).reshape(
                rover.memory_map[:, :, 3].shape).tolist()

            grid = Grid(matrix=matrix)

            start = grid.node(round(int(rover.pos[0])), round(int(rover.pos[1])))
            end = grid.node(rover.target[0], rover.target[1])
            print("computing A star")
            finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
            path, runs = finder.find_path(start, end, grid)
            print("computation finished with runs: ", runs)
            path = list(reversed(path))
            # if path:
            #     print("path exists ")
            #     # assign the current position as the destination
            #     path.pop()
            #     rover.destination_point = path.pop()
            #     rover.path = path
            # else:
            #     print("no path found")
            #     quadrant = rover.target_quadrant
            #     coordinate_bounds = get_coordinate_lower_and_upper_bounds(
            #         quadrant,
            #         rover.memory_map[
            #         :, :, 3])
            #
            #     # If no path to the target can be found, look for the closest nav point beside an unexplored point
            #
            #     new_coords = \
            #         choose_closest_flag(int(rover.pos[0]),
            #                             int(rover.pos[1]),
            #                             rover.memory_map[:, :, 3],
            #                             flag=7,
            #                             x_lower_bound=coordinate_bounds[0],
            #                             x_upper_bound=coordinate_bounds[1],
            #                             y_lower_bound=coordinate_bounds[2],
            #                             y_upper_bound=coordinate_bounds[3])[0]
            #
            #     new_target = new_coords
            #     destination_point = new_coords
            #     print("new coords ", new_coords)

    return path
