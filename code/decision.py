import numpy as np
import path_generation_helpers
import a_star

import rover_state_machine



DESTINATION_LIST = [(85, 80), (76, 73), (57, 95), (13, 98), (62, 106)]
# [(103, 75), (110, 48), (114[, 7)]

OBSTACLE_FLAG = 5
TARGET_LIST = [(200, 200), (0, 200), (200, 0), (0, 0)]

RoverStateMachine = rover_state_machine.RoverStateMachine(rover_state_machine.STOPPING)

# This is where you can build a decision tree for determining throttle, brake and steer
# commands based on the output of the perception_step() function
def decision_step(Rover):
    # Implement conditionals to decide what to do given perception data
    # Here you're all set up with some basic functionality but you'll need to
    # improve on this decision tree to do a good job of navigating autonomously!


    # Before the rover decides where to go and what to do, it should first check if it
    # knows where its going

    # 3. Before dealing with the the decision tree, we first update our values of import

    # Example:
    # Check if we have vision data to make decisions with
    if Rover.nav_angles is not None:

        state = RoverStateMachine.run_state_machine(Rover)

        while state:
            state = RoverStateMachine.run_state_machine(Rover)

        # Check for Rover.mode status
        # if Rover.mode == 'forward':
        #     # if (Rover.vel < 0.1) and (Rover.throttle > .2):
        #     #     Rover.mode = 'rut'
        #     if abs(Rover.misalignment) <= MISALIGNMENT_THRESHOLD:
        #         # and velocity is below max, then throttle
        #         if Rover.vel < Rover.max_vel:
        #             # Set throttle value to throttle setting
        #             Rover.throttle = Rover.throttle_set
        #         else:  # Else coast
        #             Rover.throttle = 0
        #         Rover.brake = 0
        #         # Set steering to be consistent with the destination angle
        #         if abs(Rover.misalignment) <= MISALIGNMENT_THRESHOLD:
        #             Rover.steer = np.clip(Rover.misalignment, -15, 15)
        #     elif abs(Rover.misalignment) > MISALIGNMENT_THRESHOLD:
        #         Rover.mode = 'stop'
        #
        #
        # # If we're already in "stop" mode then make different decisions
        # elif Rover.mode == 'stop':
        #     # If we're in stop mode but still moving keep braking
        #     if Rover.vel > 0.2:
        #         print("still too fast, will need to brake")
        #         Rover.throttle = 0
        #         Rover.brake = Rover.brake_set
        #         Rover.steer = 0
        #     # If we're not moving (vel < 0.2) then do something else
        #     elif 0 <= Rover.vel <= 0.1:
        #         print("now slow")
        #         # Now we're stopped
        #         Rover.brake = Rover.brake_set
        #
        #         with open("mymap.txt", 'w') as mapfile:
        #             mapfile.write(str(Rover.memory_map[:, :, 3].tolist()))
        #
        #         # Check if we don't have a target point. If we don't, we plan a new route
        #         if not Rover.target:
        #             # code to get new target point
        #
        #             print("not target ")
        #             Rover.target = path_generation_helpers.get_new_target(Rover)
        #             path = path_generation_helpers.generate_path_points(Rover)
        #             if path:
        #                 Rover.destination_point = path.pop()
        #                 Rover.path = path
        #             else:
        #                 Rover.destination_point = None
        #                 Rover.path = []
        #         # Check if we don't have destination points leading to the target
        #         if Rover.target and (not Rover.destination_point) and (not Rover.path):
        #             path = path_generation_helpers.generate_path_points(Rover)
        #             if path:
        #                 print("path exists ")
        #                 # assign the current position as the destination
        #                 # path.pop()
        #                 Rover.destination_point = path.pop()
        #                 Rover.path = path
        #             else:
        #                 print("no path found")
        #                 Rover.target = path_generation_helpers.get_new_target(Rover)
        #                 if Rover.target:
        #                     path = path_generation_helpers.generate_path_points(Rover)
        #                     if path:
        #                         Rover.destination_point = path.pop()
        #                         Rover.path = path
        #                     else:
        #                         Rover.destination_point = None
        #                         Rover.path = []
        #         ### END
        #
        #         # recompute angles and misalignment before dealing with rover turning and reorientation
        #         if Rover.destination_point:
        #             __, destination_radians = path_generation_helpers.to_polar_coords_with_origin(
        #                 Rover.pos[0],
        #                 Rover.pos[1],
        #                 Rover.destination_point[
        #                     0],
        #                 Rover.destination_point[
        #                     1])
        #
        #             Rover.destination_angle = (destination_radians * (180 / np.pi))
        #             Rover.misalignment = path_generation_helpers.compute_misalignment(
        #                 Rover.destination_angle, Rover.yaw)
        #
        #         # Check if we are correctly aligned, and the rover is pointing to the destination
        #         if abs(Rover.misalignment) <= 2:
        #             # if yes, check if we have space in front
        #             print("misalignment corrected")
        #             # we will need a smarter way to check if we have space in frontS
        #             if len(Rover.nav_angles) >= Rover.go_forward:
        #                 print("going forward since it's free")
        #                 # Set throttle back to stored value
        #                 Rover.throttle = Rover.throttle_set
        #                 # Release the brake
        #                 Rover.brake = 0
        #                 # Set steer to mean angle
        #                 Rover.mode = 'forward'
        #             # if there is no space in front, then we assign None to destination
        #             # for it to be replaced
        #             else:
        #                 print("no space in front nothing to do, recompute path?")
        #
        #                 # maybe recompute path?
        #
        #                 # first step is to make sure that the entire quadrant has been fully explored
        #                 # quadrant = path_generation_helpers.determine_quadrant(Rover.pos[0], Rover.pos[1],
        #                 #                                                       Rover.memory_map[:, :, 3])
        #
        #                 # coordinate_bounds = path_generation_helpers.get_coordinate_lower_and_upper_bounds(
        #                 #     Rover.target_quadrant,
        #                 #     Rover.memory_map[
        #                 #     :, :, 3])
        #                 # new_destinations = path_generation_helpers.get_nav_points_besides_unexplored_area(
        #                 #     Rover.memory_map[:, :, 3], x_lower_bound=coordinate_bounds[0],
        #                 #     x_upper_bound=coordinate_bounds[1], y_lower_bound=coordinate_bounds[2],
        #                 #     y_upper_bound=coordinate_bounds[3])
        #                 #
        #                 # print("these are potential new destinations ", new_destinations)
        #                 #
        #                 # # assign the first new destination as
        #                 # if np.any(new_destinations):
        #                 #     print("new destinations found, perusing the first one ", new_destinations)
        #                 #     print("new destinations ", new_destinations[0])
        #                 #     Rover.destination_point = tuple(new_destinations[0])
        #                 #
        #                 # # if the quadrant has been fully explored, then it's time to switch quadrants
        #                 # else:
        #                 #     print("quadrant has been fully explored")
        #                 #     Rover.return_home = True
        #                 #
        #                 #     Rover.target = path_generation_helpers.get_new_target(Rover)
        #                 #     path = path_generation_helpers.generate_path_points(Rover)
        #                 #     Rover.destination_point = path.pop()
        #                 #     Rover.path = path
        #
        #                 Rover.target = path_generation_helpers.get_new_target(Rover)
        #                 path = path_generation_helpers.generate_path_points(Rover)
        #                 if path:
        #                     Rover.destination_point = path.pop()
        #                     Rover.path = path
        #                 else:
        #                     Rover.destination = None
        #                     Rover.path = []
        #
        #                 # manually assign steering values since perception.py is probably not running currently
        #                 __, destination_radians = path_generation_helpers.to_polar_coords_with_origin(Rover.pos[0],
        #                                                                                               Rover.pos[1],
        #                                                                                               Rover.destination_point[
        #                                                                                                   0],
        #                                                                                               Rover.destination_point[
        #                                                                                                   1])
        #
        #                 Rover.destination_angle = (destination_radians * (180 / np.pi))
        #                 Rover.misalignment = path_generation_helpers.compute_misalignment(Rover.destination_angle,
        #                                                                                   Rover.yaw)
        #
        #                 # Rover.steer = Rover.misalignment
        #
        #
        #         # if we are still significantly misaligned, continue turning
        #         elif abs(Rover.misalignment) > 2:
        #             print("Correcting misalignment")
        #             # Make sure we're not throttling
        #             Rover.throttle = 0
        #             # Release the brake
        #             Rover.brake = 0
        #             # Turn to the correct orientation
        #             Rover.steer = np.clip(Rover.misalignment, -15, 15)
        # elif Rover.mode == 'rut':
        #     obstacle_angles_degrees = (np.mean(Rover.obstacle_angles) * (180 / np.pi))
        #     # Rover.misalignment = -obstacle_angles_degrees
        #     avg_angle = np.mean(Rover.nav_angles)
        #     avg_angle_degrees = avg_angle * 180 / np.pi
        #     steering = np.clip(avg_angle_degrees, -15, 15)
        #
        #     if len(Rover.nav_angles) < Rover.go_forward:
        #         Rover.throttle = 0
        #         # Release the brake to allow turning
        #         Rover.brake = 0
        #         # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
        #         Rover.steer = steering  # Could be more clever here about which way to turn
        #     # If we're stopped but see sufficient navigable terrain in front then go!
        #     if len(Rover.nav_angles) >= Rover.go_forward:
        #         # Set throttle back to stored value
        #         Rover.throttle = Rover.throttle_set
        #         # Release the brake
        #         Rover.brake = 0
        #         # Set steer to mean angle
        #         Rover.steer = steering
        #         Rover.mode = 'forward'



    # Just to make the rover do something 
    # even if no modifications have been made to the code
    else:
        print("No angles to work with")
        Rover.throttle = Rover.throttle_set
        Rover.steer = 0
        Rover.brake = 0

    # If in a state where want to pickup a rock send pickup command
    if Rover.near_sample and Rover.vel == 0 and not Rover.picking_up:
        Rover.send_pickup = True

    return Rover
