import numpy as np


# This is where you can build a decision tree for determining throttle, brake and steer 
# commands based on the output of the perception_step() function
def decision_step(Rover):

    # Implement conditionals to decide what to do given perception data
    # Here you're all set up with some basic functionality but you'll need to
    # improve on this decision tree to do a good job of navigating autonomously!

    # Example:
    # Check if we have vision data to make decisions with
    if Rover.nav_angles is not None:
        # Check for Rover.mode status
        if Rover.mode == 'forward':
            # Check the extent of navigable terrain

            if Rover.angle_to_rock != 0:
                # Rover.brake = 10
                Rover.mode = 'stop'
                # Rover.throttle = 0
                # Rover.brake = 10
                # Rover.steer = Rover.angle_to_rock
                # if Rover.near_sample:
                #     Rover.throttle = 0
                #     # Set brake to stored brake value
                #     Rover.brake = Rover.brake_set
                #     Rover.steer = 0
                #     Rover.mode = 'stop'

            if len(Rover.nav_angles) >= Rover.stop_forward:
                # If mode is forward, navigable terrain looks good 
                # and velocity is below max, then throttle 
                if Rover.vel < Rover.max_vel:
                    # Set throttle value to throttle setting
                    Rover.throttle = Rover.throttle_set
                else: # Else coast
                    Rover.throttle = 0

                Rover.brake = 0
                # Set steering to average angle clipped to the range +/- 15
                # mean_obstacle_angles = np.mean(Rover.obstacle_angles) * (180/np.pi)



                steering = Rover.angle_to_next_follow_point

                # Rover.steer = (np.clip(-complementary_angle, -15, 15))
                # Rover.steer = np.clip(np.mean((Rover.nav_angles * 180)/np.pi), -15, 15)
                Rover.steer = steering
                # print("Rover steering ", Rover.steer)
            # If there's a lack of navigable terrain pixels then go to 'stop' mode
            elif len(Rover.nav_angles) < Rover.stop_forward:
                    # Set mode to "stop" and hit the brakes!
                    Rover.throttle = 0
                    # Set brake to stored brake value
                    Rover.brake = Rover.brake_set
                    Rover.steer = 0
                    Rover.mode = 'stop'
            # If there are no navigable pixels in front
            # if Rover.front_obstacle is True:
            #     Rover.mode = 'avoid_obstacle'
            # if (Rover.front_navigable is True) and (Rover.left_obstacle is False) and (Rover.upper_left_obstacle is False):
            #     Rover.mode = 'stop'
            # if Rover.throttle > 0 and Rover.vel <= .02:
            #     Rover.throttle = 0
            #     # Rover.brake = Rover.brake_set
            #     Rover.steer = 0
            #     Rover.mode = 'avoid_rut'


        # If we're already in "stop" mode then make different decisions
        elif Rover.mode == 'stop':
            # If we're in stop mode but still moving keep braking
            if Rover.vel > 0.2:
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0

            # If we're not moving (vel < 0.2) then do something else
            elif Rover.vel <= 0.2:
                if Rover.angle_to_rock > 0:
                    Rover.steer = Rover.angle_to_rock - Rover.find_wall_angle
                    Rover.mode = "get_rock"

                # Now we're stopped and we have vision data to see if there's a path forward
                if len(Rover.nav_angles) < Rover.go_forward:
                    Rover.throttle = 0
                    # Release the brake to allow turning
                    Rover.brake = 0
                    # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
                    Rover.steer = -7 # Could be more clever here about which way to turn

                # if Rover.front_obstacle and Rover.upper_left_obstacle and Rover.left_obstacle:
                #     Rover.steer = -10
                # elif Rover.left_obstacle and (Rover.upper_left_obstacle is False) and (Rover.front_obstacle is False):
                #     Rover.steer = 10
                # elif Rover.front_obstacle:
                #     Rover.steer = -10
                # elif Rover.front_navigable and Rover.front_obstacle:
                #     Rover.steer = -10
                # else:
                #     Rover.steer = np.clip(np.mean(Rover.nav_angles * 180 / np.pi), -15, 15)

                # If we're stopped but see sufficient navigable terrain in front then go!
                if len(Rover.nav_angles) >= Rover.go_forward:
                    # Set throttle back to stored value
                    Rover.throttle = Rover.throttle_set
                    # Release the brake
                    Rover.brake = 0
                    # Set steer to mean angle
                    Rover.steer = Rover.angle_to_next_follow_point
                    # Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
                    Rover.mode = 'forward'

        elif Rover.mode == 'get_rock':
            # while Rover.steer != Rover.angle_rock:
            if Rover.steer != Rover.angle_to_rock:
                Rover.steer = Rover.angle_to_rock

            if Rover.near_sample:
                Rover.brake = Rover.brake_set
            elif Rover.vel < Rover.max_vel:
                # Set throttle value to throttle setting
                Rover.throttle = Rover.throttle_set
            else:  # Else coast
                Rover.throttle = 0
                # Rover.mode = 'stop'


        elif Rover.mode == 'avoid_rut':
            # Rover.brake = 10
            Rover.brake = 0
            Rover.throttle = 0
            # Rover.steer = np.clip(np.mean(Rover.nav_angles * 180 / np.pi), -15, 15)
            if Rover.upper_left_obstacle:
                Rover.steer = -10
            elif Rover.upper_right_obstacle:
                Rover.steer = 10
            Rover.throttle = Rover.throttle_set
            if Rover.front_navigable:
                Rover.throttle = Rover.throttle_set
                if Rover.vel >= .05:
                    Rover.mode = 'forward'
        # elif Rover.mode == 'avoid_obstacle':
        #     Rover.brake = 0
        #     # if Rover.front_obstacle and Rover.left_obstacle:
        #     #     Rover.steer = -7
        #     # elif Rover.front_obstacle and Rover.right_obstacle:
        #     #     Rover.steer = 7
        #     # elif (not Rover.front_obstacle) and (not Rover.front_navigable):
        #     #     Rover.steer = -7
        #     # else:
        #     Rover.steer = np.clip(np.mean(Rover.nav_angles * 180 / np.pi), -15, 15)
        #
        #     if Rover.front_navigable is True:
        #         Rover.mode = "forward"

    # Just to make the rover do something 
    # even if no modifications have been made to the code
    else:
        Rover.throttle = Rover.throttle_set
        Rover.steer = 0
        Rover.brake = 0

    # If in a state where want to pickup a rock send pickup command
    if Rover.near_sample and Rover.vel == 0 and not Rover.picking_up:
        Rover.send_pickup = True

    return Rover

