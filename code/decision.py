import numpy as np

MISALIGNMENT_THRESHOLD = 5


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
            # Check if we have a destination point plotted out, if not, stop and get a destination
            if Rover.destination_point is None:
                Rover.mode = 'stop'
                print("no destination point")
            # if a destination point exists
            elif Rover.destination_point:
                print("with_destination_point")
                # IF the x and y values of the current position rounded up or down are equivalent to
                # the destination point:
                x_reached = (np.ceil(Rover.pos[0]) or np.floor(Rover.pos[0])) == Rover.destination_point[0]
                y_reached = (np.ceil(Rover.pos[1]) or np.floor(Rover.pos[1])) == Rover.destination_point[1]
                # We set the destination_point property to zero so that perception can find a new one
                if x_reached and y_reached:
                    Rover.destination_point = None
                    print("destination_removed")
                    print("pixel tag", Rover.memory_map[Rover.destination_point[0], Rover.destination_point[1],3])

            # Check the extent of navigable terrain
            if len(Rover.nav_angles) >= Rover.stop_forward:
                # If mode is forward, navigable terrain looks good 
                # and velocity is below max, then throttle 
                if Rover.vel < Rover.max_vel:
                    # Set throttle value to throttle setting
                    Rover.throttle = Rover.throttle_set
                else:  # Else coast
                    Rover.throttle = 0
                Rover.brake = 0
                # Set steering to be consistent with the destination angle
                if abs(Rover.misalignment) <= MISALIGNMENT_THRESHOLD:
                    Rover.steer = np.clip(Rover.misalignment, -15, 15)
                # If the difference in angles is far too different, stop the vehicle
                elif abs(Rover.misalignment) > MISALIGNMENT_THRESHOLD:
                    print("misalignment > threshold")
                    Rover.mode = 'stop'
            # If there's a lack of navigable terrain pixels then go to 'stop' mode
            elif len(Rover.nav_angles) < Rover.stop_forward:
                print("not enough nav angles available")
                # Set mode to "stop" and hit the brakes!
                Rover.throttle = 0
                # Set brake to stored brake value
                Rover.brake = Rover.brake_set
                Rover.steer = 0
                Rover.mode = 'stop'

        # If we're already in "stop" mode then make different decisions
        elif Rover.mode == 'stop':
            # If we're in stop mode but still moving keep braking
            if Rover.vel > 0.2:
                print("still too fast, will need to brake")
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0
            # If we're not moving (vel < 0.2) then do something else
            elif Rover.vel <= 0.2:
                print("now slow")
                # Now we're stopped
                # Check if we are correctly aligned, and the rover is pointing to the destination
                if abs(Rover.misalignment) <=.05:
                    # if yes, check if we have space in front
                    print("misalignment corrected")
                    # if the destination has been reached
                    if (np.floor(Rover.pos[0]), np.floor(Rover.pos[1])) == (Rover.destination_point):
                        Rover.destination_point = None
                    # we will need a smarter way to check if we have space in frontS
                    elif len(Rover.nav_angles) >= Rover.go_forward:
                        print("going forward since it's free")
                        # Set throttle back to stored value
                        Rover.throttle = Rover.throttle_set
                        # Release the brake
                        Rover.brake = 0
                        # if abs(Rover.misalignment) <= MISALIGNMENT_THRESHOLD:
                        # Set steer to mean angle
                        # Rover.steer = np.clip(Rover.misalignment, -15, 15)
                        Rover.mode = 'forward'
                    # if there is no space in front, then we assign None to destination
                    # for it to be replaced
                    else:
                        print("no space in front, emptying midpoint and destination")
                        Rover.destination_point = None

                # if we are still significantly misaligned, continue turning
                elif abs(Rover.misalignment) > .05:
                    print("Correcting misalignment")
                    # Make sure we're not throttling
                    Rover.throttle = 0
                    # Release the brake
                    Rover.brake = 0
                    # Turn to the correct orientation
                    Rover.steer = np.clip(Rover.misalignment, -15, 15)

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