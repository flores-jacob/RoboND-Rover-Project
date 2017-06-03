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

            # Check the extent of navigable terrain
            if len(Rover.nav_angles) >= Rover.stop_forward:
                # If mode is forward, navigable terrain looks good 
                # and velocity is below max, then throttle 
                if Rover.vel < Rover.max_vel:
                    # Set throttle value to throttle setting
                    Rover.throttle = Rover.throttle_set
                else: # Else coast
                    Rover.throttle = 0
                Rover.brake = 0
                # Set steering to be consistent with the destination angle
                if abs(Rover.misalignment) <= MISALIGNMENT_THRESHOLD:
                    Rover.steer = np.clip(Rover.misalignment, -15, 15)
                # If the difference in angles is far too different, stop the vehicle
                elif abs(Rover.misalignment) > MISALIGNMENT_THRESHOLD:
                    Rover.mode = 'stop'
            # If there's a lack of navigable terrain pixels then go to 'stop' mode
            elif len(Rover.nav_angles) < Rover.stop_forward:
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
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0
            # If we're not moving (vel < 0.2) then do something else
            elif Rover.vel <= 0.2:
                # Now we're stopped and we have vision data to see if there's a path forward
                # If we're stopped butnot aligned to destination angle, then turn towards the destination
                if abs(Rover.misalignment) > MISALIGNMENT_THRESHOLD:
                    Rover.steer = np.clip(Rover.misalignment, -15, 15)
                # If we are aligned, and if we're stopped but see sufficient navigable terrain in front then go!
                elif len(Rover.nav_angles) >= Rover.go_forward:
                    # Set throttle back to stored value
                    Rover.throttle = Rover.throttle_set
                    # Release the brake
                    Rover.brake = 0
                    # if abs(Rover.misalignment) <= MISALIGNMENT_THRESHOLD:
                        # Set steer to mean angle
                        # Rover.steer = np.clip(Rover.misalignment, -15, 15)
                    Rover.mode = 'forward'

                if (len(Rover.nav_angles) < Rover.go_forward) and (Rover.misalignment < MISALIGNMENT_THRESHOLD):
                    Rover.throttle = 0
                    # Release the brake to allow turning
                    Rover.brake = 0
                    # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
                    Rover.steer = np.clip(Rover.misalignment, -15, 15)
    # Just to make the rover do something 
    # even if no modifications have been made to the code
    else:
        print("else")
        Rover.throttle = Rover.throttle_set
        Rover.steer = 0
        Rover.brake = 0

    # If in a state where want to pickup a rock send pickup command
    if Rover.near_sample and Rover.vel == 0 and not Rover.picking_up:
        Rover.send_pickup = True

    return Rover

