import path_generation_helpers
import numpy as np
from decision import MISALIGNMENT_THRESHOLD

class State:
    def run(self, rover_object):
        assert 0, "run not implemented"
    def next(self, rover_object):
        assert 0, "next not implemented"


class StateMachine:
    def __init__(self, initialState):
        self.currentState = initialState
        self.next_step = self.currentState.run()
    # Template method:
    def runAll(self, inputs):
        for i in inputs:
            print(i)
            self.currentState = self.currentState.next(i)
            self.currentState.run()

    # def proceed_to_next_step(self):


DESIGNATING_TARGET = "designating target"
GENERATING_PATH_TO_TARGET = "generating path to target"

STOPPING = "stopping rover"
STOPPED = "rover stopped"
MOVING_FORWARD = "moving forward"
REORIENTING = "reorienting rover"

# def stateswitch(state):
#     if state == DESIGNATING_TARGET:
#         return DesignatingTarget


class DesignatingTarget(State):
    def run(self, rover_object):
        print("Designating target")
        rover_object.target = path_generation_helpers.get_new_target(rover_object)

    def next(self, rover_object):
        if rover_object.target:
            print("Target designated")
            return GENERATING_PATH_TO_TARGET
        else:
            return DESIGNATING_TARGET


class GeneratingPathToTarget(State):
    def run(self, rover_object):
        path = path_generation_helpers.generate_path_points(rover_object)
        if path:
            rover_object.destination_point = path.pop()
            rover_object.path = path
        else:
            rover_object.destination_point = None
            rover_object.path = []

    def next(self, rover_object):
        if rover_object.destination_point:
            return MOVING_FORWARD
        else:
            return DESIGNATING_TARGET


class MovingForward(State):
    def __init__(self):
        self.run_status = MOVING_FORWARD

    def run(self, rover_object):
        # if (Rover.vel < 0.1) and (Rover.throttle > .2):
        #     Rover.mode = 'rut'
        if abs(rover_object.misalignment) <= MISALIGNMENT_THRESHOLD:
            # and velocity is below max, then throttle
            if rover_object.vel < rover_object.max_vel:
                # Set throttle value to throttle setting
                rover_object.throttle = rover_object.throttle_set
            else:  # Else coast
                rover_object.throttle = 0
            rover_object.brake = 0
            # Set steering to be consistent with the destination angle
            if abs(rover_object.misalignment) <= MISALIGNMENT_THRESHOLD:
                rover_object.steer = np.clip(rover_object.misalignment, -15, 15)
            self.run_status = MOVING_FORWARD

        elif abs(rover_object.misalignment) > MISALIGNMENT_THRESHOLD:
            rover_object.mode = 'stop'
            self.run_status = STOPPING

    def next(self, rover_object):
        return self.run_status


class Stopping(State):
    def __init__(self):
        self.stop_status = STOPPING

    def run(self, rover_object):
        # If we're in stop mode but still moving keep braking
        if rover_object.vel > 0.2:
            print("still too fast, will need to brake")
            rover_object.throttle = 0
            rover_object.brake = rover_object.brake_set
            rover_object.steer = 0
            self.stop_status = STOPPING
        # If we're not moving (vel < 0.2) then do something else
        elif 0 <= rover_object.vel <= 0.1:
            print("now slow")
            # Now we're stopped
            rover_object.brake = rover_object.brake_set
            self.stop_status = STOPPED

    def next(self, rover_object):
        if self.stop_status == STOPPING:
            return STOPPING
        elif self.stop_status == STOPPED:
            return REORIENTING


class Reorienting(State):
    def __init__(self):
        self.reorient_status = REORIENTING

    def run(self, rover_object):
        # recompute angles and misalignment before dealing with rover turning and reorientation
        if rover_object.destination_point:
            __, destination_radians = path_generation_helpers.to_polar_coords_with_origin(
                rover_object.pos[0],
                rover_object.pos[1],
                rover_object.destination_point[
                    0],
                rover_object.destination_point[
                    1])

            rover_object.destination_angle = (destination_radians * (180 / np.pi))
            rover_object.misalignment = path_generation_helpers.compute_misalignment(
                rover_object.destination_angle, rover_object.yaw)

            # Check if we are correctly aligned, and the rover is pointing to the destination
            if abs(rover_object.misalignment) <= 2:
                # if yes, check if we have space in front
                print("misalignment corrected")
                # we will need a smarter way to check if we have space in frontS
                if len(rover_object.nav_angles) >= rover_object.go_forward:
                    print("going forward since it's free")
                    # Set throttle back to stored value
                    rover_object.throttle = rover_object.throttle_set
                    # Release the brake
                    rover_object.brake = 0
                    # Set steer to mean angle
                    rover_object.mode = 'forward'
                    self.reorient_status = MOVING_FORWARD

                # if there is no space in front, then we assign None to destination
                # for it to be replaced
                else:
                    self.reorient_status = DESIGNATING_TARGET

            # if we are still significantly misaligned, continue turning
            elif abs(rover_object.misalignment) > 2:
                print("Correcting misalignment")
                # Make sure we're not throttling
                rover_object.throttle = 0
                # Release the brake
                rover_object.brake = 0
                # Turn to the correct orientation
                rover_object.steer = np.clip(rover_object.misalignment, -15, 15)
                self.reorient_status = REORIENTING
            
        else:
            self.reorient_status = GENERATING_PATH_TO_TARGET

    def next(self, rover_object):
        return self.reorient_status




def stateswitch(state):
    if state == DESIGNATING_TARGET:
        return DesignatingTarget
    elif state == GENERATING_PATH_TO_TARGET:
        return GeneratingPathToTarget
    elif state == STOPPING:
        return Stopping
    elif state == REORIENTING:
        return Reorienting
    elif state == MOVING_FORWARD:
        return MovingForward
    else:
        return DesignatingTarget


class RoverStateMachine:
    def __init__(self, initial_state):

        state_class = stateswitch(initial_state)
        self.current_state = None
        self.next_state = state_class

    def run_state_machine(self, rover_object):
        state_class = stateswitch(self.next_state)
        # initialize the state
        self.current_state = state_class()
        self.current_state.run(rover_object)

        self.next_state = self.current_state.next(rover_object)

        if self.next_state:
            return self.next_state
        else:
            return False
