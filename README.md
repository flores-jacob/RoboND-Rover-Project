## Search and Sample Return Project ##

### Project Overview

#### What is this?
This is my implementation of the Udacity Robotics Nanodegree Search and Sample Return Project (aka the Rover Project)

#### Problem statement
A "Mars" rover is found in a simulated environment, with terrain that probably resembles the Martian surface. It needs to move through the environment so that it can accurately create a map of it, and if possible, collect sample rocks that are scattered all throughout the area. However, the problem is that it does not know how to do this.  It does have a front facing camera, but it does not know how to make sense of the image and video feed coming from it.  It does have wheels that provides it with the ability to move, however it does not know when to move, and where to move.  It also does not know when to turn and when to stop.

Normally, a human driver is necessary to perform all these actions, however in this case, it is necessary that the rover be able to do all these things autonomously. As such, the objective is to provide the Rover with the ability to "see" and to navigate through the environment successfully, such that it would be able to map a certain amount of the environment at a given fidelity.

#### Solution and files of note
- To enable the Rover to see, we warped the images from its camera feed such that the image was basically transformed into a top down view of the terrain. We also made a note of what geenral colors constitute the navigable terrain, as well as the obstacles.  With this knowledge, it was possible to perform color thresholding to mark what areas the Rover can move to, and what areas it should avoid. Furthermore, we perform several rotations on the image that the rover sees before it so that we can properly place it on the terrain map we were constructing.
- The writeup of this project elaborating the steps taken to reach a passing submission can be found here:
    - [WRITEUP.md](./WRITEUP.md).
- The jupyter notebook that was used to test out obstacle and navigable pixel identification, thresholding, and mapping can be found here:
    - [Rover_Project_Test_Notebook.ipynb](./code/Rover_Project_Test_Notebook.ipynb)
- The orignal project home can be found here:
    - [Rover Project Home](https://github.com/udacity/RoboND-Rover-Project.git)