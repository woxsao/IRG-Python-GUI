# README

## Setup
- This code assumes you have installed the MATLAB dsltl repository and code. 
- Put the Python implementation folder at the same level as the dsltl folder
- Put the filereshape file in the python implementation folder into the dsltl folder

## How to use:
### Python end:
- Run CursorGUI.py
- Drag and click to draw trajectory
- Release when done with trajectory
- Draw as many as you would like
- When you have as many trajectories as you'd like, hit the key "d" to save
- If you ever make a mistake and want to clear the data, hit the key "c" to clear.
### Matlab end:
- Run filereshape.m (this reshapes the .mat files properly)
- Run the learning/load segments!

## Simulation how to:
###Matlab end:
- Current state of the code doesn't support learning the policy, so the dsltl Matlab code is needed for the policy generation. 
- To do this, go to dsltl code, run the section that Learns the DS. (assuming you have generated the trajectories already through python/matlab file reshape).
### Python end: 
- Current state of the code just supports displaying the learned trajectory in Python.
- Run EndGUI.py
