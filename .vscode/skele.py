import h5py
import numpy as np
import glfw
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

# Open the HDF5 file to access skeleton data
filename = "predictions.analysis.h5"
with h5py.File(filename, "r") as f:
    dset_names = list(f.keys())
    locations = f["tracks"][:].T  # Ensure correct shape after transposition
    node_names = [n.decode() for n in f["node_names"][:]]

# Check data structur
print("===filename===")
print(filename)
print()

print("===HDF5 datasets===")
print(dset_names)
print()

print("===locations data shape===")
print(locations.shape)
print()

print("===nodes===")
for i, name in enumerate(node_names):
    print(f"{i}: {name}")
print()

# Define the number of joints (assuming 13 joints)
num_joints = 13
num_frames = locations.shape[0]  # Number of frames in the dataset

# Function to draw the skeleton for a single frame
def drawSkeleton(frame_idx):
    # Get the locations of joints for the given frame
    frame_locations = locations[frame_idx]
    
    print(f"frame_locations shape: {frame_locations.shape}")  # Checking the structure
    print(f"frame_locations[0]: {frame_locations[0]}")  # Print the first joint location for debugging
    
    glBegin(GL_LINES)
    for i in range(num_joints - 1):  # Assuming joints are connected in order
        # Extract the x and y coordinates correctly from the nested structure
        x1, y1 = frame_locations[i, 0]
        x2, y2 = frame_locations[i + 1, 0]
        
        # Print to verify the coordinates before passing to OpenGL
        print(f"Joint {i} coordinates: ({x1}, {y1}) -> ({x2}, {y2})")
        
        # Pass the float coordinates to OpenGL
        glVertex2f(float(x1), float(y1))
        glVertex2f(float(x2), float(y2))
    glEnd()

    # Draw joints as points (optional)
    glPointSize(5)
    glBegin(GL_POINTS)
    for i in range(num_joints):
        # Extract the x and y coordinates correctly for each joint
        x, y = frame_locations[i, 0]
        
        # Print to verify the coordinates
        print(f"Joint {i} coordinates: ({x}, {y})")
        
        # Pass the float coordinates to OpenGL
        glVertex2f(float(x), float(y))
    glEnd()

# Initialize OpenGL
if not glfw.init():
    print("Failed to initialize GLFW")
    exit()

window = glfw.create_window(800, 600, "Skeleton Visualization", None, None)
if not window:
    glfw.terminate()
    print("Failed to create window")
    exit()

glfw.make_context_current(window)
glOrtho(0.0, 800.0, 0.0, 600.0, -1.0, 1.0)  # Orthographic 2D projection

frame_idx = 0  # Start from the first frame

while not glfw.window_should_close(window):
    glClear(GL_COLOR_BUFFER_BIT)

    # Draw the skeleton for the current frame
    drawSkeleton(frame_idx)

    # Update frame index to create animation effect (loop over frames)
    frame_idx = (frame_idx + 1) % num_frames

    # Swap buffers and process events
    glfw.swap_buffers(window)
    glfw.poll_events()

# Cleanup
glfw.terminate()
