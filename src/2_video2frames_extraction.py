import numpy as np
import cv2
import os

# Time library
import time
from time import strftime
from datetime import datetime

# CHANGE THE NAME HERE
video_folder='20241002_01_35'

# Get the parent directory
parent_directory = os.path.dirname(os.getcwd())

# Change the current working directory to the parent directory
os.chdir(parent_directory)
CWD_PATH = os.getcwd()

in_video_path = os.path.join(CWD_PATH, "Out_Video")
out_frames_path = os.path.join(CWD_PATH, "Out_Frames")

# ###################################################
# ##                LEFT CAMERA                    ##
# ###################################################
# Left video path
in_left_video_path = os.path.join(in_video_path, video_folder, 'left_video.avi')
# Out left frames path
out_left_frames_path = os.path.join(out_frames_path, video_folder, 'left_frames')
# Create directory
try:
    # Create target Directory
    os.makedirs(out_left_frames_path, exist_ok=True)
    print(out_left_frames_path,  " created ")
except FileExistsError:
    print(out_left_frames_path,  " already exists")
    
# Create a VideoCapture object and read from input file
left_cap = cv2.VideoCapture(in_left_video_path)
# a = left_cap.get(3)  # Width
# b = left_cap.get(4)  # Height
# ret = left_cap.set(3, 320)
# ret = left_cap.set(4, 240)
# print(f"Left width: {a}, Left height: {b}")
if not left_cap.isOpened():
    print ("Video Cannot be Opened")
    exit()

left_currentFrame = 0

while(left_cap.isOpened()):
    ret, left_frame = left_cap.read()
    if not ret:
        print("Failed to read frame from left video")
        break
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Saves image of the current frame in jpg file
    out_left_frames_name = os.path.join(out_left_frames_path, f'{left_currentFrame:05d}.png')
    cv2.imwrite(out_left_frames_name, left_frame)
    cv2.imshow('Left video', left_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # To stop duplicate images
    left_currentFrame += 1
print("Completed extracting left frames...!")

# ###################################################
# ##                RIGHT CAMERA                   ##
# ###################################################
# Right video path
in_right_video_path = os.path.join(in_video_path, video_folder, 'right_video.avi')
# Out right frames path
out_right_frames_path = os.path.join(out_frames_path, video_folder, 'right_frames')
# Create directory
try:
    # Create target Directory
    os.makedirs(out_right_frames_path, exist_ok=True)
    print(out_right_frames_path,  " created ")
except FileExistsError:
    print(out_right_frames_path,  " already exists")
# Create a VideoCapture object and read from input file
right_cap = cv2.VideoCapture(in_right_video_path)
# a = right_cap.get(3)  # Width
# b = right_cap.get(4)  # Height
# ret = right_cap.set(3, 320)
# ret = right_cap.set(4, 240)
# print(f"Right width: {a}, Right height: {b}")

if not right_cap.isOpened():
    print ("Video Cannot be Opened")
    exit()
    
right_currentFrame = 0

while(right_cap.isOpened()):
    ret, right_frame = right_cap.read()
    if not ret:
        print("Failed to read frame from right video")
        break
    
    # Saves image of the current frame in jpg file
    out_right_frames_name = os.path.join(out_right_frames_path, f'{right_currentFrame:05d}.png')
    cv2.imwrite(out_right_frames_name, right_frame)
    cv2.imshow('Right video',right_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # To stop duplicate images
    right_currentFrame += 1
print("Completed extracting right frames...!")

# Release the VideoCapture objects
left_cap.release()
right_cap.release()
cv2.destroyAllWindows()