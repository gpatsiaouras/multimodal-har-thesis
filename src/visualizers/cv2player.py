import cv2
import time


def frames_player(frames):
    # Display the video frame by frame
    for frame in frames:
        cv2.imshow('Frame', frame)
        cv2.waitKey(25)
        # Loop at 15 frames/sec
        time.sleep(1/15)
