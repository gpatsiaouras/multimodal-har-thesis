import cv2
import time


def frames_player(frames, n_snapshots=None):
    """
    Shows each frame from the data with 15fps speed. If n_snapshots is defined it will take and save snapshots from the
    video with interval = len(frames) // n_snapshots
    :param frames:
    :param n_snapshots:
    :return:
    """
    # For taking snapshots at equal intervals
    every_steps = len(frames) // n_snapshots
    step = 0
    # Display the video frame by frame
    for frame in frames:
        cv2.imshow('Frame', frame)
        cv2.waitKey(25)
        if not step % every_steps:
            cv2.imwrite('snapshot_%d.png' % step, frame)
        # Loop at 15 frames/sec
        time.sleep(1/15)
