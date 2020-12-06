import sys

import cv2 as cv
import numpy as np


def print_sdfdi_image(sdfdi_image, continuous=False):
    """
    Displays the sdfdi image after applying correct transformations for visualization
    :param sdfdi_image:
    :param continuous: True when the image should not stay stable
    :return:
    """
    normalized = cv.normalize(sdfdi_image, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(normalized.astype('uint8'), cv.COLOR_BGR2RGB)
    cv.imshow('sdfdi', bgr)
    cv.waitKey(1 if continuous else 0)


def _get_optical_flow(frame1, frame2):
    """
    Receives two consecutive frames and performs the following:
    1. Converts to grayscale
    2. Calculates the optical flow Ux, Uy and magnitude
    3. Computes first optical flow image d1 by stacking uX 1 as first channel, uY 1 as second channel, and magnitude
    as third channel.
    4. Returns the optical flow image d1
    :param frame1: first frame
    :param frame2: second frame
    :return: optical flow image d1
    """
    frame1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    frame2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    optical_flow = cv.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, _ = cv.cartToPolar(optical_flow[..., 0], optical_flow[..., 1])

    return np.dstack((optical_flow, magnitude))


def generate_sdfdi(frames, verbose=False):
    """
    Generates Stacked Dense Flow Difference Image (SDFDI) for a video sample V with n frames: f1, f2, ..., fn
    1: for each pair of consecutive frames fi and fi+1, extract horizontal flow component uxi and
    vertical flow component uyi
    2: Compute mag = sqrt(ux1^2 + uy1^2)
    3: Compute first optical flow image d1 by stacking ux1 as first channel, uy1 as second channel,
    and mag as third channel.
    4: Initialize SDFDI = 0
    5: for i = 2 to n − 1 do
    6:    Compute mag = sqrt(uxi^2 + uyi^2)
    7:    Compute next optical flow image d2 by stacking ux i as first channel, uy i as second
          channel, and mag as third channel.
    8:    SDFDI = SDFDI + i ∗ |d2 − d1|
    9:    d1 = d2
    10: end for
    11: return SDFDI
    :param frames: frames of a video
    :param verbose: True to show the sdfdi live calculation
    :return: Stacked Dense Flow Difference Image (SDFDI)
    """
    # Calculate optical flow
    d1 = _get_optical_flow(frames[0], frames[1])

    sdfdi = np.zeros(frames[0].shape)
    print("Calculating")
    for i in range(1, len(frames) - 1):
        print('.', end='', flush=True)
        d2 = _get_optical_flow(frames[i], frames[i + 1])

        # Construct SDFDI frame
        sdfdi += i * np.abs(d2 - d1)
        d1 = d2

        # Show sdfdi progress, if verbose is activated
        if verbose:
            print_sdfdi_image(sdfdi, continuous=True)

    return sdfdi


def generate_sdfdi_camera(camera_id=0):
    """
    Generates a Stacked Dense Flow Difference Image (SDFDI) using the frames recorded from a camera live,
    following the algorithm of generate_sdfdi, for every 30 frames.
    :param camera_id: the camera id from /dev/videoX
    """
    cap = cv.VideoCapture(camera_id)

    ret = False
    while not ret:
        ret, _ = cap.read()

    _, frame1 = cap.read()
    _, frame2 = cap.read()
    frame_shape = frame1.shape

    d1 = _get_optical_flow(frame1, frame2)
    i = 1
    sdfdi = np.zeros(frame_shape)
    while True:
        _, frame2 = cap.read()
        d2 = _get_optical_flow(frame1, frame2)

        sdfdi += i * np.abs(d2 - d1)
        frame1 = frame2
        d1 = d2
        i += 1

        # Display the resulting frame
        cv.imshow('original', frame2)
        print_sdfdi_image(sdfdi, continuous=True)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        # Reset the sdfdi image every 30 frames
        sys.stdout.write("\rAccumulating frame %d/30" % i)
        sys.stdout.flush()
        if i // 30:
            print(' Reset')
            sdfdi = np.zeros(frame_shape)
            i = 1

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()
