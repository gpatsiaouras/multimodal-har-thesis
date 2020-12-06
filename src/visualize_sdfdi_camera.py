import sys

from tools import generate_sdfdi_camera

# Set default camera to 0
camera_id = 0
# If user specified a different camera use that one
if len(sys.argv) > 1:
    camera_id = sys.argv[1]

# Debug sdfdi generation using the camera
generate_sdfdi_camera(camera_id)
