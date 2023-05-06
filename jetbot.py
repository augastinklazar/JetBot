import time
import numpy as np
import cv2
import jetson.inference
import jetson.utils

# Initialize the JetBot object
jetbot = jetson.inference.JetBot()

# Define the object detection network
net = jetson.inference.detectNet("ssd-mobilenet-v2", 0.5)

# Initialize the camera
camera = jetson.utils.videoSource("csi://0")

# Start the main loop
while True:

    # Get a frame from the camera
    frame = camera.read()

    # Detect objects in the frame
    objects = net.detect(frame)

    # Draw the bounding boxes around the objects
    for obj in objects:
        cv2.rectangle(frame, obj.boundingBox, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("JetBot", frame)

    # Check for the ESC key
    key = cv2.waitKey(1)
    if key == 27:
        break

# Close the camera
camera.stop()

# Destroy all windows
cv2.destroyAllWindows()
