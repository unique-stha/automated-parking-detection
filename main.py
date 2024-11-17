import cv2
import numpy as np
from util import get_parking_spots_bboxes, empty_or_not

# Function to calculate the difference between two images
def calc_diff(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))

# Paths to the mask and video files
mask_path = "./data/mask_crop.png"
video_path = "./data/parking_crop_loop.mp4"

# Load the binary mask to identify parking regions
mask = cv2.imread(mask_path, 0)

# Load the video of the parking area
cap = cv2.VideoCapture(video_path)

# Identify connected components in the mask (representing parking spots)
connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)

# Get bounding boxes for all detected parking spots
spots = get_parking_spots_bboxes(connected_components)

# Initialize arrays to store spot status and differences
spots_status = [None for _ in spots]
diffs = [None for _ in spots]

# Variables for tracking the previous frame and frame number
previous_frame = None
frame_nmr = 0
ret = True
step = 30  # Process every 30th frame for efficiency

# Main loop to process the video frame by frame
while ret:
    ret, frame = cap.read()  # Read the next frame from the video

    # Compare current frame with the previous frame to detect changes
    if frame_nmr % step == 0 and previous_frame is not None:
        for spot_indx, spot in enumerate(spots):
            x1, y1, w, h = spot  # Extract bounding box for the spot
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]  # Crop the spot area
            # Calculate the difference between current and previous frames
            diffs[spot_indx] = calc_diff(spot_crop, previous_frame[y1:y1 + h, x1:x1 + w, :])

    # Update spot status based on changes in the frame
    if frame_nmr % step == 0:
        if previous_frame is None:
            arr_ = range(len(spots))  # Process all spots for the first frame
        else:
            # Filter spots with significant changes
            arr_ = [j for j in np.argsort(diffs) if diffs[j] / np.max(diffs) > 0.4]

        for spot_indx in arr_:
            spot = spots[spot_indx]
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]  # Crop the spot area
            spot_status = empty_or_not(spot_crop)  # Check if the spot is empty
            spots_status[spot_indx] = spot_status  # Update the status array

    # Store the current frame for comparison in the next iteration
    if frame_nmr % step == 0:
        previous_frame = frame.copy()

    # Draw rectangles around parking spots to indicate their status
    for spot_indx, spot in enumerate(spots):
        spot_status = spots_status[spot_indx]
        x1, y1, w, h = spot
        if spot_status:  # Green rectangle for empty spots
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
        else:  # Red rectangle for occupied spots
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)

    # Display the number of available spots on the frame
    cv2.rectangle(frame, (80, 20), (550, 80), (0, 0, 0), -1)  # Black background
    cv2.putText(frame, f'Available spots: {sum(spots_status)} / {len(spots_status)}',
                (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show the updated frame in a window
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    # Increment the frame counter
    frame_nmr += 1

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
