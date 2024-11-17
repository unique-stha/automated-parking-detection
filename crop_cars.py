import os
import cv2
import numpy as np

# Define the paths for the empty and not_empty directories
output_dir_empty = './clf-data/empty'
output_dir_not_empty = './clf-data/not_empty'

# Ensure the directories exist
os.makedirs(output_dir_empty, exist_ok=True)
os.makedirs(output_dir_not_empty, exist_ok=True)

# Read the mask image
mask_path = './data/mask_1920_1080.png'
mask = cv2.imread(mask_path, 0)

# Perform connected components analysis to get slot bounding boxes
analysis = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
(totalLabels, label_ids, values, centroid) = analysis

slots = []
for i in range(1, totalLabels):
    # Extracting the bounding box coordinates and size for each slot
    x1 = values[i, cv2.CC_STAT_LEFT]
    y1 = values[i, cv2.CC_STAT_TOP]
    w = values[i, cv2.CC_STAT_WIDTH]
    h = values[i, cv2.CC_STAT_HEIGHT]
    slots.append([x1, y1, w, h])

# Load the video
video_path = "./data/parking_1920_1080.mp4"
cap = cv2.VideoCapture(video_path)

frame_nmr = 0
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_nmr)
ret, frame = cap.read()

# Function to determine if a slot is empty based on improved detection
def is_empty(cropped_slot):
    gray_slot = cv2.cvtColor(cropped_slot, cv2.COLOR_BGR2GRAY)

    # Apply a GaussianBlur to reduce noise
    gray_slot = cv2.GaussianBlur(gray_slot, (5, 5), 0)

    # Use Canny edge detection to find edges
    edges = cv2.Canny(gray_slot, 50, 150)

    # Calculate the number of edge pixels (non-zero in the edge image)
    edge_pixel_count = np.sum(edges > 0)

    # Calculate the variance in pixel intensity
    variance = np.var(gray_slot)

    # Calculate the mean intensity
    mean_intensity = np.mean(gray_slot)

    # Apply constraints: tuning based on observation
    edge_threshold = 100  # Adjust based on data
    variance_threshold = 200  # Adjust based on data
    mean_intensity_threshold = 50  # Adjust if necessary
    
    # Debug print statements for understanding the values
    print(f"Edge Count: {edge_pixel_count}, Variance: {variance}, Mean Intensity: {mean_intensity}")
    
    # Determine if the slot is empty using a combination of criteria
    if edge_pixel_count < edge_threshold and variance < variance_threshold and mean_intensity > mean_intensity_threshold:
        return True  # Empty
    return False  # Not empty (car present)

while ret:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_nmr)
    ret, frame = cap.read()

    if ret:
        # Process every parking slot
        for slot_nmr, slot in enumerate(slots):
            # Crop the slot area from the frame
            cropped_slot = frame[slot[1]:slot[1] + slot[3], slot[0]:slot[0] + slot[2], :]

            # Determine if the slot is empty or not using the refined function
            if is_empty(cropped_slot):
                # Save to the empty folder
                filename = os.path.join(output_dir_empty, '{}_{}.jpg'.format(str(frame_nmr).zfill(8), str(slot_nmr).zfill(8)))
            else:
                # Save to the not_empty folder
                filename = os.path.join(output_dir_not_empty, '{}_{}.jpg'.format(str(frame_nmr).zfill(8), str(slot_nmr).zfill(8)))

            # Save the cropped image
            cv2.imwrite(filename, cropped_slot)
            print(f"Saved: {filename}")

    # Move to the next frame every 10 frames
    frame_nmr += 10

# Release the video capture object
cap.release()
