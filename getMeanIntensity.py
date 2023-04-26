import cv2
import numpy as np

# Read training file paths from a text file and store in a list
with open('./train_val_test/train_files.txt', 'r') as f:
    train_files = f.readlines()
    train_files = [file_path.strip() for file_path in train_files]


meanFrameVals = []
# Loop over all training files
for file_path in train_files:
    cap = cv2.VideoCapture(file_path)
    frame_count = 0
    mean_gray_value = 0

    # Loop over all frames in the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute mean grayscale value of frame
        mean_gray_value += np.mean(gray_frame)/total_frames/255

    # Print mean grayscale value of video
    print(f"Mean grayscale value of {file_path}: {mean_gray_value}")
    meanFrameVals.append(mean_gray_value)

    # Release video capture object
    cap.release()

meanMeanFrameVal = np.mean(meanFrameVals)

np.savetxt("meanIntensity.txt", [meanMeanFrameVal])