import cv2
import os
import pandas as pd
import tqdm

# Read training file paths from a text file and store in a list

def readStoredFiles(folder, fileName):
    with open(os.path.join(folder, fileName), 'r') as f:
        files = f.readlines()
        files = [file_path.strip() for file_path in files]
    return files

folder = './train_val_test/'
train_files = 'train_files.txt'
valid_files = 'valid_files.txt'
test_files = 'test_files.txt'

train_targets = 'train_labels.txt'
valid_targets = 'valid_labels.txt'
test_targets = 'test_labels.txt'

downSample = 4

def grayscale_and_resize(videoFiles, targetFiles, downSample, outputPath):
    os.makedirs(outputPath, exist_ok=True)

    targets_x = []
    targets_y = []
    subjects = []
    vid_nums = []
    frame_nums = []
    for file_path in targetFiles:
        file_parts = file_path.split('/')
        subject_id = file_parts[-2]
        video_id = file_parts[-1].split('.')[0]
        with open(file_path, "r") as f:
            # Read all lines and extract x and y values
            lines = f.readlines()
            x_values = [float(line.split()[0]) / downSample for line in lines]
            y_values = [float(line.split()[1]) / downSample for line in lines]
            targets_x.extend(x_values)
            targets_y.extend(y_values)
            subjects.extend([subject_id] * len(x_values))
            vid_nums.extend([video_id] * len(x_values))
            frame_nums.extend(range(len(x_values)))

    data = {'Subject Id': subjects, 'Video Id': vid_nums, 'Frame Id': frame_nums, 'x Value': targets_x, 'y Value': targets_y}
    df = pd.DataFrame(data)
    df.to_csv(outputPath+'targets.csv', index=False)

    dummy = 1

    for file_path in videoFiles:
        cap = cv2.VideoCapture(file_path)
        frame_count = 0
        file_parts = file_path.split('/')
        subject_id = file_parts[-2]
        video_id = file_parts[-1].split('.')[0]

        # Loop over all frames in the video
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hg, wd = frame.shape
            frame = cv2.resize(frame, (int(wd / downSample), int(hg / downSample)))
            outputFrame = f'subj_{subject_id}_vid_{video_id}_frame_{frame_count:04}.png'
            frame_count += 1

            cv2.imwrite(outputPath+outputFrame, frame)
        # Release video capture object
        cap.release()





train_file_list = readStoredFiles(folder, train_files)
valid_file_list = readStoredFiles(folder, valid_files)
test_file_list = readStoredFiles(folder, test_files)

train_targets_list = readStoredFiles(folder, train_targets)
valid_targets_list = readStoredFiles(folder, valid_targets)
test_targets_list = readStoredFiles(folder, test_targets)

train_output_folder = 'D:/eye_tracking/LPW/train_frames/'
test_output_folder = 'D:/eye_tracking/LPW/test_frames/'
valid_output_folder = 'D:/eye_tracking/LPW/valid_frames/'

grayscale_and_resize(train_file_list, train_targets_list, downSample, train_output_folder)
grayscale_and_resize(valid_file_list, valid_targets_list, downSample, valid_output_folder)
grayscale_and_resize(test_file_list, test_targets_list, downSample, test_output_folder)
