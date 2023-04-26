import os
import glob
import random

folder_path = 'D:/eye_tracking/LPW/'
avi_files = glob.glob(os.path.join(folder_path, "**/*.avi"), recursive=True)

for i in range(len(avi_files)):
    avi_files[i] = avi_files[i].replace('\\', '/')

#for file_path in avi_files:
#    print(file_path)

N = len(avi_files)

# Calculate number of files for each set
train_N = int(N * 0.7)
valid_N = int(N * 0.2)
test_N = N - train_N - valid_N

# Shuffle the list of file names randomly
random.shuffle(avi_files)

# Split the files into training, validation, and test sets
train_files = avi_files[:train_N]
valid_files = avi_files[train_N:train_N+valid_N]
test_files = avi_files[train_N+valid_N:]

# Print the number of files in each set
print(f"Number of training files: {len(train_files)}")
print(f"Number of validation files: {len(valid_files)}")
print(f"Number of test files: {len(test_files)}")

def getTargetList(datasetFiles):
    targetFiles = []
    for file in datasetFiles:
        target_file = file[:-3] + "txt"
        targetFiles.append(target_file)
    return targetFiles

train_labels = getTargetList(train_files)
valid_labels = getTargetList(valid_files)
test_labels = getTargetList(test_files)


# Write file paths to a new text file
def writeFileLists(folder, fileName, fileList):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, fileName), 'w') as f:
        for file in fileList:
            f.write(file + '\n')

writeFileLists('./train_val_test/', 'train_files.txt', train_files)
writeFileLists('./train_val_test/', 'train_labels.txt', train_labels)
writeFileLists('./train_val_test/', 'test_files.txt', test_files)
writeFileLists('./train_val_test/', 'test_labels.txt', test_labels)
writeFileLists('./train_val_test/', 'valid_files.txt', valid_files)
writeFileLists('./train_val_test/', 'valid_labels.txt', valid_labels)
