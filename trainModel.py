import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import tensorflow as tf
import keras
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print('Num GPUs Available: ', len(physical_devices))
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

N = 120  # input y size
M = 160  # input x size
batch_sz = 128
num_epochs = 24

train_folder = 'D:/eye_tracking/LPW/train_frames/'
valid_folder = 'D:/eye_tracking/LPW/valid_frames/'
test_folder = 'D:/eye_tracking/LPW/test_frames/'

if (os.path.isfile('./meanIntensity.txt')):
    meanIntensity = np.loadtxt("meanIntensity.txt")
else:
    meanIntensity = np.float64(0.48)

def get_dataset_files(datasetFolder, M, N):

    df = pd.read_csv(datasetFolder+'targets.csv')
    # add a filename column based on Subject Id, Video Id, and Frame Id
    df['Filename'] = 'subj_' + df['Subject Id'].astype(str) + '_vid_' + df['Video Id'].astype(str) + '_frame_' + df['Frame Id'].apply(lambda x: f'{x:04d}.png')

    # extract x Value and y value columns into numpy arrays
    x_values = df['x Value'].to_numpy()/M
    y_values = df['y Value'].to_numpy()/N
    filename_list = df['Filename'].tolist()
    filename_list_w_path = [f"{datasetFolder}{file}" for file in filename_list]
    train_targets = np.concatenate([x_values.reshape(-1, 1), y_values.reshape(-1, 1)], axis=1)
    return filename_list_w_path, train_targets


train_files, train_targets = get_dataset_files(train_folder, M, N)
test_files, test_targets = get_dataset_files(test_folder, M, N)
valid_files, valid_targets = get_dataset_files(valid_folder, M, N)


# function to load and preprocess images
def load_and_preprocess_image(image_path, mean_gray):
    # load image from file path
    image = tf.io.read_file(image_path)
    # decode jpeg encoded image
    image = tf.image.decode_jpeg(image, channels=1)
    # normalize pixel values to be in the range [0, 1] and subtract r,g,b mean
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.subtract(image, mean_gray)
    return image


train_dataset = tf.data.Dataset.from_tensor_slices((train_files, train_targets))
train_dataset = train_dataset.map(lambda x, y: (load_and_preprocess_image(x, meanIntensity), y))
train_dataset = train_dataset.batch(batch_sz)

test_dataset = tf.data.Dataset.from_tensor_slices((test_files, test_targets))
test_dataset = test_dataset.map(lambda x, y: (load_and_preprocess_image(x, meanIntensity), y))
test_dataset = test_dataset.batch(batch_sz)

valid_dataset = tf.data.Dataset.from_tensor_slices((valid_files, valid_targets))
valid_dataset = valid_dataset.map(lambda x, y: (load_and_preprocess_image(x, meanIntensity), y))
valid_dataset = valid_dataset.batch(batch_sz)


def create_model(input_shape):
    input_layer = Input(shape=input_shape)
    conv1 = Conv2D(16, (17, 17), activation='relu', padding='same')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2,2))(conv1)
    conv2 = Conv2D(32, (9, 9), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2,2))(conv2)
    conv3 = Conv2D(64, (5, 5), activation='relu', padding='same')(pool2)
    pool3 = MaxPooling2D(pool_size=(2,2))(conv3)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool3)
    conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    flatten = Flatten()(conv5)
    dense1 = Dense(256, activation='relu')(flatten)
    dropout1 = Dropout(0.1)(dense1)
    dense2 = Dense(64, activation='relu')(dropout1)
    dropout2 = Dropout(0.1)(dense2)
    dense3 = Dense(64, activation='relu')(dropout2)
    x_output = Dense(1, activation='linear', name='x_output')(dense3)
    y_output = Dense(1, activation='linear', name='y_output')(dense3)
    model = Model(inputs=input_layer, outputs=[x_output, y_output])
    model.compile(optimizer='adam', loss='mse')
    return model

# Create the model
input_shape = (M, N, 1)
model = create_model(input_shape)
model.summary()

model.compile(optimizer='adam', loss='mse')

# set checkpoints to save after each epoch
checkpoint_filepath = './models/model_checkpoint.h5'
os.makedirs('./models', exist_ok=True)

model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

# continue training from last checkpoint if the model was trained earlier
if os.path.isfile('./models/model_checkpoint.h5'):
    model = load_model('./models/model_checkpoint.h5')

# Train the model
model.fit(train_dataset, epochs=1, batch_size=batch_sz, validation_data=valid_dataset, callbacks=[model_checkpoint_callback])

predictions = model.predict(x=test_dataset)

pred2 = np.transpose(np.squeeze(np.array(predictions)))
test_errors = test_targets - pred2
mse_test_err = np.mean(test_errors**2, axis=0)

print('Mean Square Test Errors(x, y) = ')
print(mse_test_err)

dummy = 1