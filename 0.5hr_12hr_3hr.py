# This code requires input and output numpy arrays to run.
# Due to constraints on file size that can be uploaded to GitHub, the data tuples in the form of numpy arrays could not be provided.
# See the manuscript for details on preparing data tuples and interpolating the numerical data which have been obtained from RSMC reports.

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, ConvLSTM2D, TimeDistributed, Flatten, Dense, GRU, LSTM, Concatenate, MaxPooling3D, Activation
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

path_images = input("Enter the path to the numpy array of image data tuples: ")
images_series_n = np.load(path_images)
print(images_series_n.shape)  # sanity check for image data tuple dimensions

path_labels = input("Enter the path to the numpy array of labels: ")
labels_n = np.load(path_labels)
print(labels_n.shape)  # sanity check for target label dimensions

path_positions = input("Enter the path to the numpy array of intensity and track inputs: ")
positions_series_n = np.load(path_positions)
print(positions_series_n.shape)  # sanity check for numerical data tuple dimensions

# TC data from 2013 to 2020 are used for training FIT
# TC data from 2021 to 2022 are used for validating the performance of FIT and guiding the model training in the correct direction
# TC data for 2023 are used for testing the generalizability of FIT

train_start_index = int(input("Enter the starting index of train points (from first point of 2013): "))
train_end_index = int(input("Enter the ending index of train points (till last point of 2020): "))
validation_start_index = int(input("Enter the starting index of validation points (from first point of 2021): "))
validation_end_index = int(input("Enter the ending index of validation points (till last point of 2022): "))
test_start_index = int(input("Enter the starting index of test points (from first point of 2023): "))
test_end_index = int(input("Enter the ending index of test points (till last point of 2023): "))

train_images_series_n = images_series_n[train_start_index:train_end_index]                   # 2013 to 2020
train_positions_series_n = positions_series_n[train_start_index:train_end_index]             # 2013 to 2020

validation_images_series_n = images_series_n[validation_start_index:validation_end_index]           # 2021 and 2022
validation_positions_series_n = positions_series_n[validation_start_index:validation_end_index]     # 2021 and 2022

test_images_series_n = images_series_n[test_start_index:test_end_index]                     # 2023
test_positions_series_n = positions_series_n[test_start_index:test_end_index]               # 2023

train_labels_n = labels_n[train_start_index:train_end_index]
validation_labels_n = labels_n[validation_start_index:validation_end_index]
test_labels_n = labels_n[test_start_index:test_end_index]

# Labels follow the order - LON (index: 0), LAT (index: 1), MSW (index: 2), ECP (index: 3)

train_labels_lon_lat = train_labels_n[:, :, 0:2]
train_labels_msw = train_labels_n[:, :, 2]
train_labels_ecp = train_labels_n[:, :, 3]

validation_labels_lon_lat = validation_labels_n[:, :, 0:2]
validation_labels_msw = validation_labels_n[:, :, 2]
validation_labels_ecp = validation_labels_n[:, :, 3]

test_labels_lon_lat = test_labels_n[:, :, 0:2]
test_labels_msw = test_labels_n[:, :, 2]
test_labels_ecp = test_labels_n[:, :, 3]

print(test_labels_lon_lat.shape, test_labels_msw.shape, test_labels_ecp.shape)  # sanity check through dimension verification

path_of_best_model = input("Enter the path where the best trained model will be stored: ")
path_of_best_model = path_of_best_model + "bm.keras" # name of .keras file which will contain the best model parameters for later usage

cp = ModelCheckpoint(monitor='val_loss',
                     save_best_only=True,
                     mode="min",
                     filepath=path_of_best_model,
                     save_weights_only=True,
                    verbose=True)

es = EarlyStopping(monitor='val_loss',
                   patience=20,
                   mode='min',
                   restore_best_weights=True,
                  verbose=True)

# Input layers
image_input = Input(shape=(24, 64, 64, 1), name="image_input")
feature_input = Input(shape=(24, 4), name="feature_input")

# ConvLSTM2D for satellite image time series data
x = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True)(image_input)
x = Activation('relu')(x)
x = MaxPooling3D(pool_size=(1, 2, 2))(x)

x = Activation('relu')(x)

x = Dense(128, activation='relu')(x)

# LSTM / GRU for numerical data tuples
y = LSTM(32, return_sequences=True)(feature_input)
y = LSTM(64, return_sequences=False)(y)
y = Dense(128, activation='relu')(y)

# Merge the two branches
# Flatten the time dimension of the ConvLSTM2D branch output from (None, 1, 128) to (None, 128)
x = tf.squeeze(x, axis=1)

# Concatenate the branches
merged = Concatenate()([x, y])

merged = Dense(256, activation='relu')(merged)


# Reshape the outputs back to time-step format
output_lat_lon = tf.reshape(output_lat_lon, (-1, 6, 2), name="final_lat_lon")
output_pressure = tf.reshape(output_pressure, (-1, 6, 1), name="final_ecp")
output_wind_speed = tf.reshape(output_wind_speed, (-1, 6, 1), name="final_msw")

# Final model
model = Model(inputs=[image_input, feature_input],
              outputs=[output_lat_lon, output_pressure, output_wind_speed])

print(model.summary())

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss={
        "tf.reshape": "mean_squared_error", # or euclidean_distance_loss
        "tf.reshape_1": "mean_squared_error",    # MSE for pressure
        "tf.reshape_2": "mean_squared_error"   # MSE for wind speed
    },
    metrics={
        "tf.reshape": ["mae"],                 # MAE for latitude/longitude
        "tf.reshape_1": ["mae"],                # MAE for pressure
        "tf.reshape_2": ["mae"]               # MAE for wind speed
    }
)

bs = int(input("Enter the batch size to use for model training: "))

# Train the model
history = model.fit(
    [train_images_series_n, train_positions_series_n],
    {
        "tf.reshape": train_labels_lon_lat,  # Labels for latitude/longitude
        "tf.reshape_1": train_labels_ecp,    # Labels for pressure
        "tf.reshape_2": train_labels_msw   # Labels for wind speed
    },
    validation_data=(
        [validation_images_series_n, validation_positions_series_n],
        {
            "tf.reshape": validation_labels_lon_lat,  # Validation labels for latitude/longitude
            "tf.reshape_1": validation_labels_ecp,     # Validation labels for pressure
            "tf.reshape_2": validation_labels_msw    # Validation labels for wind speed
        }
    ),
    epochs=200,
    batch_size=bs,
    callbacks=[cp, es]
)

# saving model training history for later analysis
path_of_history = input("Enter the path to store the model training history: ")
path_of_history = path_of_history + "train_history.csv"
df = pd.DataFrame(history.history)
df.to_csv(path_of_history)

test_results = model.evaluate([test_images_series_n, test_positions_series_n], [test_labels_lon_lat, test_labels_ecp, test_labels_msw])
