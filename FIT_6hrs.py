import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    ConvLSTM2D,
    TimeDistributed,
    Flatten,
    Dense,
    GRU,
    LSTM,
    Concatenate,
    MaxPooling3D,
    Activation
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

test_df = pd.DataFrame(columns=['Overall Loss', 'Lat_Long_Loss', 'ECP_Loss', 'MSW_Loss', 'Lat_Long_MAE', 'ECP_MAE', 'MSW_MAE'])

images_series_n = np.load(r'D:\ML_DL_Projects\Notebooks\Input arrays\0.5hr_24hr_24hr_images.npy')

images_series_n.shape

labels_n = np.load(r'D:\ML_DL_Projects\Notebooks\Input arrays\0.5hr_24hr_24hr_position_labels.npy')

labels_n.shape

positions_series_n = np.load(r'D:\ML_DL_Projects\Notebooks\Input arrays\0.5hr_24hr_24hr_position_inputs.npy')

positions_series_n.shape

train_images_series_n = images_series_n[0:543]                   # 2013 to 2020
train_positions_series_n = positions_series_n[0:543]             # 2013 to 2020

validation_images_series_n = images_series_n[543:665]           # 2021 and 2022
validation_positions_series_n = positions_series_n[543:665]     # 2021 and 2022

test_images_series_n = images_series_n[665:]                     # 2023
test_positions_series_n = positions_series_n[665:]               # 2023

train_images_series_n.shape, validation_images_series_n.shape, test_images_series_n.shape

train_positions_series_n.shape, validation_positions_series_n.shape, test_positions_series_n.shape

train_labels_n = labels_n[0:543]
validation_labels_n = labels_n[543:665]
test_labels_n = labels_n[665:]

train_labels_n.shape, validation_labels_n.shape, test_labels_n.shape

train_labels_lon_lat = train_labels_n[:, :, 0:2]
train_labels_msw = train_labels_n[:, :, 2]
train_labels_ecp = train_labels_n[:, :, 3]

train_labels_lon_lat.shape, train_labels_msw.shape, train_labels_ecp.shape

validation_labels_lon_lat = validation_labels_n[:, :, 0:2]
validation_labels_msw = validation_labels_n[:, :, 2]
validation_labels_ecp = validation_labels_n[:, :, 3]

validation_labels_lon_lat.shape, validation_labels_msw.shape, validation_labels_ecp.shape

test_labels_lon_lat = test_labels_n[:, :, 0:2]
test_labels_msw = test_labels_n[:, :, 2]
test_labels_ecp = test_labels_n[:, :, 3]

test_labels_lon_lat.shape, test_labels_msw.shape, test_labels_ecp.shape

cp = ModelCheckpoint(monitor='val_loss',
                     save_best_only=True,
                     mode="min",
                     filepath="new2_0.5hr_24hr_24hr_GRU_MSE_8.keras",
                     save_weights_only=True,
                    verbose=True)

es = EarlyStopping(monitor='val_loss',
                   patience=20,
                   mode='min',
                   restore_best_weights=True,
                  verbose=True)

# `images`: Satellite images of shape (num_samples, time_steps, height, width, channels)
# `features`: Time-series data of shape (num_samples, time_steps, 4) where 4 = lat, lon, pressure, wind speed
# `labels`: Corresponding labels for the next 't' hours of shape (num_samples, future_steps, 4)
# Time steps: 48 for the past 24 hours and 12 for the next 6 hours

# Input layers
image_input = Input(shape=(48, 64, 64, 1), name="image_input")
feature_input = Input(shape=(48, 4), name="feature_input")

# ConvLSTM2D for satellite images
x = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True)(image_input)
x = Activation('relu')(x)
x = MaxPooling3D(pool_size=(1, 2, 2))(x)
x = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=False)(x)
x = Activation('relu')(x)
x = TimeDistributed(Flatten())(tf.expand_dims(x, axis=1))  # Add time dimension back
x = Dense(128, activation='relu')(x)

# LSTM / GRU for time-series features
y = GRU(64, return_sequences=True)(feature_input)
y = GRU(128, return_sequences=False)(y)
y = Dense(128, activation='relu')(y)

# Merge the two branches
# Flatten the time dimension of the ConvLSTM2D branch output
x = tf.squeeze(x, axis=1)  # Remove the time dimension (from (None, 1, 128) to (None, 128))

# Concatenate the branches
merged = Concatenate()([x, y])

merged = Dense(256, activation='relu')(merged)

# Outputs for the next 24 hours
output_lat_lon = Dense(48 * 2, activation='linear', name="output_lat_lon")(merged)  # 48 time steps, 2 units each
output_pressure = Dense(48, activation='linear', name="output_pressure")(merged)   # 48 time steps, 1 unit each
output_wind_speed = Dense(48, activation='linear', name="output_wind_speed")(merged)  # 48 time steps, 1 unit each

# Reshape the outputs back to time-step format
output_lat_lon = tf.reshape(output_lat_lon, (-1, 48, 2), name="final_lat_lon")
output_pressure = tf.reshape(output_pressure, (-1, 48, 1), name="final_ecp")
output_wind_speed = tf.reshape(output_wind_speed, (-1, 48, 1), name="final_msw")

# Final model
model = Model(inputs=[image_input, feature_input],
              outputs=[output_lat_lon, output_pressure, output_wind_speed])

# Model summary
model.summary()

# Custom loss function for (latitude, longitude) using Euclidean distance
def euclidean_distance_loss(y_true, y_pred):
    # Split latitude and longitude from y_true and y_pred
    lon_true = y_true[..., 0]
    lat_true = y_true[..., 1]

    lon_pred = y_pred[..., 0]
    lat_pred = y_pred[..., 1]

    # Compute squared differences
    squared_diffs_lat = K.square(lat_true - lat_pred)
    squared_diffs_lon = K.square(lon_true - lon_pred)

    # Compute Euclidean distance
    squared_euclidean_distance = squared_diffs_lat + squared_diffs_lon
    euclidean_distance = K.sqrt(squared_euclidean_distance)

    # Return the mean of the Euclidean distance
    return K.mean(euclidean_distance)

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss={
        "tf.reshape": "mean_squared_error", # MSE/Custom loss for latitude/longitude
        "tf.reshape_1": "mean_squared_error",    # MSE for pressure
        "tf.reshape_2": "mean_squared_error"   # MSE for wind speed
    },
    metrics={
        "tf.reshape": ["mae"],                 # MAE for latitude/longitude
        "tf.reshape_1": ["mae"],                # MAE for pressure
        "tf.reshape_2": ["mae"]               # MAE for wind speed
    }
)

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
    batch_size=8,
    callbacks=[cp, es]
)

df = pd.DataFrame(history.history)
df.to_csv("new_0.5hr_24hr_24hr_GRU_MSE_8.csv")

preds = model.predict([test_images_series_n, test_positions_series_n])

model.load_weights(r'new2_0.5hr_24hr_24hr_GRU_MSE_8.keras')

test_results = model.evaluate([test_images_series_n, test_positions_series_n], [test_labels_lon_lat, test_labels_ecp, test_labels_msw])

# Multiply each value by 100 and round to 2 decimal places
percentage_values = [round(value * 100, 2) for value in test_results]

# Create a dictionary from the processed values
row_dict = dict(zip(test_df.columns, percentage_values))

# Convert the dictionary to a DataFrame and concatenate it to the original DataFrame
new_row_df = pd.DataFrame([row_dict])
test_df = pd.concat([test_df, new_row_df], ignore_index=True)

# Print the updated DataFrame
print(test_df)

test_df.to_csv(r"D:\ML_DL_Projects\Notebooks\new_0.5hr_24hr_24hr_GRU_MSE_8.csv")

np.save('0.5hr_24hr_24hr_GRU_MSE_8@LONG_LAT.npy', preds[0])  # longitude-latitude
np.save('0.5hr_24hr_24hr_GRU_MSE_8@ECP.npy', preds[1])      # ECP
np.save('0.5hr_24hr_24hr_GRU_MSE_8@MSW.npy', preds[2])     # MSW

import glob

# List of CSV file paths
csv_files = [r"D:\ML_DL_Projects\Notebooks\0.5hr_12hr_3hr.csv",
             r"D:\ML_DL_Projects\Notebooks\0.5hr_24hr_6hr.csv",
             r"D:\ML_DL_Projects\Notebooks\1hr_24hr_6hr.csv"]

# Create an empty list to store dataframes
dataframes = []

# Loop through each file and read it into a dataframe
for file in csv_files:
    df = pd.read_csv(file)  # Read the CSV file
    dataframes.append(df)   # Append the dataframe to the list

# Concatenate all dataframes into a single dataframe
combined_df = pd.concat(dataframes, ignore_index=True)

# Save the combined dataframe to a new CSV file
combined_df.to_csv(r"D:\ML_DL_Projects\Notebooks\combined_file.csv", index=False)
