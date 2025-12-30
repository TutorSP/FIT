# Custom loss function for track (LAT, LON) using Euclidean distance
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
