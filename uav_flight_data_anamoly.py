import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the flight test data from a CSV file
df = pd.read_csv('flight_test_data.csv')

# Define a function to identify anomalies in the data
def identify_anomalies(data, window_size, threshold):
    # Calculate the rolling mean and standard deviation of the data
    rolling_mean = data.rolling(window_size).mean()
    rolling_std = data.rolling(window_size).std()
    
    # Calculate the upper and lower bounds for the data
    upper_bound = rolling_mean + threshold * rolling_std
    lower_bound = rolling_mean - threshold * rolling_std
    
    # Identify any data points that fall outside the bounds
    anomalies = np.where((data < lower_bound) | (data > upper_bound), True, False)
    
    return anomalies

# Identify anomalies in the altitude data
alt_anomalies = identify_anomalies(df['altitude'], window_size=10, threshold=2)

# Plot the altitude data with anomalies highlighted
fig, ax = plt.subplots(figsize=(10, 5))
df.plot(x='time', y='altitude', ax=ax)
ax.fill_between(df['time'], df['altitude'], where=alt_anomalies, color='red', alpha=0.2)
ax.set_ylabel('Altitude (m)')
plt.show()

# Identify anomalies in the airspeed data
airspeed_anomalies = identify_anomalies(df['airspeed'], window_size=10, threshold=2)

# Plot the airspeed data with anomalies highlighted
fig, ax = plt.subplots(figsize=(10, 5))
df.plot(x='time', y='airspeed', ax=ax)
ax.fill_between(df['time'], df['airspeed'], where=airspeed_anomalies, color='red', alpha=0.2)
ax.set_ylabel('Airspeed (m/s)')
plt.show()

# Identify anomalies in the pitch angle data
pitch_anomalies = identify_anomalies(df['pitch_angle'], window_size=10, threshold=2)

# Plot the pitch angle data with anomalies highlighted
fig, ax = plt.subplots(figsize=(10, 5))
df.plot(x='time', y='pitch_angle', ax=ax)
ax.fill_between(df['time'], df['pitch_angle'], where=pitch_anomalies, color='red', alpha=0.2)
ax.set_ylabel('Pitch Angle (deg)')
plt.show()
