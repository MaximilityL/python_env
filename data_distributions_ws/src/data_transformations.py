import rosbag
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def extract_data_and_timestamps(bag_file, desired_topics):
    data_dict = {topic: [] for topic in desired_topics}
    time_dict = {topic: [] for topic in desired_topics}

    bag = rosbag.Bag(bag_file)

    # Create a dictionary to store the start time for each topic
    start_time_dict = {topic: None for topic in desired_topics}

    for topic, msg, t in bag.read_messages(topics=desired_topics):
        # If the start time for this topic is not set, set it to the current message timestamp
        if start_time_dict[topic] is None:
            start_time_dict[topic] = t.to_sec()

        # Calculate the adjusted timestamp starting from 0
        adjusted_timestamp = t.to_sec() - start_time_dict[topic]

        data_dict[topic].append(msg)
        time_dict[topic].append(adjusted_timestamp)

    bag.close()

    return data_dict, time_dict

def extract_data_arrays(data_dict, time_dict, data_topic1, data_topic2, reference_topic):
    time_vector = time_dict[reference_topic]
    pitch_data = data_dict[data_topic1]
    velocityx_data = data_dict[data_topic2]

    # Interpolate data to match the time_vector
    pitch_interp = np.interp(time_vector, time_dict[data_topic1], [msg.vector.x for msg in pitch_data])
    velocityx_interp = np.interp(time_vector, time_dict[data_topic2], [msg.twist.twist.linear.x for msg in velocityx_data])

    return pitch_interp, velocityx_interp, time_vector

def plot_data(pitch_interp, velocityx_interp, time_vector):
    # Create a new figure explicitly
    plt.figure(figsize=(12, 6))
    plt.plot(time_vector, pitch_interp, label='Pitch (Euler Angle)')
    plt.plot(time_vector, velocityx_interp, label='Linear Velocity (x)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Value')
    plt.title('Pitch and Linear Velocity vs. Time')
    plt.legend()
    plt.ioff()
    plt.show()

def cut_data_to_time_frame(pitch_array, velocityx_array, time_vector, start_time, end_time):
    start_idx = np.searchsorted(time_vector, start_time)
    end_idx = np.searchsorted(time_vector, end_time)

    pitch_cut = pitch_array[start_idx:end_idx]
    velocityx_cut = velocityx_array[start_idx:end_idx]
    time_cut = time_vector[start_idx:end_idx]

    return pitch_cut, velocityx_cut, time_cut

def calculate_mean_and_std(data_array):
    mean_value = np.mean(data_array)
    std_value = np.std(data_array)
    return mean_value, std_value

def normalize_to_standard_normal(data_array):
    mean_value = np.mean(data_array)
    std_value = np.std(data_array)
    
    # Calculate the z-score
    z_score = (data_array - mean_value) / std_value
    
    return z_score

def calculate_pearson_correlation(data1, data2):
    # Calculate the means of both datasets
    mean_data1 = np.mean(data1)
    mean_data2 = np.mean(data2)

    # Calculate the standard deviations of both datasets
    std_data1 = np.std(data1)
    std_data2 = np.std(data2)

    # Calculate the covariance between data1 and data2
    covariance = np.cov(data1, data2)[0, 1]

    # Calculate Pearson's correlation coefficient
    correlation_coefficient = covariance / (std_data1 * std_data2)

    return correlation_coefficient

def calculate_rmse(data1, data2):
    # Ensure data1 and data2 have the same length
    if len(data1) != len(data2):
        raise ValueError("Data arrays must have the same length")

    # Calculate the squared differences between corresponding elements
    squared_diff = (data1 - data2) ** 2

    # Calculate the mean of the squared differences
    mean_squared_diff = np.mean(squared_diff)

    # Calculate the square root of the mean squared difference (RMSE)
    # rmse = np.sqrt(mean_squared_diff)
    rmse = mean_squared_diff

    return rmse

def getInterpolatedPitchAndVelocityData():
    bag_file = '/home/maxim/Workspaces/python_env/data_distributions_ws/src/2023-10-30-11-53-38.bag'
    desired_topics = ['/mavros/euler', '/mavros/local_position/odom', '/mavros/local_position/pose']
    reference_topic = '/mavros/euler'
    data_topic1 = '/mavros/euler'
    data_topic2 = '/mavros/local_position/odom'

    data_dict, time_dict = extract_data_and_timestamps(bag_file, desired_topics)
    pitch_interp, velocityx_interp, time_vector = extract_data_arrays(data_dict, time_dict, data_topic1, data_topic2, reference_topic)
    return pitch_interp, velocityx_interp, time_vector

def cutAndCalcAndPrintMeanAndSTD(pitch_interp, velocityx_interp, time_vector, start_time, end_time):
    pitch_cut, velocityx_cut, time_cut = cut_data_to_time_frame(pitch_interp, velocityx_interp, time_vector, start_time, end_time)
    
    print("Pitch Mean:", calculate_mean_and_std(pitch_cut)[0])
    print("Pitch Standard Deviation:", calculate_mean_and_std(pitch_cut)[1])
    print("Velocityx Mean:", calculate_mean_and_std(velocityx_cut)[0])
    print("Velocityx Standard Deviation:", calculate_mean_and_std(velocityx_cut)[1])
    return pitch_cut, velocityx_cut, time_cut

def normalizeAndPrintPitchAndVelocity(pitch_cut,velocityx_cut, time_cut):
    pitch_normalized = normalize_to_standard_normal(pitch_cut)
    velocityx_normalized = normalize_to_standard_normal(velocityx_cut)

    print("RMSE between data1 and data2:", calculate_rmse(pitch_normalized, velocityx_normalized))
    print("Pearson Correlation Criteria:", calculate_pearson_correlation(pitch_normalized, velocityx_normalized))

    plot_data(pitch_cut, velocityx_cut, time_cut)
    plot_data(pitch_normalized, velocityx_normalized, time_cut)

    return pitch_normalized, velocityx_normalized

def calculate_rmse_at_intervals(pitch_interp, velocityx_interp, time_vector, interval, increment):
    rmse_values = []  # Initialize an empty list to store RMSE values
   
    for start_time in np.arange(0, time_vector[-1] - interval, increment):
        end_time = start_time + interval
        start_idx = np.searchsorted(time_vector, start_time)
        end_idx = np.searchsorted(time_vector, end_time)

        pitch_segment = pitch_interp[start_idx:end_idx]
        velocityx_segment = velocityx_interp[start_idx:end_idx]
        time_segment = time_vector[start_idx:end_idx]
        pitch_segment_normalized = normalize_to_standard_normal(pitch_segment)
        velocityx_segment_normalized = normalize_to_standard_normal(velocityx_segment)
        # plot_data(pitch_segment_normalized, velocityx_segment_normalized, time_segment)
        rmse = calculate_rmse(pitch_segment_normalized, velocityx_segment_normalized)
        rmse_values.append(rmse)

    return rmse_values

def plot_pitch_velocity_rmse(time_vector, pitch_interp, velocityx_interp, rmse_values):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot pitch_interp and velocityx_interp on the primary y-axis
    ax1.plot(time_vector, pitch_interp, label='Pitch (Euler Angle)', color='b')
    ax1.plot(time_vector, velocityx_interp, label='Linear Velocity (x)', color='g')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Pitch and Velocity', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    
    # Create a secondary y-axis on the right for RMSE
    ax2 = ax1.twinx()

    # Trim the last element of rmse_values to match the dimensions
    rmse_values = rmse_values[:len(time_vector)]

    ax2.plot(time_vector, rmse_values, label='RMSE', color='r', linestyle='dashed')
    ax2.set_ylabel('RMSE', color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    fig.tight_layout()
    plt.title('Pitch, Velocity, and RMSE vs. Time')
    plt.legend()
    plt.show()

def cubic_interpolate(x, y, new_x):
    cubic_interpolator = interp1d(x, y, kind='cubic')
    return cubic_interpolator(new_x)

def main():

    pitch_interp, velocityx_interp, time_vector = getInterpolatedPitchAndVelocityData()

    interval = 3  # Interval for RMSE calculation 
    increment = 1  # Increment for time calculation 
    rmse_values = calculate_rmse_at_intervals(pitch_interp, velocityx_interp, time_vector, interval, increment)
    
    # Create a time vector for the interpolated RMSE values
    interpolated_time_vector = np.linspace(time_vector[0], time_vector[-1], len(rmse_values))
    
    # Perform cubic interpolation for RMSE values
    interpolated_rmse = cubic_interpolate(interpolated_time_vector, rmse_values, time_vector)

    # Call the new function to plot the data
    plot_pitch_velocity_rmse(time_vector, pitch_interp, velocityx_interp, interpolated_rmse)

if __name__ == "__main__":
    main()
