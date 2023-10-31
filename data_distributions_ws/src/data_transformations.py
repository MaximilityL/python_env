import rosbag
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import matplotlib.gridspec as gridspec


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
    pitch_interp = np.interp(time_vector, time_dict[data_topic1], [msg.vector.y for msg in pitch_data])
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

def standartisize_to_standard_normal(data_array):
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

def plot_two_graphs(y1, y2, time1, y3, y4, time2):
    # Create a figure and two subplots in a single window
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # Plot the first graph (y1 vs. time1) in the top subplot (ax1)
    ax1.plot(time1, y1, label='Pitch', color='b')
    ax1.plot(time1, y2, label='Velocity', color='g')

    # ax1.set_ylabel('Y1', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # Plot the second graph (y2 vs. time2) in the bottom subplot (ax2)
    ax2.plot(time2, y3, label='Pitch', color='b')
    ax2.plot(time2, y4, label='Velocity', color='g')

    ax2.set_xlabel('Time')
    # ax2.set_ylabel('Y2', color='g')
    ax2.tick_params(axis='y', labelcolor='g')

    # Add grid to both subplots
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax2.grid(True, linestyle='--', alpha=0.6)

    # Add legends to both subplots
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left')

    # Display the plot with both subplots
    plt.show()

def standartisizeAndPrintPitchAndVelocity(pitch_cut,velocityx_cut, time_cut):
    pitch_standartisized = standartisize_to_standard_normal(pitch_cut)
    velocityx_stadartisized = standartisize_to_standard_normal(velocityx_cut)

    print("RMSE between data1 and data2:", calculate_rmse(pitch_standartisized, velocityx_stadartisized))
    print("Pearson Correlation Criteria:", calculate_pearson_correlation(pitch_standartisized, velocityx_stadartisized))

    plot_data(pitch_cut, velocityx_cut, time_cut)
    plot_data(pitch_standartisized, velocityx_stadartisized, time_cut)

    return pitch_standartisized, velocityx_stadartisized

    # Create a figure and two subplots in a single window
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # Plot the first dataset (x1, y1) in the top subplot (ax1)
    ax1.plot(time1, y1, label='Dataset 1', color='b')
    ax1.set_ylabel('Y1', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # Plot the second dataset (x2, y2) in the bottom subplot (ax2)
    ax2.plot(time2, y2, label='Dataset 2', color='g')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Y2', color='g')
    ax2.tick_params(axis='y', labelcolor='g')

    # Add grid to both subplots
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax2.grid(True, linestyle='--', alpha=0.6)

    # Add legends to both subplots
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left')

    # Display the plot with both subplots
    plt.show()

def custom_min_max_scaling(dataset, feature_range=(0, 1)):
    min_val = np.min(dataset, axis=0)
    max_val = np.max(dataset, axis=0)
    feature_min, feature_max = feature_range

    # Calculate the scaling factors for each feature
    scale = (feature_max - feature_min) / (max_val - min_val)
    min_scaled = feature_min - min_val * scale

    # Apply the scaling to the dataset
    scaled_data = (dataset - min_val) * scale + min_scaled

    return scaled_data

def calculate_rmse_and_pearson_at_intervals(pitch_interp, velocityx_interp, time_vector, interval, increment, time_shift):
    rmse_values = []  # Initialize an empty list to store RMSE values
    c_values = []
    rmse_times = []
    pearson_correlation_values = [] # Initialize an empty list to store pearson values
    for start_time in np.arange(0, time_vector[-1] - interval - time_shift, increment):
        end_time = start_time + interval
        start_idx = np.searchsorted(time_vector, start_time)
        end_idx = np.searchsorted(time_vector, end_time)
        idx_delta = end_idx - start_idx
        start_shifted_idx = np.searchsorted(time_vector, start_time + time_shift)
        end_shifted_idx = start_shifted_idx + idx_delta
        pitch_segment = pitch_interp[start_idx:end_idx]
        velocityx_segment = velocityx_interp[start_shifted_idx:end_shifted_idx]
        time_segment = time_vector[start_idx:end_idx]

        
        pitch_mean, pitch_std = calculate_mean_and_std(pitch_segment)
        velocityx_mean, velocityx_std = calculate_mean_and_std(velocityx_segment)



        pitch_segment_standartisized = standartisize_to_standard_normal(pitch_segment)
        velocityx_segment_stadartisized = standartisize_to_standard_normal(velocityx_segment)
        rmse = calculate_rmse(pitch_segment_standartisized, velocityx_segment_stadartisized)
        pearson_correlation = calculate_pearson_correlation(pitch_segment, velocityx_segment)
        if ((pitch_std < 1) & (velocityx_std < 0.5)):
            rmse =0


        pitch_segment_normalized = custom_min_max_scaling(pitch_segment, feature_range=(0, 1))
        velocityx_segment_normalized = custom_min_max_scaling(velocityx_segment, feature_range=(0, 1))

        pitch_normalized_mean, pitch_normalized__std = calculate_mean_and_std(pitch_segment_normalized)
        velocityx_normalized__mean, velocityx_normalized__std = calculate_mean_and_std(velocityx_segment_normalized)
        
        rmse_values.append(rmse)
        print("*****************************")
        print("rmse_stadartisize:", rmse)
        rmse_times.append(time_vector[start_idx] + (time_vector[end_idx] - time_vector[start_idx]) / 2)
        pearson_correlation_values.append(abs(pearson_correlation))
        print("pearson_correlation:", abs(pearson_correlation))
        print("pitch_std:", pitch_std)
        print("velocityx_std:", velocityx_std)
        print("pitch_normalized__std:", pitch_normalized__std)
        print("velocityx_normalized__std:", velocityx_normalized__std)
        
        print("Pitch_std / velocity std:", pitch_std / velocityx_std)
        # if ((pitch_normalized__std / velocityx_normalized__std) > 1):
        #     c = pitch_normalized__std / velocityx_normalized__std
        # else:
        #     c = velocityx_normalized__std / pitch_normalized__std
        # if ( c > 1.25):
        #     c =1

        if ((pitch_std / velocityx_std) > 1):
            c = pitch_std / velocityx_std
        else:
            c = velocityx_std / pitch_std
        # if ( c > 1.25):
        #     c =1

        print("normalized_std divide:", c - 1)
        c_values.append(c - 1)
        print("At time of" , time_vector[start_idx] + (time_vector[end_idx] - time_vector[start_idx]) / 2, " [sec]")

        # plot_two_graphs(pitch_segment, velocityx_segment, time_segment, pitch_segment_standartisized, velocityx_segment_stadartisized, time_segment)
        # plot_two_graphs(pitch_segment, velocityx_segment, time_segment, pitch_segment_normalized, velocityx_segment_normalized, time_segment)
        




    return rmse_values, rmse_times, c_values, pearson_correlation_values

def plot_pitch_velocity_rmse(time_vector, pitch_interp, velocityx_interp, c_values,rmse_values,pearson_values, rmse_times):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot pitch and velocity on the first (top) subplot
    ax1.plot(time_vector, pitch_interp, label='Pitch (Euler Angle)', color='b')
    ax1.plot(time_vector, velocityx_interp, label='Linear Velocity (x)', color='g')
    ax1.set_ylabel('Pitch and Velocity', color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    # Create a secondary y-axis on the right for RMSE in the top plot
    ax2 = ax1.twinx()
    ax2.plot(rmse_times, rmse_values, label='rmse', color='r', marker='o')  # Adjust the 's' parameter for marker size
    ax2.plot(rmse_times, c_values, label='std_divition', color='c', marker='o')  # Adjust the 's' parameter for marker size
    ax2.plot(rmse_times, (rmse_values * c_values) / pearson_values, label='Libman\'s Coefficient', color='y', marker='o')  # Adjust the 's' parameter for marker size

    ax2.set_ylabel('RMSE', color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    # Add grid to both subplots
    ax1.grid(True)
    # ax2.grid(True)

    # Adjust the space between subplots
    plt.subplots_adjust(hspace=0.3)

    # Add a common title for the two subplots
    plt.suptitle('Pitch, Velocity, and RMSE vs. Time')

    plt.legend()
    plt.show()

def plot_pitch_velocity_rmse_and_pearson(time_vector, pitch_interp, velocityx_interp, interpolated_rmse, interpolated_pearson):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot pitch and velocity on the primary y-axis (left)
    ax1.plot(time_vector, pitch_interp, label='Pitch (Euler Angle)', color='b')
    ax1.plot(time_vector, velocityx_interp, label='Linear Velocity (x)', color='g')
    ax1.set_ylabel('Pitch and Velocity', color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    # Add a grid to the top plot
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Create a secondary y-axis on the right for RMSE in the top plot
    ax2 = ax1.twinx()
    ax2.plot(time_vector, interpolated_rmse, label='RMSE', color='r', linestyle='dashed')
    ax2.set_ylabel('RMSE', color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    # Add a legend to the top plot
    ax1.legend(loc="upper left")

    

    # Create a new figure for the bottom plot
    fig, ax3 = plt.subplots(figsize=(12, 6))

    # Add a grid to the top plot
    ax3.grid(True, linestyle='--', alpha=0.6)
    # Plot pitch and velocity on the primary y-axis (left)
    ax3.plot(time_vector, pitch_interp, label='Pitch (Euler Angle)', color='b')
    ax3.plot(time_vector, velocityx_interp, label='Linear Velocity (x)', color='g')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Pitch and Velocity', color='black')
    ax3.tick_params(axis='y', labelcolor='black')

    # Create a secondary y-axis on the right for Pearson in the bottom plot
    ax4 = ax3.twinx()
    ax4.plot(time_vector, interpolated_pearson, label='Pearson', color='m', linestyle='dotted')
    ax4.set_ylabel('Pearson', color='black')
    ax4.tick_params(axis='y', labelcolor='black')

    # Add a legend to the bottom plot
    ax3.legend(loc="upper left")

    plt.show()

def cubic_interpolate(x, y, new_x):
    cubic_interpolator = interp1d(x, y, kind='cubic')
    return cubic_interpolator(new_x)

def main():

    pitch_interp, velocityx_interp, time_vector = getInterpolatedPitchAndVelocityData()

    interval = 5  # Interval for RMSE calculation 
    increment = 0.5  # Increment for time calculation 
    time_shift = 1.0 # Delay between Pitch and velocity
    rmse_values, rmse_times, c_values, pearson_correlation_values = calculate_rmse_and_pearson_at_intervals(pitch_interp, velocityx_interp, time_vector, interval, increment, time_shift)
    

    rmse_values_normalized = custom_min_max_scaling(rmse_values, feature_range=(0, 1))
    c_values_normalized = custom_min_max_scaling(c_values, feature_range=(0, 1))
    

    # plot_pitch_velocity_rmse(time_vector, pitch_interp, velocityx_interp, rmse_values, rmse_times,)
    plot_pitch_velocity_rmse(time_vector, pitch_interp, velocityx_interp, c_values_normalized,rmse_values_normalized, pearson_correlation_values, rmse_times)


    # # Create a time vector for the interpolated RMSE values
    # interpolated_time_vector = np.linspace(time_vector[0], time_vector[-1], len(rmse_values))
    
    # # Perform cubic interpolation for RMSE values
    # interpolated_rmse = cubic_interpolate(interpolated_time_vector, rmse_values, time_vector)

    # # Create a time vector for the interpolated Pearson values
    # interpolated_time_vector = np.linspace(time_vector[0], time_vector[-1], len(pearson_values))
    
    # # Perform cubic interpolation for Pearson values
    # interpolated_pearson = cubic_interpolate(interpolated_time_vector, pearson_values, time_vector)

    # Call the new function to plot the data
    # plot_pitch_velocity_rmse_and_pearson(time_vector, pitch_interp, velocityx_interp, interpolated_rmse, interpolated_pearson)

if __name__ == "__main__":
    main()
