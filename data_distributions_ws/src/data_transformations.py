import rosbag
import numpy as np
import matplotlib.pyplot as plt

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

def main():
    
    pitch_interp, velocityx_interp, time_vector = getInterpolatedPitchAndVelocityData()
    start_time = 33.0
    end_time = start_time + 3
    
    pitch_cut, velocityx_cut, time_cut = cutAndCalcAndPrintMeanAndSTD(pitch_interp, velocityx_interp, time_vector, start_time, end_time)

    normalizeAndPrintPitchAndVelocity(pitch_cut,velocityx_cut, time_cut)

if __name__ == "__main__":
    main()
