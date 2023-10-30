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
    plt.figure(figsize=(12, 6))
    plt.plot(time_vector, pitch_interp, label='Pitch (Euler Angle)')
    plt.plot(time_vector, velocityx_interp, label='Linear Velocity (x)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Value')
    plt.title('Pitch and Linear Velocity vs. Time')
    plt.legend()
    plt.show()

def cut_data_to_time_frame(pitch_array, velocityx_array, time_vector, start_time, end_time):
    start_idx = np.searchsorted(time_vector, start_time)
    end_idx = np.searchsorted(time_vector, end_time)

    pitch_cut = pitch_array[start_idx:end_idx]
    velocityx_cut = velocityx_array[start_idx:end_idx]
    time_cut = time_vector[start_idx:end_idx]

    return pitch_cut, velocityx_cut, time_cut


def main():
    bag_file = '/home/maxim/Workspaces/python_env/data_distributions_ws/src/2023-10-30-11-53-38.bag'
    desired_topics = ['/mavros/euler', '/mavros/local_position/odom', '/mavros/local_position/pose']
    reference_topic = '/mavros/euler'
    data_topic1 = '/mavros/euler'
    data_topic2 = '/mavros/local_position/odom'

    data_dict, time_dict = extract_data_and_timestamps(bag_file, desired_topics)
    
    pitch_interp, velocityx_interp, time_vector = extract_data_arrays(data_dict, time_dict, data_topic1, data_topic2, reference_topic)

    # Define the start and end times for cutting
    start_time = 10.0  # Adjust this as needed
    end_time = 20.0  # Adjust this as needed

    pitch_cut, velocityx_cut, time_cut = cut_data_to_time_frame(pitch_interp, velocityx_interp, time_vector, start_time, end_time)


    plot_data(pitch_cut, velocityx_cut, time_cut)




if __name__ == "__main__":
    main()
