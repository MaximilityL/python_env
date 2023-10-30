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


def main():
    bag_file = '/home/maxim/Workspaces/python_env/data_distributions_ws/src/2023-10-30-11-53-38.bag'
    desired_topics = ['/mavros/euler', '/mavros/local_position/odom', '/mavros/local_position/pose']
    reference_topic = '/mavros/euler'
    data_topic1 = '/mavros/euler'
    data_topic2 = '/mavros/local_position/odom'

    data_dict, time_dict = extract_data_and_timestamps(bag_file, desired_topics)
    
    pitch_interp, velocityx_interp, time_vector = extract_data_arrays(data_dict, time_dict, data_topic1, data_topic2, reference_topic)

    plot_data(pitch_interp, velocityx_interp, time_vector)




if __name__ == "__main__":
    main()
