import csv
import os


def save_obs_to_csv(file_name, timestamp, num_pods, desired_replicas, cpu_usage, mem_usage,
                    traffic_in, traffic_out, latency, lstm_1_step, lstm_5_step):
    file_exists = os.path.isfile(file_name) and os.path.getsize(file_name) > 0
    file = open(file_name, 'a+', newline='')
    with file:
        fields = ['date', 'num_pods', 'cpu', 'mem', 'desired_replicas',
                  'traffic_in', 'traffic_out', 'latency', 'lstm_1_step', 'lstm_5_step']
        writer = csv.DictWriter(file, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {'date': timestamp,
             'num_pods': int("{}".format(num_pods)),
             'cpu': int("{}".format(cpu_usage)),
             'mem': int("{}".format(mem_usage)),
             'desired_replicas': int("{}".format(desired_replicas)),
             'traffic_in': int("{}".format(traffic_in)),
             'traffic_out': int("{}".format(traffic_out)),
             'latency': float("{:.3f}".format(latency)),
             'lstm_1_step': int("{}".format(lstm_1_step)),
             'lstm_5_step': int("{}".format(lstm_5_step))}
        )


def save_to_csv(file_name, episode, avg_pods, avg_latency, reward, execution_time):
    file_exists = os.path.isfile(file_name) and os.path.getsize(file_name) > 0
    file = open(file_name, 'a+', newline='')
    with file:
        fields = ['episode', 'avg_pods', 'avg_latency', 'reward', 'execution_time']
        writer = csv.DictWriter(file, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {'episode': episode,
             'avg_pods': float("{:.2f}".format(avg_pods)),
             'avg_latency': float("{:.4f}".format(avg_latency)),
             'reward': float("{:.2f}".format(reward)),
             'execution_time': float("{:.2f}".format(execution_time))})


def get_cost_reward(deployment_list):
    reward = 0
    for d in deployment_list:
        if d.num_pods == d.desired_replicas:
            reward += 1
    return reward


def get_latency_reward_robot_shop(ID_web, deployment_list):
    reward = float(deployment_list[ID_web].latency)
    if reward > 3000.0:
        return -3000
    return -reward


def get_num_pods(deployment_list):
    return sum(d.num_pods for d in deployment_list)