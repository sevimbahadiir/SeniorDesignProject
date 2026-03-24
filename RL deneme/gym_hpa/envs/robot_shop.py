import csv
from datetime import datetime
import logging
import time
from statistics import mean

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

# MIN and MAX Replication
MIN_REPLICATION = 1
MAX_REPLICATION = 8

MAX_STEPS = 25  # MAX Number of steps per episode

# Possible Actions (Discrete)
ACTION_DO_NOTHING = 0
ACTION_ADD_1_REPLICA = 1
ACTION_ADD_2_REPLICA = 2
ACTION_ADD_3_REPLICA = 3
ACTION_ADD_4_REPLICA = 4
ACTION_ADD_5_REPLICA = 5
ACTION_ADD_6_REPLICA = 6
ACTION_ADD_7_REPLICA = 7
ACTION_TERMINATE_1_REPLICA = 8
ACTION_TERMINATE_2_REPLICA = 9
ACTION_TERMINATE_3_REPLICA = 10
ACTION_TERMINATE_4_REPLICA = 11
ACTION_TERMINATE_5_REPLICA = 12
ACTION_TERMINATE_6_REPLICA = 13
ACTION_TERMINATE_7_REPLICA = 14

# Deployments -> Robot Shop dataset services
DEPLOYMENTS = [
    "cartservice",
    "catalogueservice",
    "dispatchservice",
    "paymentservice",
    "ratingsservice",
    "shippingservice",
    "userservice",
    "webservice"
]

# Action Moves
MOVES = [
    "None", "Add-1", "Add-2", "Add-3", "Add-4", "Add-5", "Add-6", "Add-7",
    "Stop-1", "Stop-2", "Stop-3", "Stop-4", "Stop-5", "Stop-6", "Stop-7"
]

# IDs
ID_DEPLOYMENTS = 0
ID_MOVES = 1

ID_cart_service = 0
ID_catalogue_service = 1
ID_dispatch_service = 2
ID_payment_service = 3
ID_ratings_service = 4
ID_shipping_service = 5
ID_user_service = 6
ID_web_service = 7

# Reward objectives
LATENCY = 'latency'
COST = 'cost'

# Penalty tuned to dataset reward scale
INVALID_ACTION_PENALTY = -2.5


def save_to_csv(file_name, episode, avg_pods, avg_latency, total_reward, execution_time):
    header = ['episode', 'avg_pods', 'avg_latency', 'total_reward', 'execution_time']
    file_exists = False
    try:
        with open(file_name, 'r', newline=''):
            file_exists = True
    except FileNotFoundError:
        file_exists = False

    with open(file_name, 'a+', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(header)
        writer.writerow([episode, avg_pods, avg_latency, total_reward, execution_time])


def get_num_pods(deployment_list):
    return sum(d.num_pods for d in deployment_list)


class SimulatedDeployment:
    def __init__(self, name, namespace="robot-shop", min_pods=1, max_pods=8):
        self.name = name
        self.namespace = namespace
        self.min_pods = min_pods
        self.max_pods = max_pods

        self.num_pods = min_pods
        self.num_previous_pods = min_pods
        self.desired_replicas = min_pods

        self.cpu_usage = 0.0
        self.mem_usage = 0.0
        self.received_traffic = 0.0
        self.transmit_traffic = 0.0
        self.latency = 0.0

    def print_deployment(self):
        logging.info(
            f"[Deployment] {self.name} | num_pods={self.num_pods} | "
            f"desired={self.desired_replicas} | namespace={self.namespace}"
        )

    def deploy_pod_replicas(self, replicas, env):
        target = self.num_pods + replicas
        if target > self.max_pods:
            env.constraint_max_pod_replicas = True
            self.num_previous_pods = self.num_pods
            self.num_pods = self.max_pods
        else:
            self.num_previous_pods = self.num_pods
            self.num_pods = target

        self.update_replicas()

    def terminate_pod_replicas(self, replicas, env):
        target = self.num_pods - replicas
        if target < self.min_pods:
            env.constraint_min_pod_replicas = True
            self.num_previous_pods = self.num_pods
            self.num_pods = self.min_pods
        else:
            self.num_previous_pods = self.num_pods
            self.num_pods = target

        self.update_replicas()

    def update_replicas(self):
        self.desired_replicas = self.num_pods

    def update_obs_k8s(self):
        return


def get_online_boutique_deployment_list(k8s, min_pods, max_pods):
    return [
        SimulatedDeployment(name, "robot-shop", min_pods, max_pods)
        for name in DEPLOYMENTS
    ]


class RobotShop(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, k8s=False, goal_reward="cost", waiting_period=0.3):
        super(RobotShop, self).__init__()

        self.k8s = k8s
        self.name = "online_boutique_gym"
        self.__version__ = "0.0.1"
        self.seed()
        self.goal_reward = goal_reward
        self.waiting_period = waiting_period

        if self.k8s:
            logging.warning(
                "[Init] This rewritten environment is intended for offline/simulation mode. "
                "Please use k8s=False."
            )

        logging.info("[Init] Env: {} | K8s: {} | Version {} |".format(
            self.name, self.k8s, self.__version__
        ))

        self.current_step = 0
        self.num_actions = 15
        self.action_space = spaces.MultiDiscrete([len(DEPLOYMENTS), self.num_actions])

        self.min_pods = MIN_REPLICATION
        self.max_pods = MAX_REPLICATION
        self.num_apps = 1

        self.deploymentList = get_online_boutique_deployment_list(self.k8s, self.min_pods, self.max_pods)

        for d in self.deploymentList:
            d.print_deployment()

        self.file_results = f"results_{goal_reward}.csv"
        self.obs_csv = f"{self.name}_{goal_reward}_observation.csv"

        self.df = pd.read_csv("/Users/sevvalyavas/Desktop/RL deneme/dataset/robot_shop_rl_dataset_final.csv")

        self.feature_columns = []
        for dep in DEPLOYMENTS:
            self.feature_columns.extend([
                f"{dep}_num_pods",
                f"{dep}_desired_replicas",
                f"{dep}_cpu_usage",
                f"{dep}_mem_usage",
                f"{dep}_traffic_in",
                f"{dep}_traffic_out",
                f"{dep}_latency",
            ])

        missing_columns = [col for col in self.feature_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Dataset is missing required columns: {missing_columns}")

        for col in self.feature_columns:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        self.df = self.df.dropna(subset=self.feature_columns).reset_index(drop=True)

        if self.df.empty:
            raise ValueError("Dataset is empty after cleaning.")

        self.pod_columns = [f"{dep}_num_pods" for dep in DEPLOYMENTS]
        self.pod_matrix = self.df[self.pod_columns].to_numpy(dtype=np.float32)

        self.observation_space = self.get_observation_space()

        self.total_reward = None
        self.avg_pods = []
        self.avg_latency = []

        self.episode_over = False
        self.info = {}

        self.constraint_max_pod_replicas = False
        self.constraint_min_pod_replicas = False
        self.cost_weight = 0

        self.time_start = 0
        self.execution_time = 0
        self.episode_count = 0

        self.current_sample = None

    def step(self, action):
        if self.current_step == 1:
            if not self.k8s:
                self.simulation_update()
            self.time_start = time.time()

        n = int(action[ID_DEPLOYMENTS])

        self.take_action(int(action[ID_MOVES]), n)

        if self.k8s:
            if action[ID_MOVES] != ACTION_DO_NOTHING \
                    and self.constraint_min_pod_replicas is False \
                    and self.constraint_max_pod_replicas is False:
                time.sleep(self.waiting_period)

        if self.k8s:
            for d in self.deploymentList:
                d.update_obs_k8s()
        else:
            self.simulation_update()

        reward = self.get_reward
        self.total_reward += reward

        self.avg_pods.append(get_num_pods(self.deploymentList))
        self.avg_latency.append(self._get_app_latency())

        logging.info(
            '[Step {}] | Action (Deployment): {} | Action (Move): {} | Reward: {} | Total Reward: {}'.format(
                self.current_step, DEPLOYMENTS[action[0]], MOVES[action[1]], reward, self.total_reward
            )
        )

        ob = self.get_state()
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.save_obs_to_csv(self.obs_csv, np.array(ob), date, self._get_app_latency())

        self.info = dict(
            total_reward=self.total_reward,
            app_latency=self._get_app_latency(),
            total_pods=get_num_pods(self.deploymentList)
        )

        self.constraint_max_pod_replicas = False
        self.constraint_min_pod_replicas = False

        if self.current_step == MAX_STEPS:
            self.episode_count += 1
            self.execution_time = time.time() - self.time_start

            save_to_csv(
                self.file_results,
                self.episode_count,
                mean(self.avg_pods) if self.avg_pods else 0,
                mean(self.avg_latency) if self.avg_latency else 0,
                self.total_reward,
                self.execution_time
            )

        terminated = self.episode_over
        truncated = False

        return np.array(ob, dtype=np.float32), reward, terminated, truncated, self.info

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.episode_over = False
        self.total_reward = 0
        self.avg_pods = []
        self.avg_latency = []

        self.constraint_max_pod_replicas = False
        self.constraint_min_pod_replicas = False

        self.deploymentList = get_online_boutique_deployment_list(self.k8s, self.min_pods, self.max_pods)

        if seed is not None:
            sample_index = self.np_random.integers(0, len(self.df))
            self.current_sample = self.df.iloc[int(sample_index)]
        else:
            self.current_sample = self.df.sample(n=1, random_state=None).iloc[0]

        self._apply_sample_to_deployments(self.current_sample, initialize_previous=True)

        obs = np.array(self.get_state(), dtype=np.float32)
        info = {}

        return obs, info

    def render(self):
        return

    def take_action(self, action, id):
        self.current_step += 1

        if self.current_step == MAX_STEPS:
            self.episode_over = True

        if action == ACTION_DO_NOTHING:
            pass
        elif action == ACTION_ADD_1_REPLICA:
            self.deploymentList[id].deploy_pod_replicas(1, self)
        elif action == ACTION_ADD_2_REPLICA:
            self.deploymentList[id].deploy_pod_replicas(2, self)
        elif action == ACTION_ADD_3_REPLICA:
            self.deploymentList[id].deploy_pod_replicas(3, self)
        elif action == ACTION_ADD_4_REPLICA:
            self.deploymentList[id].deploy_pod_replicas(4, self)
        elif action == ACTION_ADD_5_REPLICA:
            self.deploymentList[id].deploy_pod_replicas(5, self)
        elif action == ACTION_ADD_6_REPLICA:
            self.deploymentList[id].deploy_pod_replicas(6, self)
        elif action == ACTION_ADD_7_REPLICA:
            self.deploymentList[id].deploy_pod_replicas(7, self)
        elif action == ACTION_TERMINATE_1_REPLICA:
            self.deploymentList[id].terminate_pod_replicas(1, self)
        elif action == ACTION_TERMINATE_2_REPLICA:
            self.deploymentList[id].terminate_pod_replicas(2, self)
        elif action == ACTION_TERMINATE_3_REPLICA:
            self.deploymentList[id].terminate_pod_replicas(3, self)
        elif action == ACTION_TERMINATE_4_REPLICA:
            self.deploymentList[id].terminate_pod_replicas(4, self)
        elif action == ACTION_TERMINATE_5_REPLICA:
            self.deploymentList[id].terminate_pod_replicas(5, self)
        elif action == ACTION_TERMINATE_6_REPLICA:
            self.deploymentList[id].terminate_pod_replicas(6, self)
        elif action == ACTION_TERMINATE_7_REPLICA:
            self.deploymentList[id].terminate_pod_replicas(7, self)
        else:
            logging.info('[Take Action] Unrecognized Action: ' + str(action))

    @property
    def get_reward(self):
        if self.constraint_max_pod_replicas:
            if self.goal_reward == COST:
                return -1
            elif self.goal_reward == LATENCY:
                return INVALID_ACTION_PENALTY

        if self.constraint_min_pod_replicas:
            if self.goal_reward == COST:
                return -1
            elif self.goal_reward == LATENCY:
                return INVALID_ACTION_PENALTY

        reward = self.calculate_reward()
        return reward

    def get_state(self):
        ob = []
        for dep in self.deploymentList:
            ob.extend([
                float(dep.num_pods),
                float(dep.desired_replicas),
                float(dep.cpu_usage),
                float(dep.mem_usage),
                float(dep.received_traffic),
                float(dep.transmit_traffic),
                float(dep.latency),
            ])
        return tuple(ob)

    def get_observation_space(self):
        low = []
        high = []

        for dep in DEPLOYMENTS:
            low.extend([
                self.min_pods,
                self.min_pods,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0
            ])

            high.extend([
                self.max_pods,
                self.max_pods,
                float(self.df[f"{dep}_cpu_usage"].max()),
                float(self.df[f"{dep}_mem_usage"].max()),
                float(self.df[f"{dep}_traffic_in"].max()),
                float(self.df[f"{dep}_traffic_out"].max()),
                float(self.df[f"{dep}_latency"].max())
            ])

        return spaces.Box(
            low=np.array(low, dtype=np.float32),
            high=np.array(high, dtype=np.float32),
            dtype=np.float32
        )

    def _get_app_latency(self):
        latencies = [d.latency for d in self.deploymentList]
        return float(np.mean(latencies))

    def calculate_reward(self):
        reward = 0

        if self.goal_reward == COST:
            matches = 0
            for dep in self.deploymentList:
                if dep.num_pods == dep.desired_replicas:
                    matches += 1
            reward = matches / len(self.deploymentList)

        elif self.goal_reward == LATENCY:
            app_latency = self._get_app_latency()
            reward = -app_latency

        return reward

    def simulation_update(self):
        if self.current_step == 1 or self.current_sample is None:
            sample = self.df.sample(n=1, random_state=None).iloc[0]
        else:
            current_pods = np.array([d.num_pods for d in self.deploymentList], dtype=np.float32)

            distances = np.abs(self.pod_matrix - current_pods).sum(axis=1)
            min_distance = distances.min()
            candidate_indices = np.where(distances == min_distance)[0]

            chosen_index = np.random.choice(candidate_indices)
            sample = self.df.iloc[chosen_index]

        self.current_sample = sample
        self._apply_sample_to_deployments(sample, initialize_previous=False)
        return

    def _apply_sample_to_deployments(self, sample, initialize_previous=False):
        for i, dep_name in enumerate(DEPLOYMENTS):
            dep = self.deploymentList[i]

            sampled_num_pods = int(round(float(sample[f"{dep_name}_num_pods"])))
            sampled_desired = int(round(float(sample[f"{dep_name}_desired_replicas"])))

            sampled_num_pods = max(self.min_pods, min(self.max_pods, sampled_num_pods))
            sampled_desired = max(self.min_pods, min(self.max_pods, sampled_desired))

            if initialize_previous:
                dep.num_previous_pods = sampled_num_pods
            else:
                dep.num_previous_pods = dep.num_pods

            dep.num_pods = sampled_num_pods
            dep.desired_replicas = sampled_desired
            dep.cpu_usage = float(sample[f"{dep_name}_cpu_usage"])
            dep.mem_usage = float(sample[f"{dep_name}_mem_usage"])
            dep.received_traffic = float(sample[f"{dep_name}_traffic_in"])
            dep.transmit_traffic = float(sample[f"{dep_name}_traffic_out"])
            dep.latency = float(sample[f"{dep_name}_latency"])

    def save_obs_to_csv(self, obs_file, obs, date, latency):
        fields = ['date']
        for d in self.deploymentList:
            fields.extend([
                d.name + '_num_pods',
                d.name + '_desired_replicas',
                d.name + '_cpu_usage',
                d.name + '_mem_usage',
                d.name + '_traffic_in',
                d.name + '_traffic_out',
                d.name + '_latency'
            ])

        file_exists = False
        try:
            with open(obs_file, 'r', newline=''):
                file_exists = True
        except FileNotFoundError:
            file_exists = False

        row = {'date': date}
        idx = 0
        for d in self.deploymentList:
            row[d.name + '_num_pods'] = int(round(float(obs[idx]))); idx += 1
            row[d.name + '_desired_replicas'] = int(round(float(obs[idx]))); idx += 1
            row[d.name + '_cpu_usage'] = float(obs[idx]); idx += 1
            row[d.name + '_mem_usage'] = float(obs[idx]); idx += 1
            row[d.name + '_traffic_in'] = float(obs[idx]); idx += 1
            row[d.name + '_traffic_out'] = float(obs[idx]); idx += 1
            row[d.name + '_latency'] = float(obs[idx]); idx += 1

        with open(obs_file, 'a+', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        return
