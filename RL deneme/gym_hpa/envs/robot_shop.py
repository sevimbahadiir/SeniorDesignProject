import os
import csv
import datetime
from datetime import datetime
import logging
import time
from statistics import mean

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from gym_hpa.envs.deployment import get_max_cpu, get_max_mem, get_max_traffic, get_robot_shop_deployment_list
from gym_hpa.envs.util import save_to_csv, get_num_pods, get_cost_reward, get_latency_reward_robot_shop

MIN_REPLICATION = 1
MAX_REPLICATION = 8
MAX_STEPS = 25

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

DEPLOYMENTS = ["cart", "catalogue", "dispatch", "payment",
               "ratings", "shipping", "user", "web"]

MOVES = ["None", "Add-1", "Add-2", "Add-3", "Add-4", "Add-5", "Add-6", "Add-7",
         "Stop-1", "Stop-2", "Stop-3", "Stop-4", "Stop-5", "Stop-6", "Stop-7"]

ID_DEPLOYMENTS = 0
ID_MOVES = 1

ID_cart = 0
ID_catalogue = 1
ID_dispatch = 2
ID_payment = 3
ID_ratings = 4
ID_shipping = 5
ID_user = 6
ID_web = 7

LATENCY = 'latency'
COST = 'cost'


class RobotShop(gym.Env):
    """Horizontal Scaling for Robot Shop in Kubernetes - an OpenAI gym environment"""

    metadata = {'render_modes': ['human', 'ansi', 'array']}

    def __init__(self, k8s=False, goal_reward="cost", waiting_period=0.3):
        super(RobotShop, self).__init__()

        self.k8s = k8s
        self.name = "robot_shop_gym"
        self.__version__ = "0.0.1"
        self.goal_reward = goal_reward
        self.waiting_period = waiting_period

        logging.info("[Init] Env: {} | K8s: {} | Version {} |".format(
            self.name, self.k8s, self.__version__))

        self.current_step = 0
        self.num_actions = 15

        self.action_space = spaces.MultiDiscrete([8, self.num_actions])

        self.min_pods = MIN_REPLICATION
        self.max_pods = MAX_REPLICATION
        self.num_apps = 8

        self.deploymentList = get_robot_shop_deployment_list(self.k8s, self.min_pods, self.max_pods)

        for d in self.deploymentList:
            d.print_deployment()

        self.observation_space = self.get_observation_space()

        self.total_reward = 0
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
        self.file_results = "results.csv"
        self.obs_csv = self.name + "_observation.csv"

        # Dataset — simulation mode
        self.df = pd.read_csv("/Users/sevvalyavas/Desktop/RL denem/dataset/robot_shop_rl_dataset_final.csv")

    def step(self, action):
        if self.current_step == 1:
            if not self.k8s:
                self.simulation_update()
            self.time_start = time.time()

        if action[ID_DEPLOYMENTS] == 0:
            n = ID_cart
        elif action[ID_DEPLOYMENTS] == 1:
            n = ID_catalogue
        elif action[ID_DEPLOYMENTS] == 2:
            n = ID_dispatch
        elif action[ID_DEPLOYMENTS] == 3:
            n = ID_payment
        elif action[ID_DEPLOYMENTS] == 4:
            n = ID_ratings
        elif action[ID_DEPLOYMENTS] == 5:
            n = ID_shipping
        elif action[ID_DEPLOYMENTS] == 6:
            n = ID_user
        else:
            n = ID_web

        self.take_action(action[ID_MOVES], n)

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
        self.avg_latency.append(self.deploymentList[ID_web].latency)

        logging.info('[Step {}] | Action (Deployment): {} | Action (Move): {} | Reward: {} | Total Reward: {}'.format(
            self.current_step, DEPLOYMENTS[action[0]], MOVES[action[1]], reward, self.total_reward))

        ob = self.get_state()
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.save_obs_to_csv(self.obs_csv, np.array(ob), date, self.deploymentList[ID_web].latency)

        self.info = dict(total_reward=self.total_reward)

        self.constraint_max_pod_replicas = False
        self.constraint_min_pod_replicas = False

        if self.current_step == MAX_STEPS:
            self.episode_count += 1
            self.execution_time = time.time() - self.time_start
            save_to_csv(self.file_results, self.episode_count,
                        mean(self.avg_pods), mean(self.avg_latency),
                        self.total_reward, self.execution_time)

        # FIX 1: gymnasium step → 5 değer döndürür (terminated, truncated ayrı)
        return np.array(ob), reward, self.episode_over, False, self.info

    def reset(self, seed=None, options=None):
        # FIX 2: gymnasium reset → (obs, info) döndürür
        super().reset(seed=seed)

        self.current_step = 0
        self.episode_over = False
        self.total_reward = 0
        self.avg_pods = []
        self.avg_latency = []

        self.constraint_max_pod_replicas = False
        self.constraint_min_pod_replicas = False

        self.deploymentList = get_robot_shop_deployment_list(self.k8s, self.min_pods, self.max_pods)

        return np.array(self.get_state()), {}

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
                return -3000

        if self.constraint_min_pod_replicas:
            if self.goal_reward == COST:
                return -1
            elif self.goal_reward == LATENCY:
                return -3000

        return self.calculate_reward()

    def get_state(self):
        ob = (
            self.deploymentList[ID_cart].num_pods,
            self.deploymentList[ID_cart].desired_replicas,
            self.deploymentList[ID_cart].cpu_usage,
            self.deploymentList[ID_cart].mem_usage,
            self.deploymentList[ID_cart].received_traffic,
            self.deploymentList[ID_cart].transmit_traffic,
            self.deploymentList[ID_catalogue].num_pods,
            self.deploymentList[ID_catalogue].desired_replicas,
            self.deploymentList[ID_catalogue].cpu_usage,
            self.deploymentList[ID_catalogue].mem_usage,
            self.deploymentList[ID_catalogue].received_traffic,
            self.deploymentList[ID_catalogue].transmit_traffic,
            self.deploymentList[ID_dispatch].num_pods,
            self.deploymentList[ID_dispatch].desired_replicas,
            self.deploymentList[ID_dispatch].cpu_usage,
            self.deploymentList[ID_dispatch].mem_usage,
            self.deploymentList[ID_dispatch].received_traffic,
            self.deploymentList[ID_dispatch].transmit_traffic,
            self.deploymentList[ID_payment].num_pods,
            self.deploymentList[ID_payment].desired_replicas,
            self.deploymentList[ID_payment].cpu_usage,
            self.deploymentList[ID_payment].mem_usage,
            self.deploymentList[ID_payment].received_traffic,
            self.deploymentList[ID_payment].transmit_traffic,
            self.deploymentList[ID_ratings].num_pods,
            self.deploymentList[ID_ratings].desired_replicas,
            self.deploymentList[ID_ratings].cpu_usage,
            self.deploymentList[ID_ratings].mem_usage,
            self.deploymentList[ID_ratings].received_traffic,
            self.deploymentList[ID_ratings].transmit_traffic,
            self.deploymentList[ID_shipping].num_pods,
            self.deploymentList[ID_shipping].desired_replicas,
            self.deploymentList[ID_shipping].cpu_usage,
            self.deploymentList[ID_shipping].mem_usage,
            self.deploymentList[ID_shipping].received_traffic,
            self.deploymentList[ID_shipping].transmit_traffic,
            self.deploymentList[ID_user].num_pods,
            self.deploymentList[ID_user].desired_replicas,
            self.deploymentList[ID_user].cpu_usage,
            self.deploymentList[ID_user].mem_usage,
            self.deploymentList[ID_user].received_traffic,
            self.deploymentList[ID_user].transmit_traffic,
            self.deploymentList[ID_web].num_pods,
            self.deploymentList[ID_web].desired_replicas,
            self.deploymentList[ID_web].cpu_usage,
            self.deploymentList[ID_web].mem_usage,
            self.deploymentList[ID_web].received_traffic,
            self.deploymentList[ID_web].transmit_traffic,
        )
        return ob

    def get_observation_space(self):
        low_list = []
        high_list = []
        for _ in DEPLOYMENTS:
            low_list += [self.min_pods, self.min_pods, 0, 0, 0, 0]
            high_list += [self.max_pods, self.max_pods,
                          get_max_cpu(), get_max_mem(),
                          get_max_traffic(), get_max_traffic()]
        return spaces.Box(
            low=np.array(low_list),
            high=np.array(high_list),
            dtype=np.float32
        )

    def calculate_reward(self):
        if self.goal_reward == COST:
            return get_cost_reward(self.deploymentList)
        elif self.goal_reward == LATENCY:
            return get_latency_reward_robot_shop(ID_web, self.deploymentList)

    def simulation_update(self):
        def get_svc(name):
            return name + 'service' if name != 'web' else 'webservice'

        if self.current_step == 1:
            sample = self.df.sample()
            for i in range(len(DEPLOYMENTS)):
                svc = get_svc(DEPLOYMENTS[i])
                self.deploymentList[i].num_pods = int(sample[svc + '_num_pods'].values[0])
                self.deploymentList[i].num_previous_pods = int(sample[svc + '_num_pods'].values[0])
        else:
            pods = [d.num_pods for d in self.deploymentList]
            previous_pods = [d.num_previous_pods for d in self.deploymentList]

            data = self.df.copy()
            for i in range(len(DEPLOYMENTS)):
                svc = get_svc(DEPLOYMENTS[i])
                col = svc + '_num_pods'
                diff_col = 'diff-' + svc
                data[diff_col] = data[col].diff()
                diff_val = pods[i] - previous_pods[i]
                filtered = data.loc[(data[col] == pods[i]) & (data[diff_col] == diff_val)]
                if filtered.empty:
                    filtered = data.loc[data[col] == pods[i]]
                data = filtered if not filtered.empty else data

            sample = data.sample()

        for i in range(len(DEPLOYMENTS)):
            svc = get_svc(DEPLOYMENTS[i])
            self.deploymentList[i].cpu_usage = float(sample[svc + '_cpu_usage'].values[0])
            self.deploymentList[i].mem_usage = float(sample[svc + '_mem_usage'].values[0])
            self.deploymentList[i].received_traffic = float(sample[svc + '_traffic_in'].values[0])
            self.deploymentList[i].transmit_traffic = float(sample[svc + '_traffic_out'].values[0])
            self.deploymentList[i].latency = float(
                "{:.3f}".format(sample[svc + '_latency'].values[0]))

        for d in self.deploymentList:
            d.update_replicas()

    def save_obs_to_csv(self, obs_file, obs, date, latency):
        def get_svc(name):
            return name + 'service' if name != 'web' else 'webservice'

        fields = ['date']
        for d in self.deploymentList:
            svc = get_svc(d.name)
            fields += [svc + '_num_pods', svc + '_desired_replicas',
                    svc + '_cpu_usage', svc + '_mem_usage',
                    svc + '_traffic_in', svc + '_traffic_out', svc + '_latency']

        file_exists = os.path.isfile(obs_file) and os.path.getsize(obs_file) > 0
        file = open(obs_file, 'a+', newline='')
        with file:
            writer = csv.DictWriter(file, fieldnames=fields)
            if not file_exists:
                writer.writeheader()
            row = {'date': date}
            for i, d in enumerate(self.deploymentList):
                svc = get_svc(d.name)
                base = i * 6
                row[svc + '_num_pods'] = int(obs[base])
                row[svc + '_desired_replicas'] = int(obs[base + 1])
                row[svc + '_cpu_usage'] = float(obs[base + 2])
                row[svc + '_mem_usage'] = float(obs[base + 3])
                row[svc + '_traffic_in'] = float(obs[base + 4])
                row[svc + '_traffic_out'] = float(obs[base + 5])
                row[svc + '_latency'] = float("{:.3f}".format(latency))
            writer.writerow(row)