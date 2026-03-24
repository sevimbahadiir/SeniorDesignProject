import logging
import math
import random
import time
import requests
from kubernetes import client

# Constants
MAX_CPU = 10000  # cpu in m
MAX_MEM = 10000  # memory in MiB
MAX_TRAFFIC = 20000  # MAX Number of requests (in Kbit/s)

CPU_WEIGHT = 0.7
MEM_WEIGHT = 0.3

# port-forward in k8s cluster
PROMETHEUS_URL = 'http://localhost:9090/'

# Endpoint of your Kube cluster: kube proxy enabled
HOST = "http://localhost:8080"

# TODO: Add the TOKEN from your cluster!
TOKEN = ""


def get_robot_shop_deployment_list(k8s, min, max):
    deployment_list = [
        # 1
        DeploymentStatus(k8s, "cart", "robot-shop", "cart",
                         "robotshop/rs-cart:latest",
                         max, min, 100, 200, 64, 128),
        # 2
        DeploymentStatus(k8s, "catalogue", "robot-shop", "catalogue",
                         "robotshop/rs-catalogue:latest",
                         max, min, 100, 200, 64, 128),
        # 3
        DeploymentStatus(k8s, "dispatch", "robot-shop", "dispatch",
                         "robotshop/rs-dispatch:latest",
                         max, min, 100, 200, 64, 128),
        # 4
        DeploymentStatus(k8s, "payment", "robot-shop", "payment",
                         "robotshop/rs-payment:latest",
                         max, min, 100, 200, 64, 128),
        # 5
        DeploymentStatus(k8s, "ratings", "robot-shop", "ratings",
                         "robotshop/rs-ratings:latest",
                         max, min, 100, 200, 64, 128),
        # 6
        DeploymentStatus(k8s, "shipping", "robot-shop", "shipping",
                         "robotshop/rs-shipping:latest",
                         max, min, 100, 200, 64, 128),
        # 7
        DeploymentStatus(k8s, "user", "robot-shop", "user",
                         "robotshop/rs-user:latest",
                         max, min, 100, 200, 64, 128),
        # 8
        DeploymentStatus(k8s, "web", "robot-shop", "web",
                         "robotshop/rs-web:latest",
                         max, min, 100, 200, 64, 128),
    ]
    return deployment_list


def get_max_cpu():
    return MAX_CPU


def get_max_mem():
    return MAX_MEM


def get_max_traffic():
    return MAX_TRAFFIC


def convert_to_milli_cpu(value):
    new_value = int(value[:-1])
    if value[-1] == "n":
        new_value = int(value[:-1])
        new_value = int(new_value / 1000000)
    return new_value


def change_usage(min, max, max_threshold):
    if max > max_threshold:
        max = max_threshold
    if min < 0:
        min = 0
    return random.randint(min, max)


def convert_to_mega_memory(value):
    last_two = value[-2:]
    new_value = 0
    if last_two == "Ki":
        size = len(value)
        new_value = int(value[:size - 2])
        new_value = int(new_value / 1000)
    return new_value


class DeploymentStatus:  # Deployment Status (Workload)
    def __init__(self, k8s, name, namespace, container_name, container_image, max_pods, min_pods,
                 cpu_request, cpu_limit, mem_request, mem_limit, threshold=0.75):
        self.name = name
        self.namespace = namespace
        self.container_name = container_name
        self.container_image = container_image

        self.threshold = threshold
        self.cpu_weight = CPU_WEIGHT
        self.mem_weight = MEM_WEIGHT

        self.pod_names = ["pod-1"]
        self.max_pods = max_pods
        self.min_pods = min_pods
        self.num_pods = 1
        self.num_previous_pods = 1
        self.desired_replicas = 1

        self.cpu_request = cpu_request
        self.cpu_limit = cpu_limit
        self.mem_request = mem_request
        self.mem_limit = mem_limit

        self.cpu_target = int(self.threshold * self.cpu_request)
        self.mem_target = int(self.threshold * self.mem_request)

        self.MAX_CPU = MAX_CPU
        self.MAX_MEM = MAX_MEM
        self.MAX_TRAFFIC = MAX_TRAFFIC

        self.cpu_usage = random.randint(1, get_max_cpu())
        self.mem_usage = random.randint(1, get_max_mem())
        self.received_traffic = random.randint(1, get_max_traffic())
        self.transmit_traffic = random.randint(1, get_max_traffic())

        self.k8s = k8s
        self.csv = self.namespace + '_' + self.name + '.csv'
        self.sleep = 0.2
        self.latency = 0

        if self.k8s:
            logging.info("[Deployment] Consider a real k8s cluster ... ")

            self.token = TOKEN
            self.config = client.Configuration()
            self.config.verify_ssl = False
            self.config.api_key = {"authorization": "Bearer " + self.token}
            self.config.host = HOST
            self.client = client.ApiClient(self.config)
            self.v1 = client.CoreV1Api(self.client)
            self.apps_v1 = client.AppsV1Api(self.client)
            self.deployment_object = self.apps_v1.read_namespaced_deployment(
                name=self.name, namespace=self.namespace)
            self.num_pods = self.deployment_object.spec.replicas
            self.num_previous_pods = self.deployment_object.spec.replicas
            self.update_obs_k8s()

    def update_obs_k8s(self):
        self.pod_names = []
        pods = self.v1.list_namespaced_pod(namespace=self.namespace)
        for p in pods.items:
            if p.metadata.labels.get('app') == self.name:
                self.pod_names.append(p.metadata.name)

        self.cpu_usage = 0
        self.mem_usage = 0
        self.received_traffic = 0
        self.transmit_traffic = 0

        self.num_previous_pods = self.deployment_object.spec.replicas
        self.deployment_object = self.apps_v1.read_namespaced_deployment(
            name=self.name, namespace=self.namespace)
        self.num_pods = self.deployment_object.spec.replicas

        for p in self.pod_names:
            query_cpu = 'sum(irate(container_cpu_usage_seconds_total{namespace=' \
                        '"' + self.namespace + '", pod="' + p + '"}[5m])) by (pod)'
            query_mem = 'sum(irate(container_memory_working_set_bytes{namespace=' \
                        '"' + self.namespace + '", pod="' + p + '"}[5m])) by (pod)'
            query_received = 'sum(irate(container_network_receive_bytes_total{namespace=' \
                             '"' + self.namespace + '", pod="' + p + '"}[5m])) by (pod)'
            query_transmit = 'sum(irate(container_network_transmit_bytes_total{namespace="' \
                             + self.namespace + '", pod="' + p + '"}[5m])) by (pod)'

            results_cpu = self.fetch_prom(query_cpu)
            if results_cpu:
                cpu = int(float(results_cpu[0]['value'][1]) * 1000)
                self.cpu_usage += cpu

            results_mem = self.fetch_prom(query_mem)
            if results_mem:
                mem = int(float(results_mem[0]['value'][1]) / 1000000)
                self.mem_usage += mem

            results_received = self.fetch_prom(query_received)
            if results_received:
                rec = int(float(results_received[0]['value'][1]))
                rec = int(rec / 1000)
                self.received_traffic += rec

            results_transmit = self.fetch_prom(query_transmit)
            if results_transmit:
                trans = int(float(results_transmit[0]['value'][1]))
                trans = int(trans / 1000)
                self.transmit_traffic += trans

            # Robot Shop latency: web service response time via Prometheus
            # Uses locust avg response time for GET / (web frontend)
            if self.name == 'web':
                query_latency = 'locust_requests_avg_response_time{method="GET", name="/"}'
                get_web = 0
                results_latency = self.fetch_prom(query_latency)
                if results_latency:
                    dur = float(results_latency[0]['value'][1])
                    get_web = float("{:.3f}".format(dur))
                self.latency = float("{:.3f}".format(get_web))

        self.update_replicas()
        return

    def update_replicas(self):
        cpu_target_usage = self.num_pods * self.cpu_target
        mem_target_usage = self.num_pods * self.mem_target

        desired_replicas_cpu = math.ceil(self.num_pods * (self.cpu_usage / cpu_target_usage))
        desired_replicas_mem = math.ceil(self.num_pods * (self.mem_usage / mem_target_usage))

        self.desired_replicas = math.ceil(
            (self.cpu_weight * desired_replicas_cpu) + (self.mem_weight * desired_replicas_mem))

        if self.desired_replicas == 0:
            self.desired_replicas = 1
        if self.desired_replicas > self.max_pods:
            self.desired_replicas = self.max_pods
        return

    def fetch_prom(self, query):
        try:
            response = requests.get(PROMETHEUS_URL + '/api/v1/query',
                                    params={'query': query})
        except requests.exceptions.RequestException as e:
            print(e)
            print("Retrying in {}...".format(self.sleep))
            time.sleep(self.sleep)
            return self.fetch_prom(query)

        if response.json()['status'] != "success":
            print("Error processing the request: " + response.json()['status'])
            print("Retrying in {}s...".format(self.sleep))
            time.sleep(self.sleep)
            return self.fetch_prom(query)

        result = response.json()['data']['result']
        return result

    def print_deployment(self):
        logging.info("[Deployment] Name: " + str(self.name))
        logging.info("[Deployment] Namespace: " + str(self.namespace))
        logging.info("[Deployment] Number of pods: " + str(self.num_pods))
        logging.info("[Deployment] Desired Replicas: " + str(self.desired_replicas))
        logging.info("[Deployment] Pod Names: " + str(self.pod_names))
        logging.info("[Deployment] MAX Pods: " + str(self.max_pods))
        logging.info("[Deployment] MIN Pods: " + str(self.min_pods))
        logging.info("[Deployment] CPU Usage (in m): " + str(self.cpu_usage))
        logging.info("[Deployment] MEM Usage (in Mi): " + str(self.mem_usage))
        logging.info("[Deployment] Received traffic (in Kbit/s): " + str(self.received_traffic))
        logging.info("[Deployment] Transmit traffic (in Kbit/s): " + str(self.transmit_traffic))
        logging.info("[Deployment] latency: " + str(self.latency))

    def update_deployment(self, new_replicas):
        self.deployment_object = self.apps_v1.read_namespaced_deployment(
            name=self.name, namespace=self.namespace)
        self.num_previous_pods = self.deployment_object.spec.replicas
        self.deployment_object.spec.replicas = new_replicas
        self.patch_deployment(new_replicas)

    def patch_deployment(self, new_replicas):
        try:
            self.apps_v1.patch_namespaced_deployment(
                name=self.name, namespace=self.namespace, body=self.deployment_object)
        except Exception as e:
            print(e)
            print("Retrying in {}s...".format(self.sleep))
            time.sleep(self.sleep)
            return self.update_deployment(new_replicas)

    def deploy_pod_replicas(self, n, env):
        replicas = self.num_pods + n
        if replicas <= self.max_pods:
            if self.k8s:
                self.update_deployment(replicas)
            else:
                self.num_previous_pods = self.num_pods
                self.num_pods = replicas
            return
        else:
            env.constraint_max_pod_replicas = True

    def terminate_pod_replicas(self, n, env):
        replicas = self.num_pods - n
        if replicas >= self.min_pods:
            if self.k8s:
                self.update_deployment(replicas)
            else:
                self.num_previous_pods = self.num_pods
                self.num_pods = replicas
            return
        else:
            env.constraint_min_pod_replicas = True
