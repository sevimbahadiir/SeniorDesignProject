import sys
sys.path.insert(0, '/Users/sevvalyavas/Desktop/RL denem')
import logging
import argparse

#only A2C
from stable_baselines3 import A2C
from gym_hpa.envs import RobotShop
from stable_baselines3.common.callbacks import CheckpointCallback
from policies.util.util import test_model

logging.basicConfig(filename='run.log', filemode='w', level=logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

parser = argparse.ArgumentParser(description='Run gym-hpa for Robot Shop!')
parser.add_argument('--alg', default='a2c', help='a2c')
parser.add_argument('--k8s', default=False, action="store_true")
parser.add_argument('--goal', default='cost', help='["cost", "latency"]')
parser.add_argument('--training', default=False, action="store_true")
parser.add_argument('--testing', default=False, action="store_true")
parser.add_argument('--loading', default=False, action="store_true")
parser.add_argument('--load_path', default='logs/model/test.zip')
parser.add_argument('--test_path', default='logs/model/test.zip')
parser.add_argument('--steps', default=500)
parser.add_argument('--total_steps', default=5000)

args = parser.parse_args()


def get_model(env, tensorboard_log):
    return A2C("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log)


def get_load_model(tensorboard_log, load_path):
    return A2C.load(load_path, reset_num_timesteps=False, verbose=1,
                    tensorboard_log=tensorboard_log)


def main():
    logging.info(args)

    k8s = args.k8s
    goal = args.goal
    loading = args.loading
    load_path = args.load_path
    training = args.training
    testing = args.testing
    test_path = args.test_path
    steps = int(args.steps)
    total_steps = int(args.total_steps)

    env = RobotShop(k8s=k8s, goal_reward=goal)

    scenario = 'real' if k8s else 'simulated'
    tensorboard_log = "../../results/robot_shop/" + scenario + "/" + goal + "/"
    name = "a2c_env_" + env.name + "_goal_" + goal + "_k8s_" + str(k8s) + "_totalSteps_" + str(total_steps)

    checkpoint_callback = CheckpointCallback(
        save_freq=steps, save_path="logs/" + name, name_prefix=name)

    if training:
        if loading:
            model = get_load_model(tensorboard_log, load_path)
            model.set_env(env)
        else:
            model = get_model(env, tensorboard_log)
        model.learn(total_timesteps=total_steps,
                    tb_log_name=name + "_run",
                    callback=checkpoint_callback)
        model.save(name)

    if testing:
        model = get_load_model(tensorboard_log, test_path)
        test_model(model, env, n_episodes=100, n_steps=110,
                   smoothing_window=5, fig_name=name + "_test_reward.png")


if __name__ == "__main__":
    main()