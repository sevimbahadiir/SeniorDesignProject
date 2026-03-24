import pandas as pd
from matplotlib import pyplot as plt


def test_model(model, env, n_episodes, n_steps, smoothing_window, fig_name):
    episode_rewards = []
    reward_sum = 0
    # FIX 3: gymnasium reset → (obs, info)
    obs, _ = env.reset()

    print("------------ Testing -----------------")

    for e in range(n_episodes):
        for _ in range(n_steps):
            action, _ = model.predict(obs)
            # FIX 4: gymnasium step → 5 değer
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            reward_sum += reward
            if done:
                episode_rewards.append(reward_sum)
                print("Episode {} | Total reward: {} |".format(e, str(reward_sum)))
                reward_sum = 0
                obs, _ = env.reset()
                break

    env.close()
    del model, env

    plt.figure()
    rewards_smoothed = pd.Series(episode_rewards).rolling(
        smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig(fig_name, dpi=250, bbox_inches='tight')