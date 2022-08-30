import numpy as np
import pandas as pd
import seaborn
from matplotlib import pyplot as plt

from bandit import Bandit

# may add y classification function
# better to design manually can be seen as a part reward function design

def reward1(arm, y):
    r = -1
    if arm == y:
        r = 1
    # return reward, regret
    return r, 1 - r


def reward2(arm, y):
    # eat
    r = 0
    if arm == 0:
        if y == 0:
            r = 5
        else:
            r = np.random.choice([-500, 5], p=[0.5, 0.5])

    return r, 5 - r


if __name__ == '__main__':
    data_df = pd.read_csv('data/mushroom.csv')
    epoch = 100
    k_arms = 2
    alpha = 1

    sum_plot = pd.DataFrame()

    for name in ['TS', 'LinUCB', 'Greedy']:
    # for name in ['LinUCB']:
        bandit = Bandit(data_df, policy_name=name, epoch=epoch, alpha=alpha, k_arms=k_arms,
                        reward_function=reward2, target_name='edible')
        plot_data = bandit.train()
        sum_plot = pd.concat([sum_plot, plot_data], ignore_index=True)
    seaborn.lineplot(data=sum_plot, x='datapoint', y='regret', hue='name')
    plt.show()
