import pandas as pd
import sklearn
from tqdm import tqdm

import policy


class Bandit(object):
    def __init__(self, data, policy_name, epoch, alpha, k_arms, reward_function, target_name):
        super().__init__()
        self._data = data
        self._policy_name = policy_name
        self._epoch = epoch
        self._alpha = alpha
        self._n_arms = k_arms
        self._reward_func = reward_function
        self._target_name = target_name

    def train(self):
        X_raw = self._data.copy().drop(self._target_name, axis=1)
        X = pd.get_dummies(X_raw, drop_first=True).to_numpy()
        # y = self._data[self._target_name]
        y = self._data['edible'].map({'p': 1, 'e': 0})

        y = y.to_numpy()
        d = X.shape[1]
        df1 = pd.DataFrame()
        # train epoch times
        for _ in tqdm(range(self._epoch)):
            X, y = sklearn.utils.shuffle(X, y)
            policy_object = policy.Policy(policy_name=self._policy_name, k_arms=self._n_arms, d=d, alpha=self._alpha)

            regret_cum = 0
            regret_list = []

            for counter, data_x_array in enumerate(X):
                arm_index = policy_object.select_arm(data_x_array)
                reward, regret = self._reward_func(arm_index, y[counter])
                regret_cum += regret
                regret_list.append(regret_cum)

                policy_object.arm[arm_index].reward_update(reward, data_x_array)

            df = pd.DataFrame(data=regret_list, columns=['regret'])
            df['datapoint'] = df.index
            df1 = pd.concat([df1, df], ignore_index=True)
        df1['name'] = self._policy_name

        return df1

# def run(data, policy_name, epoch, alpha, k_arms, reward_function):
#     X_raw = data.copy().drop(self._target_name, axis=1)
#     X = pd.get_dummies(X_raw, drop_first=True).to_numpy()
#     y = data['editable'].map({'p': 1, 'e': 0})
#     y = y.to_numpy()
#     d = X.shape[1]
#
#     df1 = pd.DataFrame()
#     for _ in tqdm(range(epoch)):
#         X, y = sklearn.utils.shuffle(X, y)
#         policy_object = policy.Policy(policy_name=policy_name, k_arms=k_arms, d=d, alpha=alpha)
#
#         regret_cum = 0
#         df = pd.DataFrame()
#
#         for counter, data_x_array in enumerate(X):
#             arm_index = policy_object.select_arm(data_x_array)
#             reward, regret = reward_function(arm_index, y[counter])
#             regret_cum += regret
#             policy_object.arm[arm_index].reward_update(reward, data_x_array)
#             data = pd.DataFrame()
#             data['regret'] = [regret_cum]
#             data['datapoint'] = counter
#             data['name'] = policy_name
#             df = pd.concat([df, data], ignore_index=True)
#         df1 = pd.concat([df1, df], ignore_index=True)
#     return df1
