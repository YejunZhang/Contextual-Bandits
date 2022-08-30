# Create class object for a single linear ucb disjoint arm
import numpy as np


class LinUCBArm:

    def __init__(self, arm_index, d, alpha):
        self._arm_index = arm_index
        self._alpha = alpha
        self._A = np.identity(d)
        self._b = np.zeros([d, 1])

    def calculate(self, x_array):
        A_inv = np.linalg.inv(self._A)
        theta = np.dot(A_inv, self._b)
        x = x_array.reshape([-1, 1])
        p = np.dot(theta.T, x) + self._alpha * np.sqrt(np.dot(x.T, np.dot(A_inv, x)))
        return p

    def reward_update(self, reward, x_array):
        x = x_array.reshape([-1, 1])
        self._A += np.dot(x, x.T)
        self._b += reward * x


class TSArm:
    def __init__(self, arm_index, d, alpha):
        self._arm_index = arm_index
        self._alpha = 0
        self._A = np.identity(d)
        self._b = np.zeros([d, 1])

    def calculate(self, x_array):
        A_inv = np.linalg.inv(self._A)
        theta = np.dot(A_inv, self._b)
        x = x_array.reshape([-1, 1])
        sample_theta = np.random.default_rng().multivariate_normal(theta.flatten(), A_inv, method='cholesky')
        p = np.dot(sample_theta.T, x)
        return p

    def reward_update(self, reward, x_array):
        x = x_array.reshape([-1, 1])
        self._A += np.dot(x, x.T)
        self._b += reward * x


class GreedyArm:

    def __init__(self, arm_index, d, alpha):
        self._arm_index = arm_index
        self._alpha = 0
        self._A = np.identity(d)
        self._b = np.zeros([d, 1])

        # history of x
        # self.x_his =

    def calculate(self, x_array):
        A_inv = np.linalg.inv(self._A)
        theta = np.dot(A_inv, self._b)
        x = x_array.reshape([-1, 1])
        p = np.dot(theta.T, x) + self._alpha * np.sqrt(np.dot(x.T, np.dot(A_inv, x)))
        return p

    def reward_update(self, reward, x_array):
        x = x_array.reshape([-1, 1])
        self._A += np.dot(x, x.T)
        self._b += reward * x


class OLS:
    def __init__(self, arm_index, d, q, h):
        self.arm_index = arm_index
        self.q = q
        self.h = h
        self.d = d
        self.A = np.identity(d)
        self.b = np.zeros([d, 1])
        self.set = []

    def calculate(self, x_array):
        # set the force-sample sets
        A_inv = np.linalg.inv(self.A)
        theta = np.dot(A_inv, self.b)
        x = x_array.reshape([-1, 1])
        p = np.dot(theta.T, x)
        return p

    def reward_update(self, x_array):
        pass
