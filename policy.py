import numpy as np
import arm


class Policy:
    def __init__(self, policy_name, k_arms, d, alpha):
        self.k_arms = k_arms
        if policy_name == 'LinUCB':
            self.policy = arm.LinUCBArm
        elif policy_name == 'TS':
            self.policy = arm.TSArm
        elif policy_name == 'Greedy':
            self.policy = arm.GreedyArm
        self.arm = [self.policy(arm_index=i, d=d, alpha=alpha) for i in range(k_arms)]

    def select_arm(self, x):
        arm_best = -1000
        arm_candidate = []
        for arm_index in range(self.k_arms):
            arm_cal = self.arm[arm_index].calculate(x)
            if arm_cal > arm_best:
                arm_best = arm_cal
                arm_candidate = [arm_index]
            if arm_cal == arm_best:
                arm_candidate.append(arm_index)
        return np.random.choice(arm_candidate)
