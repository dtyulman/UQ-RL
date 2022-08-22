import torch
import gym

class ChainEnvironment(gym.Env):
    def __init__(self, length=10):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Discrete(length)
        self.length = length
        self.agent_state = 0

    def reset(self, return_info=False):
        info = {} #not used, trying to follow gym API
        self.agent_state = 0
        observation = self.get_observation(self.agent_state)
        return (observation, info) if return_info else observation

    def step(self, action):
        info = {} #not used, trying to follow gym API

        self.update_agent_state(action)
        observation = self.get_observation(self.agent_state)
        reward = self.get_reward(self.agent_state)
        done = self.agent_state==(self.length-1)
        return observation, reward, done, info

    def update_agent_state(self, action):
        """Drops agent to the start of chain if it goes left"""
        if action == 0:
            self.agent_state = 0
        else:
            self.agent_state = min(self.agent_state+1, self.length-1)

    def get_observation(self, state):
        obs = torch.zeros(self.length)
        obs[state] = 1
        return obs

    def get_reward(self, state):
        return float(state==(self.length-1))


class EasyChain(ChainEnvironment):
    def update_agent_state(self, action):
        action = int(action*2-1) #input is in {0,1}, convert to {+1, -1}
        #ignore actions that would take agent outside of [0,...,self.length]
        self.agent_state = min(max(0, self.agent_state+action), self.length-1)



if __name__ == '__main__':
    from gym.utils.env_checker import check_env
    env = ChainEnvironment()
    check_env(env)
