import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
            )

    def forward(self, input):
        return self.ff(input)


class ActorCritic():
    #see also: https://openai.com/blog/baselines-acktr-a2c/
    def __init__(self, state_dim, num_actions, actor_size=100, critic_size=100,
                 gamma=1, entropy_reg=0):
        self.actor = nn.Sequential(
            SimpleNet(state_dim, actor_size, num_actions),
            nn.Softmax(dim=0)
            )
        self.policy_optimizer = torch.optim.Adam(self.actor.parameters(), maximize=True)

        if critic_size is not None:
            self.critic = SimpleNet(state_dim, critic_size, 1)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
        else:
            self.critic = None

        self.gamma = float(gamma)
        self.entropy_reg = float(entropy_reg)


    def __call__(self, state):
        policy = self.actor(state)
        value = self.critic(state) if self.critic is not None else None
        action = Categorical(policy).sample()
        return action, policy, value


    def optimization_step(self, policy_loss, critic_loss=None):
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        if critic_loss is not None:
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()


    def compute_return(self, rewards):
        discounted_return = torch.zeros_like(rewards)
        discounted_return[-1] = rewards[-1]
        for i in range(len(rewards)-1, 0, -1):
            discounted_return[i-1] = self.gamma*discounted_return[i] + rewards[i-1]
        return discounted_return


    def loss(self, log_p_actions, rewards, values, entropies=None):
        #accumulate loss over batches
        discounted_return = self.compute_return(rewards)
        critic_loss = F.mse_loss(discounted_return, values)

        if self.critic is not None:
            discounted_return = discounted_return - values.detach()
        policy_loss = (log_p_actions*discounted_return).sum()

        if self.entropy_reg > 0:
            # https://paperswithcode.com/method/entropy-regularization
            policy_loss += self.entropy_reg*entropies.sum()

        return policy_loss, critic_loss


class CuriousActorCritic(ActorCritic):
    def __init__(self, intrinsic_factor, *args, **kwargs):
        super().__init__(*args, **kwargs)

        state_dim = kwargs['state_dim']
        self.target_network = SimpleNet(state_dim, 100, state_dim)
        self.target_network.requires_grad_(False)

        self.predictor_network = SimpleNet(state_dim, 100, state_dim)
        self.predictor_optimizer = torch.optim.Adam(self.predictor_network.parameters(), lr=1e-4)

        self.intrinsic_critic = SimpleNet(state_dim, 100, 1)
        self.intrinsic_critic_optimizer = torch.optim.Adam(self.intrinsic_critic.parameters())
        self.intrinsic_factor = intrinsic_factor

    def intrinsic_reward(self, observation):
        target = self.target_network(observation)
        prediction = self.predictor_network(observation)
        mse = F.mse_loss(target, prediction)
        return mse


    def loss(self, log_p_actions,
             rewards, values,
             intrinsic_rewards, intrinsic_values,
             entropies=None):

        discounted_return = self.compute_return(rewards)
        # intrinsic_rewards = intrinsic_rewards/intrinsic_rewards.var()
        discounted_intrinsic_return = self.compute_return(intrinsic_rewards.detach())

        intrinsic_critic_loss = F.mse_loss(discounted_intrinsic_return, intrinsic_values)
        critic_loss = F.mse_loss(discounted_return, values)

        advantage = discounted_return - values.detach()
        intrinsic_advantage = discounted_intrinsic_return# - intrinsic_values.detach()
        advantage = advantage + self.intrinsic_factor*intrinsic_advantage
        policy_loss = (log_p_actions*advantage).sum()

        distillation_loss = intrinsic_rewards.mean()

        if self.entropy_reg > 0:
            # https://paperswithcode.com/method/entropy-regularization
            policy_loss += self.entropy_reg*entropies.sum()

        return policy_loss, critic_loss, intrinsic_critic_loss, distillation_loss


    def optimization_step(self, policy_loss, critic_loss, intrinsic_critic_loss, distillation_loss):
        for loss, opt in zip((policy_loss, critic_loss, intrinsic_critic_loss, distillation_loss),
                             (self.policy_optimizer, self.critic_optimizer, self.intrinsic_critic_optimizer, self.predictor_optimizer)):
            opt.zero_grad()
            loss.backward()
            opt.step()
