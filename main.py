import numpy as np
import matplotlib.pyplot as plt
import torch

from agents import ActorCritic, CuriousActorCritic
from environments import ChainEnvironment, EasyChain


def get_chain_policy(agent, env):
    return [agent(env.get_observation(s))[1][1].item() for s in range(env.observation_space.n)]


def train(agent, env, num_episodes, batch_size, rollout_len, early_end_episode=True):
    episode_total_reward = torch.zeros(num_episodes)
    episode_intrinsic_reward = torch.zeros_like(episode_total_reward)
    episode_intrinsic_reward_max = torch.zeros_like(episode_total_reward)
    episode_critic_loss = []
    episode_farthest_state = []
    for ep in range(num_episodes):
        policy_loss = torch.tensor(0.)
        critic_loss = torch.tensor(0.)
        intrinsic_critic_loss = torch.tensor(0.)
        distillation_loss = torch.tensor(0.)
        farthest_state = 0
        for b in range(batch_size):
            observation = env.reset()

            rewards = torch.zeros(rollout_len)
            intrinsic_rewards = torch.zeros_like(rewards)
            values = torch.zeros_like(rewards)
            intrinsic_values = torch.zeros_like(rewards)
            entropies = torch.zeros_like(rewards)
            log_p_actions = torch.zeros_like(rewards)
            for t in range(rollout_len):
                #select action
                action, policy, values[t] = agent(observation)
                log_policy = torch.log(policy)
                entropies[t] = -(log_policy*policy).sum()
                log_p_actions[t] = log_policy[action]

                #perform action and get next obs and reward
                observation, rewards[t], done, info = env.step(action)

                farthest_state = observation.nonzero()[0][0] if observation.nonzero()[0][0]>farthest_state else farthest_state
                intrinsic_rewards[t] = agent.intrinsic_reward(observation)
                intrinsic_values[t] = agent.intrinsic_critic(observation)

                if done and early_end_episode:
                    log_p_actions = log_p_actions[:t+1]
                    rewards = rewards[:t+1]
                    intrinsic_rewards = intrinsic_rewards[:t+1]
                    values = values[:t+1]
                    intrinsic_values = intrinsic_values[:t+1]
                    entropies = entropies[:t+1]
                    break

            #accumulate loss over batches
            episode_total_reward[ep] += rewards.sum() / batch_size
            episode_intrinsic_reward[ep] += intrinsic_rewards.sum() / batch_size
            episode_intrinsic_reward_max[ep] = max(episode_intrinsic_reward_max[ep], intrinsic_rewards.max())
            # policy_loss_b, critic_loss_b = agent.loss(log_p_actions, rewards, values, entropies)
            policy_loss_b, critic_loss_b, intrinsic_critic_loss_b, distillation_loss_b = \
                agent.loss(log_p_actions, rewards, values, intrinsic_rewards, intrinsic_values, entropies)
            policy_loss += policy_loss_b / batch_size
            critic_loss += critic_loss_b / batch_size
            intrinsic_critic_loss += intrinsic_critic_loss_b / batch_size
            distillation_loss += distillation_loss_b / batch_size
        episode_farthest_state.append(farthest_state)


        #compute gradient and optimize
        episode_critic_loss.append(critic_loss.item())
        agent.optimization_step(policy_loss, critic_loss, intrinsic_critic_loss, distillation_loss)
        if ep % 10 == 0:
            print(f'ep={ep}, reward={episode_total_reward[ep]:.2f}, '
                  f'intr_rew={episode_intrinsic_reward[ep]:.2f} (max={episode_intrinsic_reward_max[ep]:.2f}), '
                  f'critic_loss={critic_loss:.1e}, steps={t}, '
                  f'farthest={farthest_state}')

    return episode_total_reward, episode_critic_loss, episode_intrinsic_reward.detach()

#%%
discount_factor = 0.9
intrinsic_factor = 0
num_episodes = 300
batch_size = 50
early_end_episode = True
entropy_reg = 0

chain_len_list = [50] #np.arange(5,18,2)

episode_total_reward_list = []
episode_intrinsic_reward_list = []
episode_critic_loss_list = []
episode_policy_list = []
for chain_len in chain_len_list:
    #environment setup
    print('chain_len=', chain_len)
    torch.random.manual_seed(1)
    env = ChainEnvironment(length=chain_len)
    agent = CuriousActorCritic(intrinsic_factor = intrinsic_factor,
                        state_dim = chain_len,
                        num_actions = env.action_space.n,
                        actor_size = 100,
                        critic_size = 100,
                        gamma = discount_factor,
                        entropy_reg = entropy_reg)
    init_chain_policy = get_chain_policy(agent, env)

    rollout_len = chain_len*2
    episode_total_reward, episode_critic_loss, episode_intrinsic_reward = \
        train(agent, env, num_episodes, batch_size, rollout_len, early_end_episode=True)

    episode_total_reward_list.append(episode_total_reward)
    episode_intrinsic_reward_list.append(episode_intrinsic_reward)
    episode_critic_loss_list.append(episode_critic_loss)
    episode_policy_list.append(get_chain_policy(agent, env))


#%% Plot training progression and initial/final policy for one chain environment
# fig, ax = plt.subplots(2,1)
# ax[0].plot(episode_total_reward)
# ax[0].set_ylabel('Total reward')
# ax[1].plot(episode_critic_loss)
# ax[1].set_ylabel('Critic loss')
# ax[1].set_xlabel('Episodes')

# fig, ax = plt.subplots()
# ax.plot(init_chain_policy, label='Initial')
# final_chain_policy = get_chain_policy(agent, env)
# ax.plot(final_chain_policy, label='Trained')
# ax.set_title('Policy')
# ax.set_xlabel('State')
# ax.set_ylabel('P(right)')
# ax.legend()

#%%
cmap = plt.colormaps.get('cool')
fig1, ax1 = plt.subplots(2,1)
fig2, ax2 = plt.subplots()
for i, (etr, eir, cl, pol) in enumerate(zip(episode_total_reward_list, episode_intrinsic_reward_list,
                                            chain_len_list, episode_policy_list)):
    c = cmap(i/len(episode_total_reward_list))
    ax1[0].plot(etr, color=c, label=f'len={cl}')
    ax1[1].plot(eir.detach(), color=c)
    ax2.plot(pol, color=c)
ax1[0].set_title('Total reward')
ax1[0].set_ylabel('Extrinsic')
ax1[1].set_ylabel('Intrinsic')
ax1[1].set_xlabel('Episodes')
ax1[0].legend()
ax2.set_ylabel('P(right)')
ax2.set_xlabel('State')

#%%
thres = 1-1/batch_size+1e-6
eps_to_thres = []
for i, (etr, cl) in enumerate(zip(episode_total_reward_list, chain_len_list)):
    try:
        ep = (etr>=thres).nonzero()[0]
    except IndexError:
        ep = float('nan')
    eps_to_thres.append( ep )

fig, ax = plt.subplots()
ax.plot(chain_len_list, eps_to_thres)
ax.set_xlabel('Chain length')
ax.set_ylabel('Num episodes')
ax.set_title('Episodes until R=1')
