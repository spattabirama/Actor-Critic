import matplotlib.pyplot as plt
import gym
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim


class ActorNN(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(observation_space, 128),
            nn.ReLU(),
            nn.Linear(128, action_space),
            nn.Softmax(dim=1)
        )

    def forward(self, l):
        return self.layers(l)


class CriticNN(nn.Module):
    def __init__(self, observation_space):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(observation_space, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, l):
        return self.layers(l)


def return_action(network, state):
    tensor_state = torch.FloatTensor(state).unsqueeze(0)
    action_probability = network(tensor_state)
    action = torch.multinomial(action_probability, num_samples=1)
    return action.item(), action_probability[0, action.item()], torch.log(action_probability[0, action.item()])


def plot(rewards):
    plt.plot(rewards)
    plt.ylabel('Return')
    plt.xlabel('Episodes')
    plt.title('CartPole Actor-Critic')
    plt.show()


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    actor_NN = ActorNN(env.observation_space.shape[0], env.action_space.n)
    actor_optim = optim.Adam(actor_NN.parameters(), lr=0.005)

    critic_NN = CriticNN(env.observation_space.shape[0])
    critic_optim = optim.Adam(critic_NN.parameters(), lr=0.01)

    discount = 0.99
    episodes = 500
    rewards = []

    for episode in range(episodes):
        i = 0
        state, _ = env.reset()
        I = 1
        done = False
        reward = 0

        while True:
            i += 1
            action, action_prob, log_prob = return_action(actor_NN, state)
            next_state, r, term, trunc, _ = env.step(action)
            done = term or trunc
            reward += r
            state_value = critic_NN(torch.FloatTensor(state).unsqueeze(0))
            next_state_value = critic_NN(torch.FloatTensor(next_state).unsqueeze(0))

            if done:
                next_state_value = torch.tensor([0.0]).unsqueeze(0)

            delta = r + discount * next_state_value - state_value
            value_loss = func.smooth_l1_loss(r + discount * next_state_value, state_value) * I
            policy_loss = -log_prob * delta * I

            actor_optim.zero_grad()
            policy_loss.backward(retain_graph=True)
            actor_optim.step()

            critic_optim.zero_grad()
            value_loss.backward()
            critic_optim.step()

            if done:
                break

            state = next_state
            I *= discount

        if reward > 475:
            for g in actor_optim.param_groups:
                g['lr'] *= 0.9
            for g in critic_optim.param_groups:
                g['lr'] *= 0.9

        rewards.append(reward)

    plot(rewards)
