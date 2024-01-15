## Introduction

Reinforcement learning is an area of machine learning, inspired by behavioral psychology, concerned with how an agent can learn from interactions with an environment. 
RL has a wide range of applications, from robotics, where it enables autonomous movement and decision-making, to gaming, where it has been used to achieve better performance. 
Other applications include personalized recommendations, autonomous vehicles, etc. 
Different algorithms can vary significantly in terms of complexity, efficiency, and suitability for a particular type of problem. 
Evaluating these algorithms in different environments and scenarios is vital to understand their strengths, weaknesses, and applicability.

The One-Step Actor-Critic algorithm is a combination of policy and value based methods. 
It has an ’actor’ that suggests actions and a ’critic’ that evaluates them, thereby allowing the algorithm to adjust its policy based on the feedback from the critic. 

## Actor-Critic (A2C)

### Introduction
The Actor-Critic method is an algorithm that combines elements of both policy-based and value based approaches. 
The Actor(policy learner) is responsible for determining the agent’s actions based on the current state (It is represented by a neural network in our implementation). 
Its parameters are optimized to maximize the expected cumulative reward over time. Complementing the Actor, the Critic is the value function estimator. 
The Critic evaluates the chosen actions by estimating the expected cumulative reward of being in a certain state and following the current policy. 
This assessment will be used by the Actor to update the policy accordingly.

The One-step Actor-Critic algorithm is inherently fully online and TD(0) algorithm will be used by Critic to update the parameters of the value function. 
The term ”One-Step” indicates that the algorithm prioritizes evaluating actions using a single time step of experience. 
In contrast to approaches considering multi-step returns, the One-Step Actor-Critic relies on the immediate reward and the value of the next state to create an estimate of the actual return. 
The TD(0) algorithm updates the value function by utilizing the temporal difference between the target and the approximated value. 
This update is performed in the direction of the gradient, and the target is estimated through a bootstrapping approach.

### Pseudocode

<img width="792" alt="Screenshot 2024-01-15 at 5 49 14 PM" src="https://github.com/spattabirama/Actor-Critic/assets/124756255/f0b3653a-aa1d-4f57-9c0c-000525704b65">
moving the cart left or right.

## Environments

I will be implementing this algorithm on the environments briefly explained below.

### The Cartpole Environment

This well-known testbed is more dynamic and continuous when compared with Gridworld, presenting different Challenges:
* Dynamics: The task involves balancing a pole on a moving cart. The agent controls the cart’s movement (left or right) to keep the pole upright.
* State Space: The state is represented by continuous variables such as the angle of the pole, the cart’s position, and their velocities.
* Reward Structure: The agent receives a reward for every timestep that the pole remains upright. The episode ends if the pole falls over or the cart moves out of the defined boundaries.
* Action Space: Discrete actions for moving the cart left or right.
* Episode Criteria: Each episode runs until the pole falls or after a certain number of steps.

### Acrobot

The Acrobot environment, which represents a challenging domain with a continuous state space and a discrete action set, is a benchmark problem in reinforcement learning. It consists of a twolink, two-joint robot, where the objective is to swing the end of the lower link up to a given height by applying torque on the joint between the two links.
* Dynamic: A two-link pendulum with only the second joint is actuated as the Acrobot system. Initially, both links point downwards. The goal is to swing the end of the lower link up to a specified height by applying torques at the joint.
* State Space: The state of the system is defined by the two rotational joint angles and their velocities. These states are continuous, making the problem more complex.
* Actions: Agent could apply only three torques which are, positive torque, no torque, or negative torque.
* Reward Structure: The reward is structured to be negative until the agent reaches the gal, encouraging the agent to reach the goal as quickly as possible. Typically, a reward signal is provided at each step. The task is often considered solved if the lower link swings to a certain height within a specified number of time steps.
* Episode Termination: Once the goal is achieved the episode ends or it ends after a certain number of steps.
* Goal Specification: The target is to swing the end of the lower link to a height at or above the top of the two links.
* Episode Criteria: Each episode runs until the goal is achieved or a maximum number of steps is reached, to prevent endlessly long episodes.
The agent’s learning progress through metrics such as the length of each episode, total rewards accumulated, and the number of steps taken to reach the goal was monitored in the environments. This allows us to evaluate the effectiveness of the One-Step Actor-Critic and Episodic Semi-gradient n-step SARSA algorithms in distinct and challenging scenarios, providing insights into their adaptability and performance in different RL contexts.

## Actor Critic on CartPole Env

### Hyperparameter Tuning:
* Learning Rates: Learning rates play a crucial role in the convergence and stability of training. A learning rate of 0.005 was chosen for the actor and 0.01 for the critic after rigourous trial and error. Critic networks generally should be higher than actor network according to resources so I scaled it by 20\%. Higher learning rates ( >0.01 for actor and >0.05 ) for both actor and critic caused disastrous oscillations and divergence. It prematurely converged to a local optima. The lower values ( < 0.001 for actor and < 0.01 for critic ) took forever to converge. In some cases, it didn't reach convergence even after 500 episodes. A learning rate of 0.005 for the actor and 0.01 for the critic was ideal. 
* Discount Factor (Gamma): Gamma is used to calculate the expected return contributes towards the importance given to future rewards. It is a hyper-parameter that controls how much the agent values future rewards over immediate ones. In case of cart pole, the goal is to keep the pole upright for as long as possible, long term rewards play a very important role. So gamma of 0.99 was essential. Lower values of gamma gave very poor results (low rewards) as it didn't give enough importance to long term rewards.
* Reward Threshold for Learning Rate Decay: A learning rate decay strategy based on the achieved reward was implemented. If the reward surpasses 475, the learning rates for both actor and critic optimizers are reduced by 10\%. This strategy aims to fine-tune learning rates during training. If the agent consistently achieves high rewards, reducing the learning rates might help stabilize learning and prevent overshooting optimal policies. Constant learning rate caused fluctuations even after reaching the reward state 500 so inorder to exploit the learnt model, the learning rate had to be subsequently reduced.
* Hidden Layers and Neurons: A relatively simple architecture of 1 layer with 128 neurons were chosen since the environment is relatively straight-forward.  

### Results:

![AcroBot A2C](https://github.com/spattabirama/Actor-Critic/assets/124756255/57dcba4b-ba7b-40c3-8e19-876e2681096b)

Above image shows that in the early episodes, the agent tries to grasp the dynamics of the CartPole environment, resulting in low and highly variable rewards. This is indicative of the initial exploration phase, where the agent is actively probing different actions and policies. As the algorithm adapts and refines its policy, a notable transition occurs. The rewards steadily converge to the maximum value of 500, signaling a more stable and effective policy. Once the agent learns the best policy, it successfully keeps the cart from falling for the entire episode, getting a maximum reward of 500 as a result.

![AcroBot A2C Avg](https://github.com/spattabirama/Actor-Critic/assets/124756255/d09f5441-51d0-4ffc-91d5-f4528341fb0b)

Averaging the results over 10 trials reveals a consistent trend. The agent tends to converge to the maximum reward of 500 after approximately 150 episodes. This convergence indicates the robustness of the learned policy across multiple runs, reinforcing the reliability of the algorithm. The Critic network’s value function estimation, coupled with the temporal-difference learning approach, allows the agent to understand the consequences of its actions and make informed decisions. The Actor network’s policy improvement is evident through the softmax activation, which refines the probability distribution over actions. The negative log probability gradient is used to guide the policy towards actions that yield higher expected returns.

## Actor Critic on AcroBot Env

### Hyperparameter Tuning:
* Learning Rates: Elevating the learning rates beyond 0.01 for the actor and 0.002 for the critic led to premature convergence. Conversely, adopting lower values, such as below 0.0005 for the actor and below 0.001 for the critic, resulted in excessively prolonged convergence times. The optimal balance was found at a learning rate of 0.002 for the actor and 0.01 for the critic, striking a suitable equilibrium between convergence speed and stability.
* Discount Factor (Gamma): The discount factor is set to 0.99, indicating that the agent values future rewards highly. This is a common choice for tasks where the consequences of actions have a long-term impact, as in Acrobot. In Acrobot only the terminal state has reward of 0 whereas all other actions cause negative reward. So in order to reduce the number of negative rewards accumulated, it needs to look forward to the goal state. Lower values of gamma took more number of steps for the Acrobot to swing above the given height.
* Reward Threshold for Learning Rate Decay: The learning rates are adjusted based on the received reward within specific step ranges. A substantial learning rate decay for zero rewards within the first 150 steps is taken as a proactive approach to rectifying a potentially poor initial policy. The more moderate decay for non-zero rewards between steps 150 and 200 is taken to enhance learning in cases of partial success. 
* Hidden Layers and Neurons: The choice of a neural network architecture with two fully connected layers of 256 neurons each for both the Actor and Critic networks is required for Acrobot, as it is a complex task that requires capturing intricate state-action relationships, and a larger hidden layer size allows the networks to learn and represent more sophisticated features from the environment.

### Results:

![CartPole A2C](https://github.com/spattabirama/Actor-Critic/assets/124756255/80f96981-4673-4d9e-92be-df4d27917858)

The graph clearly indicates how quickly the agent is able to learn to avoid a lot of negative rewards and adapts to the challenging Acrobot task. On an average in 50-120 steps, the acrobot is able to swing above the given height in initial 50 episodes. This quick transition from the initial challenging rewards to a more manageable range, showcasing its efficiency in policy exploration.It continues to maintain that level of consistency across the rest of the 500 episodes indicating adaptability and stability of the algorithm. The reduced learning rates, paved to a more exploitation after initial exploration. The visualizations provide a comprehensive overview of the agent’s learning journey and validate the effectiveness of the chosen hyper-parameters.

![CartPole A2C Avg](https://github.com/spattabirama/Actor-Critic/assets/124756255/68d4ae92-74e6-4240-97ef-ad6049dd0a86)

