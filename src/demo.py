import gymnasium as gym
from cma_strat import cma_nn, cma_strat
from interfaces import policy_nn, optimization_strat
import numpy as np
import time

env = gym.make("CartPole-v1")
assert isinstance(env.action_space, gym.spaces.discrete.Discrete), "discrete action space is assumed."
assert isinstance(env.observation_space, gym.spaces.Box), "continuous action space is assumed."

def evaluate_policy(policy_nn: policy_nn, nreps, seed):

    rewards_reps = np.zeros(nreps)
    rs = np.random.RandomState(seed = seed)
    for rep_idx in range(nreps):
        observation, info = env.reset(seed=rs.randint(int(1e8)))
        terminated, truncated = False, False
        total_reward = 0

        while not (terminated or truncated): 
            output = policy_nn.get_output(observation)
            action = np.argmax(output) # this needs to be modified if action space is discrete
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        rewards_reps[rep_idx] = total_reward
    return np.mean(rewards_reps)

observation_space_dim = int(env.observation_space.shape[0])
action_space_dim = int(env.action_space.n)

assert len(env.observation_space.shape) == 1, "observation space needs to be flat" 

print("problem dim: ", observation_space_dim, action_space_dim)
strat = cma_strat(observation_space_dim, action_space_dim, 2)
seed=2
max_evals = 5000
rs = np.random.RandomState(seed=seed)
start_time = time.time()
for evaluation_idx in range(max_evals):
    nn = strat.show()
    f = evaluate_policy(nn, 1, rs.randint(int(1e8)))
    strat.tell(f)

    if evaluation_idx % 500 == 0:
        f = evaluate_policy(nn, 1000, seed) # Test seed always the same
        print(evaluation_idx / max_evals, f)
        strat.log(f"results/data/cartpole_{seed}.txt", f, evaluation_idx+1, env._elapsed_steps, time.time() - start_time)


env.close()