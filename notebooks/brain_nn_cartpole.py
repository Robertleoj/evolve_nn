# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: project-xOhHZUaJ-py3.10
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2
import numpy as np
from IPython.display import SVG, display
from project.brain_nn.brain import Brain, make_brain, show_brain
from project.brain_nn.type_defs import EvolutionConfig
from project.brain_nn.evolution.mutations import mutate_brain
import matplotlib.pyplot as plt
from project.utils.rand_utils import softmax
import multiprocess as mp
from typing import Callable
from tqdm import tqdm
from project.brain_nn.evolution.select_and_mutate import select_and_mutate
import gymnasium as gym
from project.utils.paths import get_date_path, get_results_dir
import warnings
warnings.filterwarnings('ignore')

# %%
# CartPole
# ENV_NAME = 'CartPole-v1'
# ev_config = EvolutionConfig(
#     steps_per_action=10,
#     num_episodes_per_training=20,
#     population_size=100,
#     num_transmitters=2,
#     softmax_temp=0.3,
#     init_weight_update_rate=0.05,
#     weight_update_depth=2,
#     top_k_stay=3
# )

ENV_NAME = 'MountainCarContinuous-v0'
ev_config = EvolutionConfig(
    steps_per_action=5,
    num_episodes_per_training=5,
    population_size=100,
    num_transmitters=2,
    softmax_temp=0.3,
    init_weight_update_rate=0.05,
    weight_update_depth=2,
    top_k_stay=3
)

# %%
env = gym.make(ENV_NAME)

action_space = env.action_space
continuous_actions: bool
if isinstance(action_space, gym.spaces.box.Box):
    num_actions = action_space.shape[0]
    continuous_actions = True
else:
    num_actions = action_space.n
    continuous_actions = False

num_inputs = env.observation_space.shape[0]
print(f"{num_actions=}, {continuous_actions=}, {num_inputs=}")

# %%

observation, info = env.reset()
print(observation)

action = env.action_space.sample()  # agent policy that uses the observation and info
print(action)
observation, reward, terminated, truncated, info = env.step(action)
print(reward)

if terminated or truncated:
    observation, info = env.reset()


# %%
def initialize_population(num_inputs, num_actions, evolution_config: EvolutionConfig):
    population = [
        mutate_brain(make_brain(
            num_inputs=num_inputs,
            num_outputs=num_actions,
            num_response_input_nodes=1,
            update_rate=evolution_config.init_weight_update_rate,
            num_transmitters=evolution_config.num_transmitters,
        ))
        for _ in range(evolution_config.population_size)
    ]
    return population


# %%
def prepare_inputs(observation: np.ndarray, evolution_config: EvolutionConfig):
    inputs = []
    for v in observation:
        inputs.append(v * np.ones(evolution_config.num_transmitters))

    return inputs

def prepare_reward(reward: np.ndarray, evolution_config: EvolutionConfig):
    return [reward * np.ones(evolution_config.num_transmitters)]

def get_action_discrete(output: list[np.ndarray]):
    probabilities= softmax(np.mean(output, axis=1))
    return np.argmax(probabilities)

def get_action_continuous(output: list[np.ndarray], range_min: float, range_max: float):
    # assume output has range -1 to 1
    action_normalized = np.mean(output, axis=1)
    action = ((action_normalized + 1) / 2) * (range_max - range_min) + range_min
    return action



# %%
population = initialize_population(num_inputs, num_actions, ev_config)

# %%


def evaluate_population(population: list[Brain], env: gym.Env, evolution_config: EvolutionConfig) -> list[float]:

    fitness_scores = []
    tq = tqdm(enumerate(population))
    best_fitness = 0
    for i, brain in tq:
        try:
            brain.clear_state()

            returns = []
            current_ep_return = 0
            reward = 0

            observation, info = env.reset()

            num_episodes = 0
            while True:

                prepared_inputs = prepare_inputs(observation, evolution_config)
                prepared_reward = prepare_reward(reward, evolution_config)
                last_output = None

                for step in range(ev_config.steps_per_action):
                    last_output = brain.step(
                        inputs = prepared_inputs,
                        last_responses=prepared_reward
                    )
                
                if isinstance(env.action_space, gym.spaces.box.Box):
                    action = get_action_continuous(last_output, env.action_space.low, env.action_space.high)
                else:
                    action = get_action_discrete(last_output)
                # action = get_action(last_output, evolution_config)

                observation, reward, term, trunc, info = env.step(action)
                done = term or trunc

                current_ep_return += reward

                if done:
                    num_episodes += 1
                    observation, info = env.reset()
                    returns.append(current_ep_return)
                    current_ep_return = 0
                    if num_episodes >= ev_config.num_episodes_per_training:
                        break
        except Exception as e:
            print(f"Error in brain {i+1}/{len(population)}: {e}")
            returns = [0]
                    

        brain_fitness = np.mean(returns)
        fitness_scores.append(float(brain_fitness))
        if brain_fitness > best_fitness:
            best_fitness = brain_fitness
        tq.set_description(f"Brain {i+1}/{len(population)} fitness: {brain_fitness}, best: {best_fitness}")
    
    return fitness_scores

# %%
import traceback


def evaluate_single_brain(args: tuple) -> float:
    brain, env_factory, evolution_config, idx, total = args
    env = env_factory()  # Create a new environment for each brain
    brain.clear_state()

    returns = []
    current_ep_return = 0
    reward = 0

    observation, info = env.reset()
    num_episodes = 0
    try:
        while True:
            prepared_inputs = prepare_inputs(observation, evolution_config)
            prepared_reward = prepare_reward(reward, evolution_config)
            last_output = None

            for step in range(evolution_config.steps_per_action):
                last_output = brain.step(inputs=prepared_inputs, last_responses=prepared_reward)
            
            if isinstance(env.action_space, gym.spaces.box.Box):
                action = get_action_continuous(last_output, env.action_space.low, env.action_space.high)
            else:
                action = get_action_discrete(last_output)
            # action = get_action(last_output, evolution_config)
            observation, reward, term, trunc, info = env.step(action)
            done = term or trunc

            current_ep_return += reward

            if done:
                num_episodes += 1
                observation, info = env.reset()
                returns.append(current_ep_return)
                current_ep_return = 0
                brain.clear_state()
                if num_episodes >= evolution_config.num_episodes_per_training:
                    break

    except Exception as e:
        returns = [0]
        traceback.print_exception(e)

    return float(np.mean(returns)), brain

def evaluate_population(population: list[Brain], env_factory: Callable, evolution_config: EvolutionConfig) -> list[float]:
    num_workers = mp.cpu_count() - 2  # Determine the number of workers based on available CPU cores
    with mp.Pool(processes=num_workers) as pool:
        args = [(brain, env_factory, evolution_config, i, len(population)) for i, brain in enumerate(population)]
        results = list(tqdm(pool.imap(evaluate_single_brain, args), total=len(args), desc="Evaluating population"))

    fitness_scores, population = zip(*results)

    return fitness_scores, population


# %%
from project.brain_nn.brain import get_brain_svg


def report_data(population, fitness_scores, probs, gen, pth):

    should_print = gen % 10 == 0
    out_path = pth / f"gen_{gen}"
    out_path.mkdir(parents=True, exist_ok=True)

    ordered = sorted(zip(population, fitness_scores, probs), key=lambda x: x[1], reverse=True)

    if should_print:
        print("Best score: ", ordered[0][1])

    brain_svg_path = out_path / "best_brain.svg"
    # save the best brain
    brain_svg = get_brain_svg(ordered[0][0])
    brain_svg_path.write_text(brain_svg)

    if should_print:
        print("Best brain:")
        display(SVG(brain_svg))

    plt.bar(range(len(fitness_scores)), sorted(fitness_scores, reverse=True))
    plt.title("Fitness scores")
    # save the plot
    plt.savefig(out_path / "fitness_scores.png")
    if should_print:
        plt.show()
    else:
        plt.close()

    plt.bar(range(len(fitness_scores)), sorted(probs, reverse=True))
    plt.title("Probabilities")

    plt.savefig(out_path / "probabilities.png")
    if should_print:
        plt.show()
    else:
        plt.close()


# %%
def env_factory(human=False):
    if human:
        return gym.make(ENV_NAME, render_mode='human')
    return gym.make(ENV_NAME)


# %%
def play(brain, n=1, max_steps:int = None):
    env = env_factory(human=True)
    observation, info = env.reset()
    num_steps = 0
    num_episodes = 0
    reward = 0
    while True:
        prepared_inputs = prepare_inputs(observation, ev_config)
        prepared_reward = prepare_reward(reward, ev_config)
        last_output = None

        for step in range(ev_config.steps_per_action):
            last_output = brain.step(inputs=prepared_inputs, last_responses=prepared_reward, update=True)
        
        if isinstance(env.action_space, gym.spaces.box.Box):
            action = get_action_continuous(last_output, env.action_space.low, env.action_space.high)
        else:
            action = get_action_discrete(last_output)

        observation, reward, term, trunc, info = env.step(action)
        done = term or trunc
        if done or (max_steps is not None and num_steps >= max_steps):
            num_episodes += 1
            if num_episodes >= n:
                env.close()
                break
            observation, info = env.reset()
            print(f"Episode {num_episodes} finished with {num_steps} steps")
            num_steps = 0

        num_steps += 1


# %%

from copy import deepcopy


def evolve(num_steps: int, population: list[Brain], env_factory: Callable[..., gym.Env], evolution_config: EvolutionConfig) -> list[Brain]:
    output_path = get_results_dir() / get_date_path()
    output_path.mkdir(parents=True)
    for gen in range(num_steps):
        population_before = deepcopy(population)
        fitness_scores, learned_population = evaluate_population(population, env_factory, evolution_config)

        # if gen % 5 == 0:
        #     best_brain_idx = np.argmax(fitness_scores)
        #     best_brain = population[best_brain_idx]
        #     play(best_brain)
            
        yield learned_population, fitness_scores

        new_population, probs = select_and_mutate(learned_population, fitness_scores, evolution_config)

        report_data(learned_population, fitness_scores, probs, gen, output_path)

        population = new_population


# %%

i = 0
for pop in evolve(100, population, env_factory, ev_config):
    population, scores = pop

# %%
np.argmax(scores)
print(scores)

# %%
# best_score_idx = 46
best_score_idx = np.argmax(scores)
print(f"Best score: {scores[best_score_idx]}")
best_brain = deepcopy(population[best_score_idx])
    
play(best_brain, n=1)
# for i, brain in enumerate(population):
    # if i < 20:
        # continue
    # print(i)
    # play(brain, n=1, max_steps=100)
    # play(brain, n=1)

# %%
show_brain(best_brain)

# %%
print(best_score_idx)

# %%
print(scores)
