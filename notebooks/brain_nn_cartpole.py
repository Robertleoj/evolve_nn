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
from project.brain_nn.brain import Brain, make_brain, show_brain, save_population, load_population
from project.brain_nn.type_defs import EvolutionConfig
from project.brain_nn.evolution.mutations import mutate_brain
import matplotlib.pyplot as plt
from project.utils.rand_utils import softmax
from pathlib import Path
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

# ENV_NAME = 'MountainCarContinuous-v0'
# ev_config = EvolutionConfig(
#     steps_per_action=5,
#     num_episodes_per_training=5,
#     population_size=100,
#     num_transmitters=2,
#     softmax_temp=0.3,
#     init_weight_update_rate=0.05,
#     weight_update_depth=2,
#     top_k_stay=3
# )

# ENV_NAME = 'Pendulum-v1'
# ev_config = EvolutionConfig(
#     steps_per_action=10,
#     num_episodes_per_training=5,
#     population_size=100,
#     num_transmitters=2,
#     softmax_temp=0.3,
#     init_weight_update_rate=0.05,
#     weight_update_depth=2,
#     top_k_stay=3
# )

ENV_NAME = "LunarLander-v2"
ev_config = EvolutionConfig(
    steps_per_action=10,
    num_episodes_per_training=20,
    population_size=100,
    num_transmitters=2,
    softmax_temp=0.1,
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
# pop_path = Path("/home/bigbrainman/projects/evolve_nn/runs/20240501_182101/gen_23/next_population.pkl")
# population = load_population(pop_path)

# %%

def generate_increasing_weights(n: int) -> np.ndarray:
    weights = np.arange(1, n + 1)
    total_weight = np.sum(weights)
    normalized_weights = weights / total_weight
    return normalized_weights

def weighted_sum_of_transformed_array(transformed_array: np.ndarray) -> np.ndarray:
    n, k, d = transformed_array.shape
    
    # Generate weights
    weights = generate_increasing_weights(n)[:, np.newaxis, np.newaxis]  # reshape for broadcasting
    
    # Compute the weighted sum using broadcasting
    weighted_sum = np.sum(weights * transformed_array, axis=0)
    
    return weighted_sum

def get_action(prepared_inputs, prepared_reward, brain, env, evolution_config):
    brain_outputs = []
    for step in range(evolution_config.steps_per_action):
        last_output = brain.step(inputs=prepared_inputs, last_responses=prepared_reward)
        brain_outputs.append(last_output)

        
    brain_outputs_np = np.array(brain_outputs)
    summed = weighted_sum_of_transformed_array(brain_outputs_np)
    
    if isinstance(env.action_space, gym.spaces.box.Box):
        action = get_action_continuous(summed, env.action_space.low, env.action_space.high)
    else:
        action = get_action_discrete(summed)

    return action

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
            action = get_action(prepared_inputs, prepared_reward, brain, env, evolution_config)

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


    best_brain = ordered[0][0]
    play(best_brain, n=2, record_path=out_path / "best_brain_episodes")


# %%
def env_factory(render_mode: str | None = None):
    if render_mode:
        return gym.make(ENV_NAME, render_mode=render_mode)
    return gym.make(ENV_NAME)


# %%
def play(brain, n=1, max_steps:int = None, record_path: Path | None=None):
    env = env_factory(render_mode='human' if record_path is None else 'rgb_array')

    if record_path:
        env = gym.wrappers.RecordVideo(
            env, video_folder=str(record_path), episode_trigger=lambda x: True, name_prefix="best_brain"
        )
        
    observation, info = env.reset()
    num_steps = 0
    num_episodes = 0
    reward = 0
    while True:
        prepared_inputs = prepare_inputs(observation, ev_config)
        prepared_reward = prepare_reward(reward, ev_config)
        action = get_action(prepared_inputs, prepared_reward, brain, env, ev_config)
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

        fitness_scores, learned_population = evaluate_population(population, env_factory, evolution_config)
        # learned_population = population

        # if gen % 5 == 0:
        #     best_brain_idx = np.argmax(fitness_scores)
        #     best_brain = population[best_brain_idx]
        #     play(best_brain)
            

        new_population, probs = select_and_mutate(learned_population, fitness_scores, evolution_config)
        yield new_population, learned_population, fitness_scores

        report_data(learned_population, fitness_scores, probs, gen, output_path)

        pth = output_path / f"gen_{gen}"
        save_population(learned_population, pth / "population.pkl")
        save_population(new_population, pth / "next_population.pkl")

        population = new_population

# %%

i = 0
for pop in evolve(100, population, env_factory, ev_config):
    new_population, old_population, scores = pop
    population = new_population

# %%
np.argmax(scores)
print(scores)

# %%
# best_score_idx = 46
best_score_idx = np.argmax(scores)
print(f"Best score: {scores[best_score_idx]}")
best_brain = deepcopy(old_population[best_score_idx])
    
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
