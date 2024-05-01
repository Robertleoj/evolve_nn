from project.brain_nn.brain import AttentionMatrices, Brain, is_valid, UpdateWeights
import numpy as np
import random
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)

NUM_TRIES = 100
NUM_MUTATIONS = 3



def delete_edge(brain: Brain) -> tuple[Brain, bool]:
    edge_list = brain.edge_list()

    for _ in range(NUM_TRIES):
        random_edge = random.choice(edge_list)
        new_brain = deepcopy(brain)
        new_brain.delete_edge(random_edge)

        if is_valid(new_brain)[0]:
            return new_brain, True

    return brain, False

def add_node(brain: Brain) -> tuple[Brain, bool]:
    # note that we can always add a new node, however we want. 
    # For future compatibility though, we will check validity.

    input_node_ids = list(brain.id_to_node.keys())
    output_node_ids = list(brain.brain_nodes())

    for _ in range(NUM_TRIES):
        num_inputs = random.randint(1, 10)
        input_nodes = random.choices(input_node_ids, k=num_inputs)
        num_outputs = random.randint(1, 5)
        output_nodes = random.choices(output_node_ids, k=num_outputs)

        new_brain = deepcopy(brain)
        new_brain.add_node(input_nodes, output_nodes)
        if is_valid(new_brain)[0]:
            return new_brain, True

    return new_brain, False

def delete_node(brain: Brain) -> tuple[Brain, bool]:
    node_ids = list(brain.brain_nodes())

    for _ in range(NUM_TRIES):
        node_id = random.choice(node_ids)
        new_brain = deepcopy(brain)
        new_brain.delete_node(node_id)

        if is_valid(new_brain)[0]:
            return new_brain, True

    return brain, False

def add_edge(brain: Brain) -> tuple[Brain, bool]:
    output_node_ids = list(brain.id_to_node.keys())
    input_node_ids = list(brain.brain_nodes())

    edge_list = brain.edge_list()

    for _ in range(NUM_TRIES):
        random_edge = (random.choice(output_node_ids), random.choice(input_node_ids))

        if random_edge[0] == random_edge[1]:
            continue

        if random_edge in edge_list:
            continue
            
        new_brain = deepcopy(brain)
        new_brain.add_edge(random_edge)

        if is_valid(new_brain)[0]:
            return new_brain, True

    return brain, False

def perturb_array(arr: np.ndarray, perturbation: float) -> np.ndarray:
    return arr + np.random.randn(*arr.shape) * perturbation

def mutate_update_weights(brain: Brain) -> tuple[Brain, bool]:
    rate_perturbation = 0.05
    new_rate = random.uniform(1 - rate_perturbation, 1 + rate_perturbation) * brain.update_rate

    new_weights = deepcopy(brain.update_weights)

    weight_perturbation = 0.1
    for i, (attention_weights, linear_weights) in enumerate(new_weights.weights):

        attention_weights.k_w = perturb_array(attention_weights.k_w, weight_perturbation)
        attention_weights.v_w = perturb_array(attention_weights.v_w, weight_perturbation)
        attention_weights.q_w = perturb_array(attention_weights.q_w, weight_perturbation)

        linear_weights += np.random.randn(*linear_weights.shape) * weight_perturbation

        new_weights.weights[i] = (attention_weights, linear_weights)

    new_brain = deepcopy(brain)

    new_brain.update_rate = new_rate

    return new_brain, True
    

def recombine_update_weights(weights1:UpdateWeights , weights2: UpdateWeights) -> UpdateWeights:
    new_weights = deepcopy(weights1)

    for i, ((aw1, lw1), (aw2, lw2)) in enumerate(zip(weights1.weights, weights2.weights)):
        new_linear = (lw1 + lw2) / 2
        new_k_w = (aw1.k_w + aw2.k_w) / 2
        new_q_w = (aw1.q_w + aw2.q_w) / 2
        new_v_w = (aw1.v_w + aw2.v_w) / 2

        new_aw = AttentionMatrices(
            k_w=new_k_w,v_w=new_v_w,q_w=new_q_w
        )

        new_weights.weights[i] = (new_aw, new_linear)

    return new_weights
    

def recombine_brains(brain1: Brain, brain2: Brain) -> Brain:
    new_brain = deepcopy(brain1)
    new_brain.update_weights = recombine_update_weights(brain1.update_weights, brain2.update_weights)
    new_brain.update_rate = (brain1.update_rate + brain2.update_rate) / 2
    return new_brain



def mutate_brain(brain: Brain) -> Brain:
    mut_probabilities = {
        'delete_edge': 1.,
        'add_node': 0.2,
        'delete_node': 0.2,
        'add_edge': 1,
        'mutate_update_weights': 0.5
    }

    name_to_func = {
        'delete_edge': delete_edge,
        'add_node': add_node,
        'delete_node': delete_node,
        'add_edge': add_edge,
        'mutate_update_weights': mutate_update_weights
    }

    names = list(mut_probabilities.keys())
    probs = list(mut_probabilities.values())

    num_successful_mutations = 0
    while num_successful_mutations < NUM_MUTATIONS:
        random_mut = random.choices(names, weights=probs)[0]
        mut_brain, success = name_to_func[random_mut](brain)
        logger.info(f"Mutation: {random_mut}, Success: {success}")
        

        if success:
            num_successful_mutations += 1
            brain = mut_brain

    return brain
            
            

