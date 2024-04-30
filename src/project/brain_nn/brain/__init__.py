import numpy as np
from dataclasses import dataclass
from uuid import uuid4
from collections import defaultdict


@dataclass
class AttentionMatrices:
    k_w: np.ndarray
    v_w: np.ndarray
    q_w: np.ndarray

@dataclass
class UpdateWeights:
    weights: list[tuple[AttentionMatrices, np.ndarray]]


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def attention(x: np.ndarray, weights: AttentionMatrices) -> np.ndarray:
    """Perform single-head attention on x with weights

    Args:
        x: a 2D array of shape (N, D)
        weights: The attention matrices

    Returns:
        np.ndarray: Output after applying attention mechanism
    """

    k = x @ weights.k_w
    v = x @ weights.v_w
    q = x @ weights.q_w

    # Scale dot product attention
    scale = np.sqrt(k.shape[1])
    attention_scores = (q @ k.T) / scale
    max_scores = np.max(attention_scores, axis=-1, keepdims=True)
    attention_probs = np.exp(attention_scores - max_scores)
    attention_probs /= np.sum(attention_probs, axis=1, keepdims=True)

    return attention_probs @ v

def get_update_coefficients(x: np.ndarray, weights: UpdateWeights) -> np.ndarray:
    """Compute update coefficients

    Args:
        x: a 2D array of shape N x D
        weights: The update weights
    
    Returns:
        A 2D array of shape N x D
    """

    h = x
    for attention_weights, linear in weights.weights:
        h = attention(h, attention_weights) + h
        h = h @ linear + h
        h = relu(h)

    return h

def get_update(
    input: np.ndarray, 
    current_weights: np.ndarray, 
    update_weights: UpdateWeights,
    weight_update_rate: float
) -> np.ndarray:
    
    update_coefficients = get_update_coefficients(input, update_weights)

    outputs = (1 - weight_update_rate) * current_weights + weight_update_rate * update_coefficients

    return outputs

def make_update_weights(num_transmitters: int, depth: int) -> UpdateWeights:
    denominator = np.sqrt(num_transmitters)
    attention_weights = AttentionMatrices(
        k_w=np.random.randn(num_transmitters, num_transmitters) / num_transmitters,
        v_w=np.random.randn(num_transmitters, num_transmitters) / num_transmitters,
        q_w=np.random.randn(num_transmitters, num_transmitters) / num_transmitters
    )

    linear_weights = np.random.randn(num_transmitters, num_transmitters) / num_transmitters

    return UpdateWeights(weights=[(attention_weights, linear_weights) for _ in range(depth)])


class BrainNode:

    transform_weights: np.ndarray
    input_weights: np.ndarray

    def __init__(
        self, 
        n_inputs: int,
        num_transmitters: int
    ) -> None:
        # self.transform_weights = np.random.rand(num_transmitters, num_transmitters) / np.sqrt(2)

        # for now, we use a soft identity matrix
        softness = 0.9
        self.transform_weights = np.eye(num_transmitters) * (softness - (1 - softness))
        self.transform_weights += (1 - softness)
        
        self.input_weights = np.random.rand(n_inputs, num_transmitters) / np.sqrt(2)

    def forward(
        self, 
        inputs: list[np.ndarray], 
        update_weights: UpdateWeights | None = None,
        weight_update_rate: float | None = None
    ) -> np.ndarray:
        inputs_np = np.stack(inputs, axis=0)

        h = inputs_np * self.input_weights

        # weighted sum of inputs
        h = np.sum(h, axis=0)

        h =  h @ self.transform_weights

        if update_weights is not None:
            assert weight_update_rate is not None
            self.input_weights = get_update(
                inputs_np, 
                self.input_weights, update_weights,
                weight_update_rate
            )

        return relu(h)

        
@dataclass
class Brain:
    id_to_node: dict[str, BrainNode]
    reverse_adj_list: dict[str, set[str]]

    input_nodes: list[str]
    response_input_nodes: list[str]
    output_nodes: list[str]

    update_weights: UpdateWeights

    curr_outputs: dict[str, np.ndarray] | None = None



def make_brain(
    num_inputs: int, 
    num_outputs: int, 
    num_response_input_nodes: int,
    num_transmitters: int
) -> Brain:

    # make inputs, outputs, and response input nodes
    input_node_ids: list[str] = [
        str(uuid4()) for _ in range(num_inputs)
    ]

    output_node_ids: list[str] = [
        str(uuid4()) for _ in range(num_outputs)
    ]

    response_input_node_ids = [
        str(uuid4()) for _ in range(num_response_input_nodes)
    ]

    reverse_adj_list: defaultdict[str, set[str]] = defaultdict(set)

    # make one auxiliary node
    aux_node_id = str(uuid4())

    # add input nodes to its inputs
    reverse_adj_list[aux_node_id].update(input_node_ids)
    reverse_adj_list[aux_node_id].update(response_input_node_ids)


    # add aux node to output nodes
    for output_node_id in output_node_ids:
        reverse_adj_list[output_node_id].add(aux_node_id)

    
    update_weights = make_update_weights(num_transmitters, 2)

    # make all the nodes
    id_to_node: dict[str, BrainNode] = {}

    # inputs, outputs, and response inputs have one input
    for node_id in input_node_ids + response_input_node_ids + output_node_ids:
        id_to_node[node_id] = BrainNode(1, num_transmitters)

    # the aux node has num_inputs + num_response_input_nodes inputs
    id_to_node[aux_node_id] = BrainNode(num_inputs + num_response_input_nodes, num_transmitters)


    return Brain(
        id_to_node=id_to_node,
        reverse_adj_list=reverse_adj_list,
        input_nodes=input_node_ids,
        response_input_nodes=response_input_node_ids,
        output_nodes=output_node_ids,
        update_weights=update_weights,
    )
    

    
    