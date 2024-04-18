import numpy as np

def weighted_random(min_val: int, max_val: int, scale: float = 2.0) -> int:
    # Define the scale to control the skewness
    scale = (max_val - min_val) / scale
    
    # Generate a random number from the exponential distribution
    random_exp = np.random.exponential(scale=scale)
    
    # Convert to an integer in the range [min_val, max_val]
    random_int = min_val + int(random_exp) % (max_val - min_val + 1)
    
    return random_int