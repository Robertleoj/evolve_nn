import math
import numpy as np

def replace_invalid_with_high(
    values: list[float] | np.ndarray, 
    high_value=20
) -> list[float]:
    return [high_value if math.isinf(x) or math.isnan(x) else min(x, high_value) for x in values]

