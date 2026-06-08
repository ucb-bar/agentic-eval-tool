import math
from typing import Any

_NA = None

def mean_std(values: list) -> tuple:
    """Compute (mean, std) for a list, handling None and non-numeric. Returns (None, None) for empty."""
    nums = [v for v in values if v is not None and isinstance(v, (int, float))]
    if not nums:
        return None, None
    mean = sum(nums) / len(nums)
    if len(nums) < 2:
        return round(mean, 4), None
    variance = sum((x - mean) ** 2 for x in nums) / (len(nums) - 1)
    return round(mean, 4), round(math.sqrt(variance), 4)

def fmt(mean, std) -> str:
    if mean is None:
        return "NA"
    if std is None:
        return str(mean)
    return f"{mean} ± {std}"

def coerce_na(value: Any) -> str:
    """Convert None to 'NA' for CSV output."""
    if value is None:
        return "NA"
    return str(value)
