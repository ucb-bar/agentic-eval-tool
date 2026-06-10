import math
from typing import Any

_NA = None

def welch_ttest(a: list, b: list) -> tuple:
    """Welch's t-test. Returns (t_stat, p_value). (None, None) if n<2 in either group."""
    a = [v for v in a if v is not None and isinstance(v, (int, float))]
    b = [v for v in b if v is not None and isinstance(v, (int, float))]
    if len(a) < 2 or len(b) < 2:
        return None, None
    try:
        from scipy.stats import ttest_ind
        r = ttest_ind(a, b, equal_var=False)
        return float(r.statistic), float(r.pvalue)
    except ImportError:
        pass
    na, nb = len(a), len(b)
    ma, mb = sum(a) / na, sum(b) / nb
    va = sum((x - ma) ** 2 for x in a) / (na - 1)
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return 0.0, 1.0
    t = (ma - mb) / se
    p = 2 * (1 - _normal_cdf(abs(t)))
    return round(t, 6), round(max(0.0, min(1.0, p)), 6)


def _normal_cdf(z: float) -> float:
    return 0.5 * math.erfc(-z / math.sqrt(2))


def confidence_interval(values: list, confidence: float = 0.95) -> tuple:
    """Returns (lower, upper) CI. (None, None) for n<2."""
    nums = [v for v in values if v is not None and isinstance(v, (int, float))]
    if len(nums) < 2:
        return None, None
    n = len(nums)
    mean = sum(nums) / n
    var = sum((x - mean) ** 2 for x in nums) / (n - 1)
    se = math.sqrt(var / n)
    alpha = 1 - confidence
    try:
        from scipy.stats import t as t_dist
        t_crit = float(t_dist.ppf(1 - alpha / 2, df=n - 1))
    except ImportError:
        t_crit = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence, 1.96)
    margin = t_crit * se
    return round(mean - margin, 6), round(mean + margin, 6)


def effect_size(a: list, b: list) -> "float | None":
    """Cohen's d. None if either group <2 or pooled_std=0."""
    a = [v for v in a if v is not None and isinstance(v, (int, float))]
    b = [v for v in b if v is not None and isinstance(v, (int, float))]
    if len(a) < 2 or len(b) < 2:
        return None
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a) / (len(a) - 1)
    vb = sum((x - mb) ** 2 for x in b) / (len(b) - 1)
    pooled = math.sqrt((va + vb) / 2)
    if pooled == 0:
        return None
    return round((ma - mb) / pooled, 6)


def jaccard_similarity(seq_a: list, seq_b: list) -> float:
    """Set-based Jaccard similarity. Returns 1.0 for two empty sequences."""
    sa, sb = set(seq_a), set(seq_b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def sequence_edit_distance(seq_a: list, seq_b: list) -> int:
    """Levenshtein distance (Wagner-Fischer DP, two-row optimisation)."""
    m, n = len(seq_a), len(seq_b)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            if seq_a[i - 1] == seq_b[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev = curr
    return prev[n]


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
