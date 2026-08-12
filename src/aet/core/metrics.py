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


# ---------------------------------------------------------------------------------------------
# Paired comparison + survival.
#
# The unpaired functions above (welch_ttest, effect_size) pool over seeds and compare two groups.
# That is the wrong test for an arm-vs-arm design where every seed is run under BOTH arms: pairing
# removes between-seed variance, which is usually the largest term. Pooling throws that away and
# then reports the resulting noise as an absence of effect.
#
# Two honesty constraints are built in rather than left to the caller:
#   * A paired test at small n has a floor on the p-value it can produce. At n=3 the smallest
#     two-sided Wilcoxon p is 0.25, so "p > 0.05" is arithmetically guaranteed and says nothing
#     about the effect. `underpowered` reports that, so a pilot cannot accidentally claim a null.
#   * A run that hit its budget without succeeding is CENSORED, not a failure and not a slow
#     success. Dropping those biases every time-to-X metric toward the lucky runs; substituting the
#     cap biases it the other way. Kaplan-Meier is what handles them correctly.
# ---------------------------------------------------------------------------------------------

#: Smallest achievable two-sided p for a Wilcoxon signed-rank test at n non-zero pairs. Below n=6 no
#: outcome can reach the conventional 0.05 threshold, whatever the effect size.
_WILCOXON_MIN_P = {1: 1.0, 2: 0.5, 3: 0.25, 4: 0.125, 5: 0.0625, 6: 0.03125}


def paired_deltas(a: list, b: list) -> "tuple[list, int]":
    """Element-wise ``a[i] - b[i]``, keeping only pairs where both sides are numbers.

    Returns ``(deltas, n_dropped)``. The count is returned rather than logged because a comparison
    computed over 2 of 3 seeds and one computed over 3 of 3 are different measurements, and the
    caller has to be able to say which one it reported.
    """
    if len(a) != len(b):
        raise ValueError(f"paired inputs must align: got {len(a)} and {len(b)}")
    deltas, dropped = [], 0
    for x, y in zip(a, b):
        ok = (isinstance(x, (int, float)) and not isinstance(x, bool)
              and isinstance(y, (int, float)) and not isinstance(y, bool))
        if ok:
            deltas.append(x - y)
        else:
            dropped += 1
    return deltas, dropped


def wilcoxon_signed_rank(a: list, b: list) -> dict:
    """Paired Wilcoxon signed-rank test over ``a`` vs ``b``.

    Returns a dict rather than a tuple because the p-value alone is misleading at pilot sizes::

        {"statistic", "p_value", "n_pairs", "n_nonzero", "n_dropped",
         "median_delta", "underpowered", "min_achievable_p"}

    ``underpowered`` is True when no outcome at this n could reach p < 0.05. Reporting a
    non-significant result without it invites the reader to conclude there is no effect, when the
    test could not have detected one.
    """
    deltas, dropped = paired_deltas(a, b)
    nonzero = [d for d in deltas if d != 0]
    n = len(nonzero)
    out = {
        "statistic": None, "p_value": None, "n_pairs": len(deltas), "n_nonzero": n,
        "n_dropped": dropped, "median_delta": None,
        "underpowered": True, "min_achievable_p": _WILCOXON_MIN_P.get(n, 0.0),
    }
    if deltas:
        s = sorted(deltas)
        mid = len(s) // 2
        out["median_delta"] = s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2
    if n == 0:
        return out
    out["underpowered"] = _WILCOXON_MIN_P.get(n, 0.0) > 0.05

    try:
        from scipy.stats import wilcoxon
        r = wilcoxon([x for x, y in zip(a, b) if isinstance(x, (int, float))
                      and isinstance(y, (int, float)) and x - y != 0],
                     [y for x, y in zip(a, b) if isinstance(x, (int, float))
                      and isinstance(y, (int, float)) and x - y != 0])
        out["statistic"], out["p_value"] = float(r.statistic), float(r.pvalue)
        return out
    except (ImportError, ValueError):
        pass

    # Exact stdlib fallback by enumerating sign assignments. Exact is feasible and correct at the
    # sizes this is for; a normal approximation at n=3 would be nonsense.
    ranks = _average_ranks([abs(d) for d in nonzero])
    w_plus = sum(r for d, r in zip(nonzero, ranks) if d > 0)
    w_minus = sum(r for d, r in zip(nonzero, ranks) if d < 0)
    stat = min(w_plus, w_minus)
    out["statistic"] = float(stat)
    if n > 20:
        mu = n * (n + 1) / 4
        sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        out["p_value"] = None if sigma == 0 else round(
            2 * (1 - _normal_cdf(abs(stat - mu) / sigma)), 6)
        return out
    total, at_least = 0, 0
    for mask in range(1 << n):
        total += 1
        wp = sum(r for i, r in enumerate(ranks) if mask >> i & 1)
        if min(wp, sum(ranks) - wp) <= stat:
            at_least += 1
    out["p_value"] = round(at_least / total, 6)
    return out


def _average_ranks(values: list) -> list:
    """Ranks 1..n with ties averaged (the standard signed-rank tie correction)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def paired_bootstrap_ci(a: list, b: list, confidence: float = 0.95,
                        n_resamples: int = 10000, seed: int = 0) -> dict:
    """Percentile bootstrap CI on the MEAN PAIRED DIFFERENCE ``mean(a - b)``.

    Seeded, so the same inputs give the same interval — an unseeded CI that moves between runs of
    the same analysis is not reportable.

    Returns ``{"mean_delta", "lower", "upper", "n_pairs", "n_dropped", "n_resamples"}``, with
    ``lower``/``upper`` ``None`` for fewer than 2 usable pairs. At pilot n the interval will be wide;
    that width IS the result, and it is more informative than a p-value the design cannot support.
    """
    import random

    deltas, dropped = paired_deltas(a, b)
    n = len(deltas)
    out = {"mean_delta": None, "lower": None, "upper": None,
           "n_pairs": n, "n_dropped": dropped, "n_resamples": n_resamples}
    if n == 0:
        return out
    out["mean_delta"] = round(sum(deltas) / n, 6)
    if n < 2:
        return out
    rng = random.Random(seed)
    means = []
    for _ in range(n_resamples):
        s = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(s) / n)
    means.sort()
    alpha = 1 - confidence
    lo = means[max(0, int(alpha / 2 * n_resamples) - 1)]
    hi = means[min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))]
    out["lower"], out["upper"] = round(lo, 6), round(hi, 6)
    return out


def kaplan_meier(durations: list, events: list) -> dict:
    """Kaplan-Meier survival estimate for time-to-success with right-censored observations.

    ``durations``  time observed for each run (to success, or to the budget cap).
    ``events``     1 if the run reached the endpoint, 0 if it was censored (hit its cap first).

    "Survival" here is the fraction NOT yet succeeded, so it starts at 1.0 and steps down. A censored
    run contributes to the risk set up to its cap and then leaves without an event — which is exactly
    the information a dropped run destroys and a run scored as "failed at the cap" fabricates.

    Returns ``{"times", "survival", "at_risk", "n", "n_events", "n_censored", "median"}``.
    ``median`` is ``None`` when survival never reaches 0.5 — not the largest duration, because
    "more than half never got there" is not a median.
    """
    pairs = [(float(d), int(e)) for d, e in zip(durations, events)
             if isinstance(d, (int, float)) and d is not None]
    if len(durations) != len(events):
        raise ValueError(f"durations and events must align: {len(durations)} vs {len(events)}")
    out = {"times": [], "survival": [], "at_risk": [], "n": len(pairs),
           "n_events": sum(e for _, e in pairs), "n_censored": sum(1 - e for _, e in pairs),
           "median": None}
    if not pairs:
        return out
    pairs.sort(key=lambda p: (p[0], -p[1]))   # at a tie, events precede censorings
    surv, n_at_risk = 1.0, len(pairs)
    i = 0
    while i < len(pairs):
        t = pairs[i][0]
        tied = [p for p in pairs if p[0] == t]
        d = sum(e for _, e in tied)
        if d:
            surv *= (1 - d / n_at_risk)
            out["times"].append(t)
            out["survival"].append(round(surv, 6))
            out["at_risk"].append(n_at_risk)
            if out["median"] is None and surv <= 0.5:
                out["median"] = t
        n_at_risk -= len(tied)
        i += len(tied)
    return out
