import time
def monotonic_start() -> float:
    return time.monotonic()
def elapsed(start: float) -> float:
    return round(time.monotonic() - start, 3)
