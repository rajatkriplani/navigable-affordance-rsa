"""
Device resolution for PyTorch-based stages.

Torch is imported lazily so that analysis-only modules (stats, io) can run
in environments without torch installed.
"""


def resolve_device(requested="auto"):
    """
    Resolve a device request to a concrete torch device string.

    Parameters
    ----------
    requested : str
        "auto" (pick best available), or one of "cuda", "mps", "cpu".

    Returns
    -------
    str
        One of "cuda", "mps", "cpu".
    """
    allowed = {"auto", "cuda", "mps", "cpu"}
    if requested not in allowed:
        raise ValueError(
            f"requested must be one of {allowed}, got {requested!r}"
        )

    if requested != "auto":
        return requested

    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def device_description(device):
    """Human-readable one-line description for logging."""
    if device == "cuda":
        try:
            import torch
            name = torch.cuda.get_device_name(0)
            return f"cuda ({name})"
        except Exception:
            return "cuda"
    if device == "mps":
        return "mps (Apple Silicon)"
    return "cpu"