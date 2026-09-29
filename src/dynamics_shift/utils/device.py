"""Explicit execution device selection with a real CUDA preflight."""
import torch


def check_device(requested: str) -> torch.device:
    """Fail before training if CUDA initialization or a GPU kernel fails."""
    device = torch.device(requested)
    if device.type == "cpu":
        print("Device: cpu (explicitly selected)", flush=True)
        return device
    if device.type != "cuda":
        raise ValueError("Supported devices: cpu, cuda, cuda:N")
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is False")
        index = device.index if device.index is not None else 0
        if index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device index {index} is unavailable")
        device = torch.device(f"cuda:{index}")
        # Availability alone does not prove a usable driver/kernel combination.
        with torch.cuda.device(device), torch.no_grad():
            probe = torch.ones((8, 8), device=device)
            result = probe @ probe
            torch.cuda.synchronize(device)
            if not torch.all(result == 8).item():
                raise RuntimeError("CUDA arithmetic check failed")
        print(f"GPU ready: {device} | {torch.cuda.get_device_name(device)} | "
              f"PyTorch {torch.__version__} | CUDA build {torch.version.cuda}", flush=True)
        return device
    except Exception as error:
        raise RuntimeError(
            f"GPU preflight failed for {requested}: {error}. No CPU fallback. "
            "Check nvidia-smi, CUDA_VISIBLE_DEVICES, and the PyTorch/driver installation."
        ) from error
