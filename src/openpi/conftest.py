import os

import pytest


def set_jax_cpu_backend_if_no_gpu() -> None:
    try:
        import pynvml  # type: ignore

        try:
            pynvml.nvmlInit()
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            os.environ["JAX_PLATFORMS"] = "cpu"
    except ModuleNotFoundError:
        # Treat as "no GPU found" when pynvml isn't installed.
        os.environ["JAX_PLATFORMS"] = "cpu"


def pytest_configure(config: pytest.Config) -> None:
    set_jax_cpu_backend_if_no_gpu()
