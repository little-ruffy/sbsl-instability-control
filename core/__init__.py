from __future__ import annotations

import os as _os

if _os.environ.get("SONOFUSION_ALLOW_F32", "0") != "1":
    import jax as _jax

    # force float64 before any array allocation
    _jax.config.update("jax_enable_x64", True)

__version__ = "0.1.0"

__all__ = ["__version__"]
