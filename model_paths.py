"""Central registry of local model-weight directories for designlora.

The weights were relocated off the (near-full) root partition to /data1/models.
Override the base directory with the DESIGNLORA_MODELS_DIR environment variable,
e.g. ``export DESIGNLORA_MODELS_DIR=/some/other/disk/models``.
"""
import os

MODELS_DIR = os.environ.get("DESIGNLORA_MODELS_DIR", "/data1/models")

FLUX2_DEV = os.path.join(MODELS_DIR, "flux2dev")
QWEN_IMAGE = os.path.join(MODELS_DIR, "Qwen-Image")
IDEFICS2 = os.path.join(MODELS_DIR, "idefics2")
