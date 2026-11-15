"""Example script demonstrating explicit KorkutConfig construction and result serialization."""

import argparse
from korkutwhisper import (
    KorkutWhisper,
    KorkutConfig,
    WhisperConfig,
    InferenceConfig,
    GateConfig,
)

# Parse command-line arguments for model directory and input audio file path
p = argparse.ArgumentParser()
p.add_argument('model')
p.add_argument('audio')
a = p.parse_args()

# Construct an explicit composite configuration object
config = KorkutConfig(
    whisper=WhisperConfig(model_path=a.model),
    inference=InferenceConfig(strategy='g2', seed=42),
    gate=GateConfig(),
)

# Initialize KorkutWhisper from the config and print serialized config and result dicts
with KorkutWhisper(config=config) as asr:
    print(asr.get_config())
    print(asr.transcribe(a.audio).to_dict())

