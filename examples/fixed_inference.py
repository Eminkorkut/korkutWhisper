"""Example script demonstrating fixed 3000-frame window transcription without G2 gating."""

import argparse
from korkutwhisper import KorkutWhisper

# Parse command-line arguments for model directory and input audio file path
p = argparse.ArgumentParser()
p.add_argument('model')
p.add_argument('audio')
a = p.parse_args()

# Run single-stage transcription with a fixed 3000-frame encoder window
with KorkutWhisper(a.model, strategy='fixed_3000') as asr:
    print(asr.transcribe(a.audio).text)

