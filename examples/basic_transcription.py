"""Example script demonstrating single-file transcription with G2 gate diagnostics."""

import argparse
from korkutwhisper import KorkutWhisper

# Parse command-line arguments for model directory and input audio file path
p = argparse.ArgumentParser()
p.add_argument('model')
p.add_argument('audio')
a = p.parse_args()

# Run transcription using the default G2 strategy and print text and gate risk score
with KorkutWhisper(a.model) as asr:
    result = asr.transcribe(a.audio)
    print(result.text)
    if result.gate:
        print('risk score:', result.gate.risk_score, 'retry:', result.retry_performed)

