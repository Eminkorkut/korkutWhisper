"""Example script demonstrating sequential batch transcription across multiple audio files."""

import argparse
from korkutwhisper import KorkutWhisper

# Parse command-line arguments for model directory and one or more audio file paths
p = argparse.ArgumentParser()
p.add_argument('model')
p.add_argument('audio', nargs='+')
a = p.parse_args()

# Transcribe each audio file sequentially and print either its text or error details
with KorkutWhisper(a.model) as asr:
    for item in asr.transcribe_many(a.audio):
        print(
            item.index,
            item.result.text if item.result else f'{item.error_type}: {item.error_message}',
        )

