"""Shared path config.

Code lives in this repo (synced via OneDrive between machines). Bulk
generated/downloaded data (audio clips, manifests, synthetic corpora) does
NOT - it's regenerable junk, often large, and syncing gigabytes of WAV files
through OneDrive between the laptop and the RTX 3080 desktop is pure waste.
So DATA_ROOT defaults to a local-only path outside the OneDrive tree; each
machine gets its own copy, produced by re-running the dataprep/model scripts.

Override with the MORSE_AI_DATA environment variable if you want it elsewhere.
"""
import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("MORSE_AI_DATA", r"C:\morse-ai-data"))
