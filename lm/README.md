# Contextual correction (TODO)

Cleans up raw CTC output (`model/decoder.py`) using ham-radio text structure:
callsigns, Q-codes (QTH, QSL, QRM...), prosigns (CQ, DE, K, KN, 73, BT), and
common QSO phrasing. Generic English language models will mis-correct this
text, so this needs a small model/constrained decoder trained on real ham CW
text, not a generic spell-checker.

Build this after `model/train.py` has a real trained checkpoint to evaluate
against - until then there's no decoder output to correct.

Planned approach: assemble a corpus from ARRL transcripts + public Q-code/
prosign lists + sample QSO logs, then either a small char-level seq2seq
correction model or a constrained beam search over the acoustic model's CTC
output using an n-gram LM built from that corpus.
