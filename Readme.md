


## Process
### Data Engineering

```python
from seg_audio import *
seg = SegAudio()
seg.load_train_wavs()
seg.load_val_wavs()
seg.update_train_batch_wavs2specs()
seg.update_val_batch_wavs2specs()

```