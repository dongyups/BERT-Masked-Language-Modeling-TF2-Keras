# BERT-Masked-Language-Modeling-tf2-keras
original (tensorflow 1.X version): https://github.com/google-research/bert

### Required
This code is tested with python==3.8, tensorflow-gpu==2.7, and cuda/cudnn 11.2/8.1
- Python 3.X
- Tensorflow 2.X
- CUDA 10.X or 11.X for gpu settings depends on your hardware

### Differences from the original code
The original BERT pre-training code is supposed to be run by two tasks: MLM & NSP.
In this code, however, NSP is intentionally excluded thus the original segment embedding is omitted in the inputs.

### How to use
Upload your own datasets in `datasets` folder and load them inside `run_pretraining.py` and `run_classifier`.\
Refer to the annotations. FP16 option is set to default and the indices of GPUs support the multi-GPU option.
```python
# train with val
python run_pretraining.py --is_training
python run_classifier.py --is_training
# eval
python run_pretraining.py
python run_classifier.py
```
