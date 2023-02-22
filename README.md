# BERT-Masked-Language-Modeling-tf2-keras
original (tensorflow 1.X version): https://github.com/google-research/bert

### Required
This code is tested with python==3.8, tensorflow-gpu==2.7, and cuda/cudnn 11.2/8.1
- Python 3.X
- Tensorflow 2.X
- CUDA 10.X or 11.X for gpu settings depends on your hardware

### How to use
Upload your own datasets in `datasets` folder and load them inside `run_pretraining.py`. Refer to the annotations.
```
python run_training.py --is_training
```

