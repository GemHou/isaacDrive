# Install
```bash
conda create -n isaacDrive python==3.8 -y
```

```bash
conda activate isaacDrive
```

GPU:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

CPU:
```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```



```bash
pip install matplotlib
pip install tqdm
```

# Train
```bash
python ./main_train_grad.py
```

# Test and Render

```bash
python ./main_test.py
```