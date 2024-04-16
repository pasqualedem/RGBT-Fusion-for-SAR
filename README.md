# RGBT-Fusion-for-SAR

#### Prepare the conda environment:

```bash
conda env create -f environment.yml
```

#### Download and prepare the WiSARD dataset:

https://drive.google.com/file/d/1PKjGCqUszHH1nMbXUBTwPSDqRabAt_ht

##### Extract images from the WiSARD dataset and prepare it:

```bash
unzip WiSARDv1.zip -d dataset/WiSARD
python3 main.py preprocess_wisard
```


#### Extract classification patches from the SARDATA dataset:

```bash
python3 main.py preprocess_classification
```

#### Train the pose classifier
    
```bash
python3 main.py experiment --parameters="parameters/SARDPose.yaml"
```

