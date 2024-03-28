# RGBT-Fusion-for-SAR

#### Prepare the conda environment:

```bash
conda env create -f environment.yml
```

#### Download and prepare the WiSARD dataset:

https://drive.google.com/file/d/1PKjGCqUszHH1nMbXUBTwPSDqRabAt_ht


#### Extract classification patches from the SARDATA dataset:

```bash
python3 main.py preprocess_classification
```