# RGBT-Fusion-for-SAR

#### Prepare the conda environment:

```bash
conda env create -f environment.yml
conda activate sarfusion
```

#### Download and prepare the WiSARD dataset:

https://drive.google.com/file/d/1PKjGCqUszHH1nMbXUBTwPSDqRabAt_ht

##### Extract images from the WiSARD dataset and prepare it:

```bash
unzip dataset/WiSARDv1.zip -d dataset/WiSARD
python3 main.py preprocess_wisard
```
ya29.a0AXooCgt_F9qsoEu2V4lGqvH6BavWwHbLbTok6Wq3psJmjEhe6YLEeGNREAWqX6ao-3eJ7AAI4L2YiJsiW1Lvh81xYeYK8340c6AgiDGHlNjw1KY-Ke4HiExt1w2WkGXd4CWikd1sXcQ4x_lbSRR9JnGDVfwJfN3OSuB9aCgYKAfcSARESFQHGX2MiUD86xj-ikWK-m0pXoXJYcQ0171

curl -H "Authorization: Bearer ya29.a0AXooCgt_F9qsoEu2V4lGqvH6BavWwHbLbTok6Wq3psJmjEhe6YLEeGNREAWqX6ao-3eJ7AAI4L2YiJsiW1Lvh81xYeYK8340c6AgiDGHlNjw1KY-Ke4HiExt1w2WkGXd4CWikd1sXcQ4x_lbSRR9JnGDVfwJfN3OSuB9aCgYKAfcSARESFQHGX2MiUD86xj-ikWK-m0pXoXJYcQ0171" https://www.googleapis.com/drive/v3/files/1PKjGCqUszHH1nMbXUBTwPSDqRabAt_ht?alt=media -o dataset/WiSARDv1.zip 


#### Extract classification patches from the SARDATA dataset:

```bash
python3 main.py preprocess_classification
```

#### Train the pose classifier
    
```bash
python3 main.py experiment --parameters="parameters/SARDPose.yaml"
```

#### Preprocess the WiSARD dataset

```bash
python3 main.py preprocess_wisard
```

#### Annotate the WiSARD dataset with the pose classifier
Move the pose classifier checkpoint to the `checkpoints` folder and run the following command:

```bash
python3 main.py annotate_wisard
```
