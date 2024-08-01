# RGBT-Fusion-for-SAR

## Run the FastAPI server

### With Docker

```sh
docker build -t sarfusion .
docker run -it --rm -p 8000:8000 sarfusion
```

### Without Docker
After having installed the environment
```sh
python main.py app
```

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

#### Download the YOLOv9 pre-trained weights:

```bash
wget https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c-converted.pt -O checkpoints/yolov9-c-converted.pt
```