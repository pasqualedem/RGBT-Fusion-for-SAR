import base64
from http import HTTPStatus
from tempfile import NamedTemporaryFile
from types import GeneratorType
import cv2
from fastapi import FastAPI, File, Header, Request, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from matplotlib import pyplot as plt
from pydantic import BaseModel
from typing import List, Optional, Union
from PIL import Image
import io
import torch
import torchvision.transforms as transforms

from sarfusion.data.wisard import collate_rgb_ir
from sarfusion.models import YOLOv10WiSARD


description = """SAR-Fusion API allows you to deal with wilderness images and videos.

## Users

You will be able to:
* **Generate bounding boxes from images and videos** 
* **Compute how many people are present in the photo or video frames** 

"""

image_mime_types = ["image/jpeg", "image/png", "image/gif", "image/bmp", "image/webp"]
video_mime_types = [
    "video/mp4",
    "video/mpeg",
    "video/ogg",
    "video/webm",
    "video/quicktime",
]


def build_model():
    try:
        model = YOLOv10WiSARD.from_pretrained("pasqualedem/YOLOv10fusion-WiSARD")
    except Exception as e:
        print("Error loading model from Hugging Face, trying to load from local checkpoint")
        checkpoint_path = "checkpoints/yolo-fusion-v10-s.pt"
        model = YOLOv10WiSARD(model=checkpoint_path, task="detect")
    print("Model loaded successfully")
    model.model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.model.to(device)
    return model


def yolo_preprocess(tensor):
    global imgsz
    h0, w0 = tensor.shape[:2]  # orig hw
    if not (h0 == w0 == imgsz):  # resize by stretching image to square imgsz
        array = tensor.permute(1, 2, 0).numpy()
        tensor = cv2.resize(array, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(tensor).permute(2, 0, 1)
    return tensor, (h0, w0), tensor.shape[:2]


def get_bytes_img(array):
    bytes_pred = io.BytesIO()
    plt.imsave(bytes_pred, array, format="png")
    bytes_pred.seek(0)
    return bytes_pred


def make_video(frames, output_path):
    fps = 30
    frame_width, frame_height = frames[0].shape[1], frames[0].shape[0]

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID') # or use 'X264' or 'MJPG'
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Write frames to the video file
    for frame in frames:
        out.write(frame)

    # Release the VideoWriter object
    out.release()


app = FastAPI(
    title="SAR Fusion-API 👥🚁",
    description=description,
    version="1.0.0",
)

# Instantiate the model
model = build_model()
imgsz = model.model.imgsz


class BoundingBox(BaseModel):
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    confidence: float
    class_id: int


@app.get("/", tags=["General"])  # path operation decorator
def _index(request: Request):
    """Root endpoint."""

    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"message": "Welcome to SAR-Fusion API! Please, read the `/docs`!"},
    }
    return response


def fusion_prepare(rgb_file, infrared_file):
    rgb_img = Image.open(rgb_file.file)
    ir_img = Image.open(infrared_file.file)

    rgb_tensor = transforms.ToTensor()(rgb_img)
    ir_tensor = transforms.ToTensor()(ir_img)

    input = collate_rgb_ir(rgb_tensor, ir_tensor)
    input, _, _ = yolo_preprocess(input)
    return input


def get_input(rgb_file, infrared_file):
    input = []
    if rgb_file is not None and infrared_file is not None:
        input = fusion_prepare(rgb_file, infrared_file)
    elif rgb_file is not None:
        suffix = rgb_file.filename.split(".")[-1]
        temp_file = NamedTemporaryFile(delete=False, suffix=f".{suffix}")
        temp_file.write(rgb_file.file.read())
        input = temp_file.name
    elif infrared_file is not None:
        ir_img = Image.open(infrared_file.file)
        ir_tensor = transforms.ToTensor()(ir_img)
        ir_tensor, _, _ = yolo_preprocess(ir_tensor)
        if ir_tensor.shape[0] == 3:
            ir_tensor = ir_tensor[:1]
        input = ir_tensor
    return input


def turn_orig_img_to_rgb(predictions):
    if len(predictions.orig_img.shape) == 2 or predictions.orig_img.shape[2] == 1:
        # Repeat the grayscale channel to have 3 channels
        predictions.orig_img = predictions.orig_img.repeat(3, 2)
    elif predictions.orig_img.shape[2] == 4:
        # Remove the infrared channel
        predictions.orig_img = predictions.orig_img[:, :, :3]
    return predictions

def video_frame_iterator(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError("Error opening video file")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Normalize the frame between 0 and 1
        ir_tensor = (torch.tensor(frame) / 255.0).permute(2, 0, 1)
        
        # Apply the yolo_preprocess function
        ir_tensor, _, _ = yolo_preprocess(ir_tensor)
        
        # Check the condition on the shape of the tensor
        if ir_tensor.shape[0] == 3:
            ir_tensor = ir_tensor[:1]
        
        yield ir_tensor
    
    cap.release()
    
    
def fusion_video_prepare_iterator(rgb_file, infrared_file):
    rgb_cap = cv2.VideoCapture(rgb_file)
    ir_cap = cv2.VideoCapture(infrared_file)
    
    if not rgb_cap.isOpened() or not ir_cap.isOpened():
        raise ValueError("Error opening video file")
    
    while True:
        ret_rgb, frame_rgb = rgb_cap.read()
        ret_ir, frame_ir = ir_cap.read()
        
        if not ret_rgb or not ret_ir:
            break
        
        rgb_tensor = (torch.tensor(frame_rgb) / 255.0).permute(2, 0, 1)
        ir_tensor = (torch.tensor(frame_ir) / 255.0).permute(2, 0, 1)
        
        rgb_tensor, _, _ = yolo_preprocess(rgb_tensor)
        ir_tensor, _, _ = yolo_preprocess(ir_tensor)
        
        input = collate_rgb_ir(rgb_tensor, ir_tensor)
        yield input
    
    rgb_cap.release()
    ir_cap.release()
    
    
def get_temp_video_file(video_file):
    suffix = video_file.filename.split(".")[-1]
    temp_file = NamedTemporaryFile(delete=False, suffix=f".{suffix}")
    temp_file.write(video_file.file.read())
    return temp_file.name


def get_video_input(rgb_file, infrared_file):
    input = [] 
    if rgb_file is not None and infrared_file is not None:
        rgb_file = get_temp_video_file(rgb_file)
        infrared_file = get_temp_video_file(infrared_file)
        input = fusion_video_prepare_iterator(rgb_file, infrared_file)
    elif rgb_file is not None:
        input = get_temp_video_file(rgb_file)
    elif infrared_file is not None:
        filename = get_temp_video_file(infrared_file)
        return video_frame_iterator(filename)
    return input


@app.post(
    "/predict",
    summary="Predict bounding boxes from RGB and Infrared images",
)
async def predict(
    rgb_file: Optional[UploadFile] = File(None),
    infrared_file: Optional[UploadFile] = File(None),
    return_plots: Optional[bool] = Header(False),
):
    if rgb_file is None and infrared_file is None:
        raise HTTPException(status_code=400, detail="At least one image is required.")

    predictions = []
    input = get_input(rgb_file, infrared_file)

    predictions = model.predict(input)[0]
    boxes = predictions.boxes.data
    predictions = turn_orig_img_to_rgb(predictions)

    # Format the predictions for the response
    bounding_boxes = [
        BoundingBox(
            xmin=box[0],
            ymin=box[1],
            xmax=box[2],
            ymax=box[3],
            confidence=box[4],
            class_id=int(box[5]),
        )
        for box in boxes
    ]

    if return_plots:
        return StreamingResponse(
            get_bytes_img(predictions.plot()), media_type="image/png"
        )

    return bounding_boxes


@app.post("/predict/video", summary="Predict bounding boxes from a video")
def predict_video(
    rgb_video_file: Optional[UploadFile] = File(None),
    infrared_video_file: Optional[UploadFile] = File(None),
    return_plots: Optional[bool] = Header(False),
):
    if rgb_video_file is None and infrared_video_file is None:
        raise HTTPException(status_code=400, detail="At least one image is required.")

    predictions = []
    input = get_video_input(rgb_video_file, infrared_video_file)

    # If it is a generator
    if isinstance(input, GeneratorType):
        result = []
        for frame in input:
            predictions = model.predict(frame)
            result.append(predictions)
        # Linearize list of lists
        predictions = [
            item for sublist in result for item in sublist
        ]
    else:
        predictions = model.predict(input, stream=True)

    if return_plots:
        # Convert the predictions to RGB if necessary
        predictions = [turn_orig_img_to_rgb(prediction) for prediction in predictions]
        # Get plots
        frames = [prediction.plot() for prediction in predictions]
        video_path = NamedTemporaryFile(delete=False, suffix=".mp4").name
        make_video(frames, video_path)
        return FileResponse(video_path, media_type="video/mp4", filename="predictions.mp4")
 
    return [
        [
            BoundingBox(
                xmin=box[0],
                ymin=box[1],
                xmax=box[2],
                ymax=box[3],
                confidence=box[4],
                class_id=int(box[5]),
            )
            for box in frame.boxes.data
        ]
        for frame in predictions
    ]


def run_app():
    import uvicorn

    uvicorn.run(
        "sarfusion.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["sarfusion"],
    )
