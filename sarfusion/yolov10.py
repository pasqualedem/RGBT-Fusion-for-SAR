from ultralytics import YOLOv10

model = YOLOv10.from_pretrained('jameslahm/yolov10n')

model.train(data='coco.yaml', epochs=500, batch=4, imgsz=640)