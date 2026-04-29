import os
import torch
from ultralytics import YOLO

class SegmentationModel:
    def __init__(self, model_path="model/best.pt", labels_path="model/labels.txt"):
        self.model_path = model_path
        self.labels_path = labels_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.model = self.load_model()
        self.class_names = self.load_labels()

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        try:
            model = YOLO(self.model_path)
            model.to(self.device)
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}")

    def load_labels(self):
        if not os.path.exists(self.labels_path):
            print(f"Warning: Labels file not found at {self.labels_path}. Using model's default names.")
            return self.model.names if hasattr(self.model, 'names') else {}
        
        try:
            with open(self.labels_path, 'r') as f:
                labels = {i: line.strip() for i, line in enumerate(f.readlines()) if line.strip()}
            return labels
        except Exception as e:
            print(f"Warning: Failed to read labels file: {e}")
            return self.model.names if hasattr(self.model, 'names') else {}

    def predict(self, image, conf_threshold=0.25):
        if image is None:
            raise ValueError("Input image cannot be None")
            
        results = self.model.predict(source=image, conf=conf_threshold, device=self.device, verbose=False)
        
        detections = []
        if not results or len(results) == 0:
            return detections
            
        result = results[0]
        
        if result.boxes is None or result.masks is None:
            return detections
            
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        masks = result.masks.data.cpu().numpy()  # Extract the masks as tensors
        
        for i in range(len(boxes)):
            class_id = class_ids[i]
            class_name = self.class_names.get(class_id, self.model.names.get(class_id, f"Class {class_id}"))
            
            mask = masks[i]
            
            detections.append({
                "box": boxes[i].tolist(),
                "mask": mask,
                "class_id": int(class_id),
                "class_name": class_name,
                "confidence": float(confs[i])
            })
            
        return detections
