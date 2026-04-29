import os
import cv2
import time
from PIL import Image
import numpy as np

from utils.detector import SegmentationModel
from utils.visualization import draw_segmentation

def main():
    print("--- YOLO Instance Segmentation Local Testing ---")
    
    # Ensure paths
    model_path = "model/best.pt"
    labels_path = "model/labels.txt"
    demo_image_path = "assets/demo.png"
    output_path = "output_segmented.jpg"
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please place your best.pt file there.")
        return
        
    # Load model
    print("Loading model...")
    try:
        detector = SegmentationModel(model_path=model_path, labels_path=labels_path)
        print(f"Model loaded successfully on device: {detector.device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    # Load image
    if not os.path.exists(demo_image_path):
        print(f"Error: Demo image not found at {demo_image_path}. Please place an image there.")
        return
        
    print(f"Loading image from {demo_image_path}...")
    try:
        image = Image.open(demo_image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        return
        
    # Run inference
    print("Running inference...")
    start_time = time.time()
    try:
        detections = detector.predict(image, conf_threshold=0.25)
    except Exception as e:
        print(f"Error during inference: {e}")
        return
    end_time = time.time()
    print(f"Inference complete in {end_time - start_time:.3f} seconds.")
    
    # Print detections
    print(f"\nFound {len(detections)} objects:")
    for i, det in enumerate(detections):
        print(f"  {i+1}. {det['class_name']} (Conf: {det['confidence']:.2f})")
        
    # Visualize
    image_np = np.array(image)
    annotated_image = draw_segmentation(image_np, detections, alpha=0.5)
    
    # Save output (OpenCV uses BGR for saving, so convert RGB to BGR)
    cv2.imwrite(output_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    print(f"\nSaved segmented image to {output_path}")

if __name__ == "__main__":
    main()
