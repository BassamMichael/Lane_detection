import cv2
import numpy as np
from .mask_utils import extract_mask, masks_to_polygons, generate_color, blend_mask_onto_image

def draw_segmentation(image, detections, alpha=0.5):
    """
    Draw segmentation masks, contours, bounding boxes, labels, and confidence on the image.
    """
    if isinstance(image, np.ndarray):
        annotated_image = image.copy()
    else:
        annotated_image = np.array(image).copy()
        
    target_shape = annotated_image.shape[:2]
    
    # Adaptive metrics based on image size
    height, width = target_shape
    thickness = max(1, int(min(height, width) / 400))
    font_scale = max(0.4, min(height, width) / 1000)
    
    for det in detections:
        box = det["box"]
        mask = det["mask"]
        class_id = det["class_id"]
        class_name = det["class_name"]
        conf = det["confidence"]
        
        color = generate_color(class_id)
        
        # 1. Extract and resize mask
        mask_binary = extract_mask(mask, target_shape)
        
        # 2. Alpha blending
        annotated_image = blend_mask_onto_image(annotated_image, mask_binary, color, alpha)
        
        # 3. Contour extraction and converting to polygons
        polygons = masks_to_polygons(mask_binary)
        
        # Draw contours (polygons)
        cv2.polylines(annotated_image, polygons, isClosed=True, color=color, thickness=thickness)
        
        # 4. Draw bounding box
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, thickness)
        
        # 5. Draw label + confidence
        label = f"{class_name} {conf:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        # Background rectangle for text
        cv2.rectangle(annotated_image, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
        
        # Text color (white or black depending on background color brightness)
        brightness = (color[0]*299 + color[1]*587 + color[2]*114) / 1000
        text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
        
        cv2.putText(annotated_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
        
    return annotated_image
