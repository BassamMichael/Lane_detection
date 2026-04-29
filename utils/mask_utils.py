import cv2
import numpy as np

def extract_mask(mask, target_shape):
    """
    Extracts and resizes the YOLO mask output to the target image shape.
    target_shape: (height, width)
    """
    height, width = target_shape[:2]
    # Resize expects (width, height)
    mask_resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
    # Binarize
    mask_binary = (mask_resized > 0.5).astype(np.uint8)
    return mask_binary

def extract_contours(mask_binary):
    """
    Extract contours from a binary mask.
    """
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def masks_to_polygons(mask_binary):
    """
    Convert binary masks to polygons using extracted contours.
    """
    contours = extract_contours(mask_binary)
    polygons = []
    for contour in contours:
        if len(contour) >= 3:  # Valid polygon needs at least 3 points
            polygons.append(contour.reshape(-1, 2))
    return polygons

def generate_color(class_id):
    """
    Assign an adaptive color dynamically based on class ID.
    """
    np.random.seed(class_id * 100)
    color = np.random.randint(0, 255, size=(3,), dtype=np.uint8)
    return tuple(map(int, color))

def blend_mask_onto_image(image, mask_binary, color, alpha=0.5):
    """
    Alpha blend the binary mask onto the original image.
    """
    colored_mask = np.zeros_like(image, dtype=np.uint8)
    colored_mask[mask_binary == 1] = color
    
    blended = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
    
    # Only keep blended pixels where mask is 1
    result = np.where(mask_binary[..., None] == 1, blended, image)
    return result
