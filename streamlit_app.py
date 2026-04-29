import os
import time
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch

# --- PYTORCH 2.6 SECURITY FIX ---
# We import the internal Ultralytics class and register it as safe
from ultralytics.nn.tasks import SegmentationModel as YOLO_Seg_Model
torch.serialization.add_safe_globals([YOLO_Seg_Model])

# Now import your custom local modules
from utils.detector import SegmentationModel
from utils.visualization import draw_segmentation

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="YOLO Instance Segmentation",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CACHE MODEL LOADING ---
@st.cache_resource
def load_model(model_path="model/best.pt", labels_path="model/labels.txt"):
    if not os.path.exists(model_path):
        return None, "Model file not found. Please place 'best.pt' in the 'model/' directory."
    try:
        # This call eventually triggers torch.load, which now has the safe globals registered
        model = SegmentationModel(model_path=model_path, labels_path=labels_path)
        return model, "Success"
    except Exception as e:
        return None, f"Failed to load model: {str(e)}"

# --- MAIN APP ---
def main():
    st.title("🔍 YOLO Instance Segmentation")
    st.markdown("Upload an image or use the demo image to perform instance segmentation.")
    
    # Sidebar controls
    st.sidebar.header("⚙️ Configuration")
    conf_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.01, max_value=1.0, value=0.25, step=0.01)
    mask_alpha = st.sidebar.slider("Mask Opacity", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    
    # Load Model
    with st.spinner("Loading model..."):
        detector, status = load_model()
        
    if detector is None:
        st.error(status)
        st.stop()
        
    st.sidebar.success(f"Model loaded on: **{detector.device}**")
    
    # Image Input
    st.sidebar.header("🖼️ Input Image")
    upload_method = st.sidebar.radio("Choose input method:", ("Upload Image", "Use Demo Image"))
    
    image = None
    if upload_method == "Upload Image":
        uploaded_file = st.sidebar.file_uploader("Upload an image...", type=["jpg", "jpeg", "png", "webp"])
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert("RGB")
            except Exception as e:
                st.error(f"Invalid image file: {e}")
    else:
        demo_path = "assets/demo.png"
        if os.path.exists(demo_path):
            image = Image.open(demo_path).convert("RGB")
        else:
            st.warning(f"Demo image not found at {demo_path}. Please upload an image.")
            
    if image is None:
        st.info("Please provide an image to start.")
        return
        
    # Layout for original and result
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
        
    # Run Inference
    run_btn = st.sidebar.button("🚀 Run Segmentation", type="primary")
    
    if run_btn:
        with st.spinner("Running instance segmentation..."):
            try:
                start_time = time.time()
                detections = detector.predict(image, conf_threshold=conf_threshold)
                inf_time = time.time() - start_time
                
                # Visualize
                image_np = np.array(image)
                annotated_img = draw_segmentation(image_np, detections, alpha=mask_alpha)
                
                with col2:
                    st.subheader("Segmented Image")
                    st.image(annotated_img, use_container_width=True)
                    
                st.success(f"Inference completed in {inf_time:.3f} seconds.")
                
                # Display Summary Table
                st.markdown("### 📊 Detection Summary")
                if detections:
                    summary = []
                    for det in detections:
                        summary.append({
                            "Class": det["class_name"],
                            "Confidence": f"{det['confidence']:.2f}"
                        })
                        
                    df = pd.DataFrame(summary)
                    class_counts = df["Class"].value_counts().reset_index()
                    class_counts.columns = ["Class", "Count"]
                    
                    col_table1, col_table2 = st.columns(2)
                    with col_table1:
                        st.markdown("**All Detections**")
                        st.dataframe(df, use_container_width=True)
                    with col_table2:
                        st.markdown("**Class Counts**")
                        st.dataframe(class_counts, use_container_width=True)
                else:
                    st.info("No objects detected above the confidence threshold.")
                    
            except Exception as e:
                st.error(f"Error during inference: {e}")

if __name__ == "__main__":
    main()
