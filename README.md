# YOLO Instance Segmentation Streamlit App

A modular, production-ready scaffold for deploying YOLO Instance Segmentation models using Streamlit Cloud.

## 📁 Folder Structure

```
yolo-seg-deployment/
├── app.py                 # Local testing script
├── streamlit_app.py       # Main Streamlit web application
├── requirements.txt       # Python dependencies
├── packages.txt           # System-level dependencies for Streamlit Cloud
├── README.md              # Project documentation
├── model/                 
│   ├── best.pt            # Your trained YOLO segmentation model (Place here)
│   └── labels.txt         # Class labels, one per line (Place here)
├── utils/                 # Utility modules for inference and visualization
│   ├── detector.py        # YOLO model wrapper and prediction logic
│   ├── visualization.py   # Drawing bounding boxes, masks, and labels
│   └── mask_utils.py      # Mask extraction, blending, and contour utilities
└── assets/                
    └── demo.png           # Demo image for the Streamlit app
```

## 🚀 Local Execution

1. **Install Dependencies:**
   Make sure you have Python 3.8+ installed.
   ```bash
   pip install -r requirements.txt
   ```

2. **Place your Model and Labels:**
   - Drop your trained `best.pt` into the `model/` folder.
   - Edit `model/labels.txt` to match your classes (one per line).
   - Place a sample image as `assets/demo.png`.

3. **Test Locally:**
   Run the CLI testing script to verify the model and dependencies work.
   ```bash
   python app.py
   ```
   Check the generated `output_segmented.jpg`.

4. **Run Streamlit Locally:**
   ```bash
   streamlit run streamlit_app.py
   ```

## ☁️ Streamlit Cloud Deployment Guide

1. **Initialize Git and Push to GitHub:**
   Run the following commands in your terminal to push this project to GitHub.
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io/).
   - Click **New app**.
   - Select the GitHub repository you just created.
   - Set the **Main file path** to: `streamlit_app.py`.
   - Click **Deploy!**

## 🛠️ Common Troubleshooting

*   **`ModuleNotFoundError`:** Ensure all required Python packages are listed in `requirements.txt`. The provided file covers the essentials (`ultralytics`, `streamlit`, `opencv-python-headless`, `Pillow`, `numpy`).
*   **`libGL.so.1: cannot open shared object file`:** This is a common issue with OpenCV on cloud environments. It is solved by including `libgl1` and `libglib2.0-0` in `packages.txt`, which tells Streamlit Cloud to install these Linux system dependencies. Alternatively, using `opencv-python-headless` in `requirements.txt` avoids this entirely, which we have done as a double safety measure.
*   **Missing Model/Labels:** Ensure `best.pt` and `labels.txt` are pushed to GitHub if you want them deployed. If they are too large, consider downloading them dynamically or using Git LFS.
