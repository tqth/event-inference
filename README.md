# 📷 Event Inference from Photo Albums

This repository provides a pipeline for **event recognition from photo albums**. Given an album (a folder of images), it infers the type of event (e.g., *Wedding*, *Graduation*, *Birthday*) and highlights the **most important images** using an attention mechanism.

> 🧠 **The focus of this project is on inference**, using a pretrained attention-based deep learning model and CLIP + RAM features.

---

## ✨ What It Does

Given a folder of images (an album), this system:

1. Extracts visual features from [CLIP](https://github.com/openai/CLIP)
2. Extracts semantic tags using [RAM (Recognize Anything Model)](https://github.com/zhang-tao-whu/Recognize-Anything)
3. Combines both features and passes them into a pretrained attention model
4. Outputs:
   - **Predicted event label(s)** (multi-label)
   - **Top-N important images** based on learned attention scores

---

## 🔧 Setup

### 1. Clone the repository

```bash
git clone https://github.com/tqth/event-inference.git
cd event-inference
```
---
### 2. Install dependencies
```bash
pip install -r requirements.txt
```

---

---
## ⚠️ Important Note: Run on CPU only

This project and its dependencies are currently tested and intended to run only on CPU environments.
Running on GPU may cause unexpected errors or unstable behavior due to compatibility issues with some libraries.

---

## 🎯 Example Usage

This project includes a Jupyter Notebook (`inference_event_demo.ipynb`) that demonstrates a full inference pipeline from loading an album folder to predicting event labels and extracting important images.

---


## 🚀 Run Inference on an Album
🔍 Step-by-step
```python
from predict_event import predict_event_and_top_n_images_from_folder
album_path = "/path/example_album"
event, key_images = predict_event_and_top_n_images_from_folder(album_path, n_key_images= 5)
print("Prediction:", event)
print("Key images:", key_images)
```

🖼️ Example Output

Predict: ['Graduation']

Top 3 important images:
 - 4785949094.jpg
 - 4785955252.jpg
 - 4785314023.jpg


