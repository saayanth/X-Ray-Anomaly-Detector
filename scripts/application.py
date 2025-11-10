import streamlit as st
from tensorflow import keras
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

CLASS_NAMES = [
    "Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass",
    "Nodule","Pneumonia","Pneumothorax","Consolidation","Edema",
    "Emphysema","Fibrosis","Pleural_Thickening","Hernia","No Finding"
]

DISEASE_INFO = {
    "Atelectasis": "Partial lung collapse causing reduced oxygen exchange.",
    "Cardiomegaly": "Enlarged heart due to hypertension or heart disease.",
    "Effusion": "Fluid buildup around the lungs.",
    "Infiltration": "Fluid or cells entering lung tissue.",
    "Mass": "Abnormal lung tissue growth.",
    "Nodule": "Small round growth; mostly benign.",
    "Pneumonia": "Infection inflaming air sacs in the lungs.",
    "Pneumothorax": "Air leaks into pleural space causing lung collapse.",
    "Consolidation": "Lung tissue filled with fluid or pus.",
    "Edema": "Fluid accumulation in lungs due to heart failure.",
    "Emphysema": "Chronic condition damaging lung air sacs.",
    "Fibrosis": "Scarring or thickening of lung tissue.",
    "Pleural_Thickening": "Thickening of pleural lining.",
    "Hernia": "Protrusion of an organ through the diaphragm.",
    "No Finding": "No abnormality detected in this X-ray image."
}

model_path = r"C:\Users\PC\xray_project\models\xray_densenet_final.h5"
model = keras.models.load_model(model_path, compile=False)

st.title("🩻Chest X-ray Anomaly Detector")
st.caption("Upload chest x-rays to find possible abnormalities")

uploaded_file = st.file_uploader("Upload Chest X-ray", type=["jpg", "jpeg", "png"])

def preprocess(img):
    """Resize and normalize image for model"""
    img = img.resize((224,224))
    array = keras.utils.img_to_array(img)
    array = np.expand_dims(array, axis=0)/255.0
    return array

def get_gradcam(model, img_array, class_idx):
    """Generate Grad-CAM heatmap for given class index"""
    last_conv_layer = None
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:
            last_conv_layer = layer.name
            break

    grad_model = keras.models.Model([model.inputs],
                                   [model.get_layer(last_conv_layer).output, model.output])
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_idx]
    
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap.numpy(),0)
    heatmap /= np.max(heatmap)+1e-8
    heatmap = cv2.resize(heatmap,(224,224))
    return heatmap

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img_array = preprocess(img)

    preds = model.predict(img_array)[0]
    top_idx = np.argmax(preds)
    pred_class = CLASS_NAMES[top_idx]
    confidence = preds[top_idx]
    explanation = DISEASE_INFO[pred_class]

    st.markdown(f"### 🧠 Prediction: **{pred_class}** ({confidence*100:.2f}%)")
    st.markdown(f"**Explanation:** {explanation}")

    heatmap = get_gradcam(model, img_array, top_idx)
    img_cv = np.array(img.resize((224,224)))
    heatmap_color = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_cv,0.6,heatmap_color,0.4,0)

    st.image(img, caption="Original X-ray", use_container_width=True)
    st.image(overlay, caption="Grad-CAM Overlay", use_container_width=True)
