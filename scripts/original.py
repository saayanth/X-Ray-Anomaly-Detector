# app_pytorch_densenet_gradcam.py
import streamlit as st
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import os
import importlib.util

# -------------------------
# CONFIG
# -------------------------
MODEL_PATH = r"C:\Users\PC\Downloads\new model\m-25012018-123527.pth"
IMG_SIZE = 224
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

# -------------------------
# Ensure densenet module is importable
# -------------------------
# If densenet.py is in the same folder as script, this will import it.
# Otherwise set DN_PATH to the folder containing densenet.py
DN_PATH = r"C:\Users\PC\xray_project\scripts"  # <-- change if needed
if DN_PATH not in os.sys.path:
    os.sys.path.append(DN_PATH)
try:
    import densenet as dn
except Exception as e:
    st.error(f"Could not import densenet module from {DN_PATH}: {e}")
    st.stop()

# -------------------------
# Helper: build the same DenseNet3 model used during training
# Defaults are taken from your training script:
# dn.DenseNet3(layers, num_classes, growth, reduction, bottleneck, dropRate)
# -------------------------
def build_model(num_classes):
    # Defaults from your training script; change if needed
    layers = 100
    growth = 12
    reduction = 0.5
    bottleneck = True
    dropRate = 0.0
    model = dn.DenseNet3(layers, num_classes, growth, reduction=reduction,
                         bottleneck=bottleneck, dropRate=dropRate)
    return model

# -------------------------
# Load checkpoint (handles both raw state_dict and saved dicts)
# -------------------------
@st.cache_resource
def load_checkpoint(path):
    device = torch.device("cpu")
    checkpoint = torch.load(path, map_location=device)
    # checkpoint may be dict with 'state_dict' or be a raw state_dict
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    return state_dict

# -------------------------
# Prepare model and load weights
# -------------------------
st.write("Loading model...")
try:
    # Build model with number of classes equal to length of CLASS_NAMES
    model = build_model(num_classes=len(CLASS_NAMES))
    state_dict = load_checkpoint(MODEL_PATH)

    # If state_dict keys are prefixed (e.g., module.), try to strip that
    def clean_state_dict(sd):
        new_sd = {}
        for k, v in sd.items():
            new_k = k
            if k.startswith("module."):
                new_k = k[len("module."):]
            new_sd[new_k] = v
        return new_sd

    state_dict = clean_state_dict(state_dict)
    model.load_state_dict(state_dict)
    model.eval()
    st.success("Model loaded.")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# -------------------------
# Find last conv layer automatically
# -------------------------
def find_last_conv(module):
    last = None
    for name, m in module.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            last = (name, m)
    return last

last_conv = find_last_conv(model)
if last_conv is None:
    st.error("Could not find a Conv2d layer in model to use for Grad-CAM.")
    st.stop()
last_conv_name, last_conv_layer = last_conv
st.write(f"Using last conv layer: `{last_conv_name}` for Grad-CAM")

# -------------------------
# Grad-CAM implementation (PyTorch hooks)
# -------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activation = None
        self.gradient = None
        # Register hooks
        def forward_hook(module, input, output):
            self.activation = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradient = grad_out[0].detach()
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        # forward
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        score = logits[:, class_idx]
        # backward
        self.model.zero_grad()
        score.backward(retain_graph=True)
        # get pooled gradients
        pooled_gradients = torch.mean(self.gradient, dim=[0, 2, 3])  # C
        activations = self.activation[0]  # C x H x W
        # weight channels by averaged gradients
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]
        heatmap = torch.sum(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() != 0:
            heatmap = heatmap / heatmap.max()
        heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
        return heatmap, class_idx

# -------------------------
# Preprocess transforms (match what you used during training)
# -------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                         std=[x/255.0 for x in [63.0, 62.1, 66.7]])
])

# -------------------------
# Streamlit UI
# -------------------------
st.title("PyTorch DenseNet (.pth) — Prediction + Grad-CAM")
uploaded = st.file_uploader("Upload chest X-ray (png/jpg)", type=["png","jpg","jpeg"])

if uploaded is not None:
    pil = Image.open(uploaded).convert("RGB")
    st.image(pil, caption="Uploaded image", use_column_width=True)

    # preprocess and run
    input_tensor = transform(pil).unsqueeze(0)  # 1 x C x H x W
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        pred_name = CLASS_NAMES[pred_idx]
        conf = probs[pred_idx]

    st.markdown(f"**Prediction:** {pred_name} — **{conf*100:.2f}%**")

    # Grad-CAM (no_grad can't be used because we need gradients)
    cam = GradCAM(model, last_conv_layer)
    heatmap, hm_class = cam.generate(input_tensor, class_idx=pred_idx)

    # Overlay heatmap on image
    img_np = np.array(pil.resize((IMG_SIZE, IMG_SIZE)))
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_np, 0.6, heatmap_color, 0.4, 0)

    st.image(overlay, caption="Grad-CAM overlay", use_column_width=True)

st.markdown("---")
st.caption("Make sure CLASS_NAMES matches your trained labels. Edit DN_PATH if densenet.py is in another folder.")
