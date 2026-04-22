import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from PIL import Image

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Plant Disease Detection", layout="wide")

# ---------- CUSTOM UI ----------
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
.title {
    font-size: 40px;
    font-weight: bold;
    color: #2e7d32;
}
.card {
    padding: 20px;
    border-radius: 15px;
    background-color: white;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🌿 Plant Disease Detection System</div>', unsafe_allow_html=True)
st.write("Upload a leaf image to detect disease, severity, and treatment")

# ---------- LANGUAGE ----------
language = st.selectbox("🌍 Language", ["English", "Malayalam"])

# ---------- LOAD MODELS ----------
@st.cache_resource
def load_models():
    yolo_model = YOLO("best.pt")
    cnn_model = load_model("tomato_disease_mobilenetv2.h5")
    return yolo_model, cnn_model

with st.spinner("🚀 Loading AI Models..."):
    yolo_model, cnn_model = load_models()

# ---------- CLASSES ----------
classes = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# ---------- SEVERITY ----------
def get_severity(conf):
    if conf > 0.85:
        return "High 🔴", (0,0,255)
    elif conf > 0.65:
        return "Medium 🟠", (0,165,255)
    else:
        return "Low 🟢", (0,255,0)

# ---------- DATA ----------
desc_en = {
    "Tomato Bacterial spot": "A bacterial disease causing dark lesions and rapid spread in humid conditions.",
    "Tomato Early blight": "A fungal disease with ring patterns affecting older leaves.",
    "Tomato Late blight": "A dangerous disease capable of destroying entire crops quickly.",
    "Tomato Leaf Mold": "Occurs in humid environments with yellow patches.",
    "Tomato Septoria leaf spot": "Small spots leading to leaf drop.",
    "Tomato Spider mites Two-spotted spider mite": "Tiny pests sucking plant sap.",
    "Tomato Target Spot": "Brown ring spots on leaves.",
    "Tomato Tomato Yellow Leaf Curl Virus": "Virus causing curling and yellowing.",
    "Tomato Tomato mosaic virus": "Virus causing mottled leaf patterns.",
    "Tomato healthy": "The plant is healthy."
}

desc_ml = {k: "ഈ രോഗം ഇലകളിൽ ബാധിച്ച് ചെടിയുടെ വളർച്ച കുറയ്ക്കുന്നു." for k in desc_en}

rem_en = {
    "Tomato Bacterial spot": "• Copper spray\n• Remove infected parts\n• Crop rotation\n• Avoid wet leaves",
    "Tomato Early blight": "• Fungicide\n• Remove leaves\n• Air circulation\n• Mulching",
    "Tomato Late blight": "• Remove plants\n• Resistant seeds\n• Fungicide\n• Monitor regularly",
    "Tomato Leaf Mold": "• Reduce humidity\n• Ventilation\n• Fungicide",
    "Tomato Septoria leaf spot": "• Remove leaves\n• Fungicide\n• Clean tools",
    "Tomato Spider mites Two-spotted spider mite": "• Neem oil\n• Soap spray\n• Wash leaves",
    "Tomato Target Spot": "• Fungicide\n• Remove infected parts",
    "Tomato Tomato Yellow Leaf Curl Virus": "• Control insects\n• Resistant seeds",
    "Tomato Tomato mosaic virus": "• Remove plants\n• Disinfect tools",
    "Tomato healthy": "• Maintain watering\n• Sunlight\n• Fertilizer"
}

rem_ml = {k: "• ഫംഗിസൈഡ് ഉപയോഗിക്കുക\n• രോഗബാധിത ഭാഗങ്ങൾ നീക്കം ചെയ്യുക\n• ചെടി സംരക്ഷിക്കുക" for k in rem_en}

# ---------- DETECTION ----------
def detect_and_classify(img):

    results = yolo_model.predict(img, verbose=False)

    for r in results:

        boxes = r.boxes.xyxy.cpu().numpy()
        if len(boxes) == 0:
            return img, "No Leaf Detected", 0, "", "", "", None

        x1, y1, x2, y2 = map(int, boxes[0])

        cropped = img[y1:y2, x1:x2]
        resized = cv2.resize(cropped, (224,224)) / 255.0
        resized = np.expand_dims(resized, axis=0)

        prediction = cnn_model.predict(resized, verbose=0)
        confidence = float(np.max(prediction))

        predicted_class = classes[np.argmax(prediction)]
        predicted_class = predicted_class.replace("___", " ").replace("_", " ")

        severity, color = get_severity(confidence)

        description = desc_ml[predicted_class] if language=="Malayalam" else desc_en[predicted_class]
        remedy = rem_ml[predicted_class] if language=="Malayalam" else rem_en[predicted_class]

        cv2.rectangle(img,(x1,y1),(x2,y2),color,2)

        return img, predicted_class, confidence, severity, description, remedy, prediction


# ---------- UPLOAD ----------
uploaded = st.file_uploader("📤 Upload Leaf Image", type=["jpg","png","jpeg"])

if uploaded:

    image = Image.open(uploaded)
    img = np.array(image)

    result, disease, confidence, severity, description, remedy, prediction = detect_and_classify(img)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original")

    with col2:
        st.image(result, caption="AI Detection")

    # ---------- CONFIDENCE ----------
    st.markdown("### 📊 Confidence")
    st.progress(int(confidence * 100))
    st.write(f"{confidence*100:.2f}%")

    # ---------- MAIN INFO ----------
    st.markdown(f"### 🦠 Disease: {disease}")
    st.markdown(f"### ⚠ Severity: {severity}")

    # ---------- DESCRIPTION ----------
    st.markdown("### 🧾 Description")
    st.info(description)

    # ---------- REMEDY ----------
    st.markdown("### 💊 Remedies")
    st.warning(remedy)

    # ---------- PROBABILITY CHART ----------
    st.markdown("### 📊 All Class Probabilities")

    for i, prob in enumerate(prediction[0]):
        class_name = classes[i].replace("___", " ").replace("_", " ")
        st.progress(float(prob))
        st.write(f"{class_name}: {prob:.3f}")

    # ---------- DOWNLOAD ----------
    report = f"""
Disease: {disease}
Confidence: {confidence}
Severity: {severity}

Description:
{description}

Remedy:
{remedy}
"""
    st.download_button("📄 Download Report", report, file_name="report.txt")