import streamlit as st
import torch
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
import pickle
import pandas as pd
import os

# ==== CONFIG ====
MODEL_PATH = "models/best_artist_classifier_cnn.pth"
ENCODER_PATH = "models/label_encoder.pkl"
ARTIST_CSV_PATH = "data/mapped.csv"
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== LOAD MODEL ====
@st.cache_resource(show_spinner=False)
def load_model():
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, 16)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# ==== LOAD ENCODER ====
@st.cache_resource(show_spinner=False)
def load_label_encoder():
    with open(ENCODER_PATH, 'rb') as f:
        return pickle.load(f)

# ==== LOAD ARTIST DATA ====
@st.cache_data(show_spinner=False)
def load_artist_data():
    df = pd.read_csv(ARTIST_CSV_PATH)
    df.columns = df.columns.str.strip().str.lower()
    return df

# ==== TRANSFORM ====
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_artist(image: Image.Image, model, label_encoder, artist_df):
    img_t = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(img_t)
        _, predicted = torch.max(output, 1)
        predicted_label = label_encoder.inverse_transform(predicted.cpu().numpy())[0]

    predicted_label_clean = predicted_label.strip().lower()
    artist_row = artist_df[artist_df['artist'].str.strip().str.lower() == predicted_label_clean]

    info = {}
    if not artist_row.empty:
        info = {
            "Born–Died": artist_row.iloc[0].get('born-died', 'N/A'),
            "Period": artist_row.iloc[0].get('period', 'N/A'),
            "Nationality": artist_row.iloc[0].get('nationality', 'N/A')
        }
    return predicted_label, info

# ==== SAMPLE PICKER ====
SAMPLE_DIR = "images/sample"

def get_sample_files():
    if not os.path.isdir(SAMPLE_DIR):
        return []
    return sorted(f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png")))

# ==== STREAMLIT UI ====
st.title("Artist Classifier 🎨")
st.write("Upload a painting image and get the predicted artist along with artist details.")

if "mode" not in st.session_state:
    st.session_state.mode = None
if "last_upload_key" not in st.session_state:
    st.session_state.last_upload_key = None

sample_files = get_sample_files()
if sample_files:
    st.write("**Try a sample:**")
    cols = st.columns(len(sample_files))
    for col, fname in zip(cols, sample_files):
        with col:
            st.image(os.path.join(SAMPLE_DIR, fname), use_container_width=True)
            if st.button("Use this", key=f"sample_{fname}"):
                st.session_state.mode = "sample"
                st.session_state.selected_sample = fname

st.write("**Or upload your own:**")
uploaded_file = st.file_uploader("Drag and drop an image here", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    upload_key = (uploaded_file.name, uploaded_file.size)
    if upload_key != st.session_state.last_upload_key:
        st.session_state.mode = "upload"
        st.session_state.last_upload_key = upload_key
elif st.session_state.mode == "upload":
    st.session_state.mode = None
    st.session_state.last_upload_key = None

image = None
if st.session_state.mode == "upload" and uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
elif st.session_state.mode == "sample" and st.session_state.get("selected_sample"):
    image = Image.open(os.path.join(SAMPLE_DIR, st.session_state.selected_sample)).convert("RGB")

if image is not None:
    try:
        st.image(image, caption="Selected Image", use_container_width=True)

        model = load_model()
        label_encoder = load_label_encoder()
        artist_df = load_artist_data()

        predicted_artist, artist_info = predict_artist(image, model, label_encoder, artist_df)

        st.markdown(f"### Predicted Artist: {predicted_artist}")

        if artist_info:
            st.markdown("#### Artist Details:")
            st.write(f"**Born–Died:** {artist_info.get('Born–Died', 'N/A')}")
            st.write(f"**Period:** {artist_info.get('Period', 'N/A')}")
            st.write(f"**Nationality:** {artist_info.get('Nationality', 'N/A')}")
        else:
            st.write("No additional artist details found.")
    except Exception as e:
        st.error(f"Error processing image: {e}")
