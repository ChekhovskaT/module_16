import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from PIL import Image
import numpy as np

# ======== Завантаження міток класів ========
with open("labels.txt", "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f.readlines()]

# ======== Завантаження моделей ========
cnn_model = load_model("models/cnn_model.h5")
vgg16_model = load_model("models/vgg16_model.h5")

# ======== Отримати форму входу моделі (без None) ========
def get_model_input_size(model):
    input_shape = model.input_shape  # (None, height, width, channels)
    return input_shape[1:4]

# ======== Обробка зображення під модель ========
def preprocess_img(img: Image.Image, input_size, grayscale=False, use_vgg=False):
    if grayscale:
        img = img.convert("L")
        img = img.resize(input_size[:2])
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=-1)
    else:
        img = img.convert("RGB")
        img = img.resize(input_size[:2])
        img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    if use_vgg:
        img_array = vgg_preprocess(img_array)
    else:
        img_array = img_array / 255.0

    return img_array

# ======== Класифікація ========
def predict_image(img, model, use_vgg=False, grayscale=False):
    input_size = get_model_input_size(model)
    preprocessed = preprocess_img(img, input_size, grayscale, use_vgg)
    predictions = model.predict(preprocessed)[0]
    predicted_index = np.argmax(predictions)
    predicted_class = labels[predicted_index]
    return predicted_class, predictions

# ======== Streamlit інтерфейс ========
st.title("🧠 Класифікація одягу з нейромережею")

st.write("⬆ Завантажте зображення, оберіть модель і перегляньте результат.")

model_choice = st.selectbox("Оберіть модель:", ["CNN (Fashion MNIST)", "VGG"])

uploaded_file = st.file_uploader("Завантажити зображення...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Зображення завантажено", use_container_width=True)

    if model_choice == "VGG":
        model = vgg16_model
        use_vgg = True
        grayscale = False
    else:
        model = cnn_model
        use_vgg = False
        grayscale = True

    predicted_label, probabilities = predict_image(img, model, use_vgg, grayscale)

    st.subheader("🔍 Передбачений клас:")
    st.markdown(f"**{predicted_label}**")

    st.subheader("📊 Ймовірності:")
    for lbl, prob in zip(labels, probabilities):
        st.write(f"{lbl}: {prob:.2%}")

    st.subheader("📈 Графіки тренування:")
    col1, col2 = st.columns(2)
    with col1:
        st.image("plots/loss_plot.png", caption="Втрати", use_container_width=True)
    with col2:
        st.image("plots/accuracy_plot.png", caption="Точність", use_container_width=True)
else:
    st.info("Очікую на зображення...")
