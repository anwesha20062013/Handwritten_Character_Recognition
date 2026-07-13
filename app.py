import streamlit as st
import numpy as np
import cv2

from streamlit_drawable_canvas import st_canvas

from predictor import Predictor
from segment import segment_digits

st.set_page_config(
    page_title="AI Handwritten Recognition",
    layout="wide"
)

st.title("🧠 AI Handwritten Character & Digit Recognition using CNN")

st.sidebar.title("Recognition Mode")

mode = st.sidebar.radio(

    "Choose Model",

    ["Digits", "Characters"]

)
st.sidebar.markdown("---")

st.sidebar.subheader("Project Info")

st.sidebar.write("📚 Dataset: MNIST + EMNIST Balanced")

st.sidebar.write("🧠 Model: Convolutional Neural Network (CNN)")

st.sidebar.write("⚙️ Framework: TensorFlow & Streamlit")

st.sidebar.markdown("---")

st.sidebar.info(

"""
Models

🔢 Digit CNN (MNIST)

🔤 Character CNN (EMNIST Balanced)
"""

)


@st.cache_resource
def load_predictor(mode):

    if mode == "Digits":
        return Predictor("digit")

    return Predictor("character")


predictor = load_predictor(mode)

st.info("✍️ Draw one or more handwritten characters or digits in the canvas below.")

canvas_result = st_canvas(

    fill_color="black",

    stroke_width=18,

    stroke_color="white",

    background_color="black",

    width=700,

    height=220,

    drawing_mode="freedraw",

    key="canvas"

)

if st.button("🚀 Predict"):

    if canvas_result.image_data is not None:

        img = canvas_result.image_data.astype(np.uint8)

        img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)

        digit_images = segment_digits(img)

        st.success(f"✅ Detected {len(digit_images)} character(s)/digit(s)")

        if len(digit_images) == 0:

            st.error("Nothing detected!")

            st.stop()

        predictions, confidences = predictor.predict(digit_images)

        result = "".join(predictions)

        st.markdown("## 🎯 Prediction")

        st.success(result)

        avg = sum(confidences)/len(confidences)

        st.metric(

            "Overall Confidence",

            f"{avg:.2f}%"

        )

        st.subheader("Detected Characters / Digits")

        cols = st.columns(len(digit_images))

        for i,digit in enumerate(digit_images):

            with cols[i]:

                st.image(

                    digit,

                    width=70,

                    clamp=True

                )

                st.markdown(

                    f"### {predictions[i]}"

                )

                st.caption(

                    f"{confidences[i]:.2f}%"

                )
st.markdown("---")
st.caption("Developed by Anwesha Barik | AI Handwritten Character & Digit Recognition using CNN")