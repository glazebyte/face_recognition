import os
import pickle
from typing import List, Tuple

import numpy as np
import streamlit as st
from PIL import Image
from mtcnn.mtcnn import MTCNN
from keras.models import load_model
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from keras_facenet import FaceNet


# ----------------------------
# Configuration
# ----------------------------
FACENET_MODEL_PATH = "facenet_keras.h5"     # put your facenet_keras.h5 here
EMBEDDINGS_DB_PATH = "face_db.pkl"
CLASSIFIER_PATH = "svm_classifier.pkl"


# ----------------------------
# Utility: database for embeddings
# ----------------------------
class FaceEmbeddingsDB:
    def __init__(self, path: str = EMBEDDINGS_DB_PATH):
        self.path = path
        self.embeddings: List[np.ndarray] = []
        self.labels: List[str] = []
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, "rb") as f:
                data = pickle.load(f)
                self.embeddings = data.get("embeddings", [])
                self.labels = data.get("labels", [])
        else:
            self.embeddings = []
            self.labels = []

    def _save(self):
        with open(self.path, "wb") as f:
            pickle.dump(
                {"embeddings": self.embeddings, "labels": self.labels},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    def add_embedding(self, embedding: np.ndarray, label: str):
        self.embeddings.append(embedding)
        self.labels.append(label)
        self._save()

    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        if len(self.embeddings) == 0:
            return None, None
        X = np.asarray(self.embeddings)
        y = np.asarray(self.labels)
        return X, y

    def count_per_label(self):
        from collections import Counter

        return Counter(self.labels)


# ----------------------------
# Cached global resources
# ----------------------------
@st.cache_resource
def load_facenet_model():
    embedder = FaceNet()   # automatically loads model
    return embedder


@st.cache_resource
def load_mtcnn_detector():
    detector = MTCNN()
    return detector


# ----------------------------
# Face processing / embedding
# ----------------------------
def extract_face(image: Image.Image, detector: MTCNN, required_size=(160, 160)):
    """
    Detect the largest face and return as numpy array (160x160x3)
    """
    image = image.convert("RGB")
    pixels = np.asarray(image)

    results = detector.detect_faces(pixels)
    if len(results) == 0:
        return None

    # pick face with largest bounding box area
    areas = []
    for r in results:
        x, y, w, h = r["box"]
        areas.append(w * h)
    best_idx = int(np.argmax(areas))
    x, y, w, h = results[best_idx]["box"]

    x, y = max(x, 0), max(y, 0)
    x2, y2 = x + w, y + h

    face = pixels[y:y2, x:x2]
    if face.size == 0:
        return None

    face_image = Image.fromarray(face)
    face_image = face_image.resize(required_size)
    face_array = np.asarray(face_image)
    return face_array


def prewhiten(x: np.ndarray) -> np.ndarray:
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = (x - mean) / std_adj
    return y


# ----------------------------
# Classifier handling
# ----------------------------
def train_classifier(db: FaceEmbeddingsDB):
    X, y = db.get_data()
    if X is None or y is None or len(y) == 0:
        raise ValueError("No embeddings in database. Add persons first.")

    if len(np.unique(y)) < 2:
        raise ValueError("Need at least 2 different persons (classes) to train SVM.")

    # L2 normalization
    normalizer = Normalizer(norm="l2")
    X_norm = normalizer.fit_transform(X)

    # SVM with RBF kernel
    model = SVC(kernel="rbf", probability=True)
    model.fit(X_norm, y)

    with open(CLASSIFIER_PATH, "wb") as f:
        pickle.dump({"normalizer": normalizer, "model": model}, f, protocol=pickle.HIGHEST_PROTOCOL)

    return model, normalizer


def load_classifier():
    if not os.path.exists(CLASSIFIER_PATH):
        return None, None
    with open(CLASSIFIER_PATH, "rb") as f:
        data = pickle.load(f)
        normalizer = data["normalizer"]
        model = data["model"]
    return model, normalizer


# ----------------------------
# Streamlit UI
# ----------------------------
def page_add_person():
    st.header("Add Person (Capture from Camera)")

    name = st.text_input("Person name (label)")

    st.write("Capture multiple images. Each press of 'Capture Image' will store one image in memory.")
    
    if "captured_images" not in st.session_state:
        st.session_state["captured_images"] = []

    # Camera input
    captured = st.camera_input("Take a picture")

    # Each time a new picture is captured, store it
    if captured is not None:
        img = Image.open(captured)
        st.session_state["captured_images"].append(img)
        st.success(f"Captured image #{len(st.session_state['captured_images'])}")

    # Show captured images
    if len(st.session_state["captured_images"]) > 0:
        st.write("Captured Images:")
        for idx, im in enumerate(st.session_state["captured_images"]):
            st.image(im, caption=f"Image #{idx+1}", width=200)

    # Save embeddings
    if st.button("Save embeddings"):
        if not name:
            st.error("Please enter a name.")
            return

        if len(st.session_state["captured_images"]) == 0:
            st.error("Please capture at least one image.")
            return

        facenet = load_facenet_model()
        detector = load_mtcnn_detector()
        db = FaceEmbeddingsDB()

        added_count = 0
        for image in st.session_state["captured_images"]:
            face = extract_face(image, detector)
            if face is None:
                st.warning("Face not detected in one image. Skipped.")
                continue

            emb = facenet.embeddings([face])[0]

            db.add_embedding(emb, name)
            added_count += 1

        st.session_state["captured_images"] = []  # reset captures

        if added_count > 0:
            st.success(f"Saved {added_count} face embeddings for '{name}'.")
            st.write("Database:", db.count_per_label())
        else:
            st.error("No valid faces detected. Nothing saved.")



def page_train():
    st.header("Train SVM (RBF) on L2-normalized embeddings")

    if st.button("Train"):
        db = FaceEmbeddingsDB()
        try:
            model, normalizer = train_classifier(db)
        except ValueError as e:
            st.error(str(e))
            return

        st.success("Training complete! Classifier saved.")
        X, y = db.get_data()
        st.write(f"Training samples: {len(y)}")
        st.write("Classes:", np.unique(y))


def page_recognize():
    st.header("Recognize Face")

    model, normalizer = load_classifier()
    if model is None or normalizer is None:
        st.warning("Classifier not trained yet. Go to 'Train' page first.")
        return

    option = st.radio("Input type", ["Upload image", "Camera"])

    image = None
    if option == "Upload image":
        file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if file is not None:
            image = Image.open(file)
    else:
        camera_image = st.camera_input("Take a picture")
        if camera_image is not None:
            image = Image.open(camera_image)

    if image is not None:
        st.image(image, caption="Input image", use_column_width=True)

    if st.button("Recognize") and image is not None:
        detector = load_mtcnn_detector()
        facenet = load_facenet_model()

        face = extract_face(image, detector)
        if face is None:
            st.error("No face detected.")
            return

        emb = facenet.embeddings([face])[0]
        emb_norm = normalizer.transform([emb])
        probs = model.predict_proba(emb_norm)[0]
        classes = model.classes_
        best_idx = int(np.argmax(probs))
        best_class = classes[best_idx]
        best_prob = probs[best_idx]

        st.subheader("Prediction")
        st.write(f"**Person:** {best_class}")
        st.write(f"**Confidence:** {best_prob:.2%}")

        st.subheader("All class probabilities")
        for cls, p in zip(classes, probs):
            st.write(f"{cls}: {p:.2%}")


def main():
    st.sidebar.title("Face Recognition App")
    page = st.sidebar.selectbox(
        "Navigation",
        ["Add Person", "Train", "Recognize"],
    )

    if page == "Add Person":
        page_add_person()
    elif page == "Train":
        page_train()
    elif page == "Recognize":
        page_recognize()


if __name__ == "__main__":
    main()
