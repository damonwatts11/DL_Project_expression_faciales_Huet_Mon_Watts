# Reconnaissance d'expressions faciales - application Streamlit.
#
# Deux sections :
#   1. Image : televerser une photo ou en prendre une -> prediction du CNN.
#   2. Video / Webcam : detection de visages (YOLOv8-face) + emotion en direct,
#      ou sur une video televersee.
#
# Lancement : streamlit run app/app.py
# Prerequis : le modele entraine dans le notebook, sauvegarde en
#             models/emotion_cnn.keras (voir app/README.md).

import os
import tempfile

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# --- Constantes (identiques au notebook) ---
CLASS_NAMES = ['neutre', 'joie', 'surprise', 'tristesse',
               'colere', 'degout', 'peur', 'mepris']
IMG_SIZE = (48, 48)
MODEL_PATH = os.environ.get('EMOTION_MODEL', 'models/emotion_cnn.keras')
FACE_WEIGHTS = 'models/yolov8n-face.pt'
FACE_URL = 'https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.pt'

st.set_page_config(page_title='Expressions faciales', layout='wide')


# --- Chargement (mis en cache : une seule fois) ---
@st.cache_resource(show_spinner='Chargement du modele...')
def load_emotion_model():
    from tensorflow import keras
    return keras.models.load_model(MODEL_PATH)


@st.cache_resource(show_spinner='Chargement du detecteur de visages...')
def load_detector():
    os.makedirs('models', exist_ok=True)
    try:
        from ultralytics import YOLO
        if not os.path.exists(FACE_WEIGHTS):
            import urllib.request
            urllib.request.urlretrieve(FACE_URL, FACE_WEIGHTS)
        return ('yolo', YOLO(FACE_WEIGHTS))
    except Exception as e:  # repli OpenCV Haar si YOLO indisponible
        st.warning('YOLO indisponible, repli sur OpenCV Haar (' + str(e) + ').')
        haar = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        return ('haar', haar)


# --- Traitement ---
def detect_faces(detector, bgr, conf=0.3):
    # Retourne une liste de boites (x1, y1, x2, y2).
    kind, obj = detector
    if kind == 'yolo':
        res = obj.predict(bgr, conf=conf, verbose=False)[0]
        return [tuple(map(int, b.xyxy[0].tolist())) for b in res.boxes]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = obj.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    return [(x, y, x + w, y + h) for (x, y, w, h) in faces]


def preprocess_face(bgr):
    # Un visage BGR -> tenseur (1, 48, 48, 1) normalise, comme a l'entrainement.
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, IMG_SIZE, interpolation=cv2.INTER_AREA)
    gray = gray.astype('float32') / 255.0
    return gray.reshape(1, *IMG_SIZE, 1)


def predict_emotion(model, bgr):
    probs = model.predict(preprocess_face(bgr), verbose=0)[0]
    return CLASS_NAMES[int(probs.argmax())], float(probs.max()), probs


def annotate(model, detector, bgr, conf=0.3):
    # Dessine les boites + emotions sur une image BGR. Retourne (image, visages).
    out = bgr.copy()
    h, w = bgr.shape[:2]
    visages = []
    for (x1, y1, x2, y2) in detect_faces(detector, bgr, conf):
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        label, score, probs = predict_emotion(model, crop)
        visages.append({'label': label, 'score': score, 'probs': probs,
                        'box': (x1, y1, x2, y2)})
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, label + ' ' + format(score, '.2f'), (x1, max(15, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out, visages


def probs_to_df(probs):
    import pandas as pd
    return pd.DataFrame({'probabilite': probs}, index=CLASS_NAMES)


# --- Interface ---
st.title("Reconnaissance d'expressions faciales")
st.caption('CNN entraine sur FER+ (8 classes) + detection de visages YOLOv8-face.')

if not os.path.exists(MODEL_PATH):
    st.error(
        'Modele introuvable : ' + MODEL_PATH + '. Dans le notebook, apres avoir '
        'choisi le modele final, executez  cnn.save("models/emotion_cnn.keras")  '
        'puis placez ce fichier a cet emplacement (ou definissez la variable '
        "d'environnement EMOTION_MODEL)."
    )
    st.stop()

model = load_emotion_model()
detector = load_detector()

tab_image, tab_video = st.tabs(['Image', 'Video / Webcam (YOLO)'])

# ---- Section 1 : image ----
with tab_image:
    st.subheader('Prediction sur une image')
    source = st.radio('Source', ['Televerser une image', 'Prendre une photo'],
                      horizontal=True)
    if source == 'Televerser une image':
        file = st.file_uploader('Image', type=['jpg', 'jpeg', 'png'])
    else:
        file = st.camera_input('Prenez une photo')

    if file is not None:
        image = Image.open(file).convert('RGB')
        bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        out, visages = annotate(model, detector, bgr)

        col1, col2 = st.columns(2)
        col1.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB),
                   caption=str(len(visages)) + ' visage(s) detecte(s)',
                   use_container_width=True)
        if visages:
            principal = max(visages, key=lambda v: v['box'][2] - v['box'][0])
            col2.metric('Expression (plus grand visage)',
                        principal['label'], format(principal['score'], '.0%'))
            col2.bar_chart(probs_to_df(principal['probs']))
        else:
            label, score, probs = predict_emotion(model, bgr)
            col2.info("Aucun visage detecte - prediction sur l'image entiere.")
            col2.metric('Expression', label, format(score, '.0%'))
            col2.bar_chart(probs_to_df(probs))

# ---- Section 2 : video / webcam ----
with tab_video:
    st.subheader('Detection de plusieurs visages (YOLO)')
    mode = st.radio('Mode', ['Webcam en direct', 'Televerser une video'],
                    horizontal=True)
    conf = st.slider('Seuil de confiance (detection)', 0.1, 0.9, 0.3, 0.05)

    if mode == 'Webcam en direct':
        st.write('Autorisez la camera puis lancez le flux.')
        try:
            import av
            from streamlit_webrtc import webrtc_streamer, RTCConfiguration

            def video_frame_callback(frame):
                img = frame.to_ndarray(format='bgr24')
                out, _ = annotate(model, detector, img, conf)
                return av.VideoFrame.from_ndarray(out, format='bgr24')

            webrtc_streamer(
                key='live-emotion',
                video_frame_callback=video_frame_callback,
                rtc_configuration=RTCConfiguration(
                    {'iceServers': [{'urls': ['stun:stun.l.google.com:19302']}]}),
                media_stream_constraints={'video': True, 'audio': False},
            )
        except ImportError:
            st.error('Installez streamlit-webrtc et av pour la webcam en direct '
                     '(voir app/requirements.txt).')

    else:
        vid = st.file_uploader('Video', type=['mp4', 'avi', 'mov', 'mkv'])
        if vid is not None:
            tin = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tin.write(vid.read())
            tin.close()
            cap = cv2.VideoCapture(tin.name)
            fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

            out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                     fps, (w, h))
            progress = st.progress(0.0, 'Traitement de la video...')
            n = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                out, _ = annotate(model, detector, frame, conf)
                writer.write(out)
                n += 1
                if total:
                    progress.progress(min(n / total, 1.0))
            cap.release()
            writer.release()
            progress.empty()
            st.success(str(n) + ' images traitees.')
            st.video(out_path)
