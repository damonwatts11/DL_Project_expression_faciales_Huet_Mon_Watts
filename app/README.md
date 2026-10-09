# Application Streamlit - Expressions faciales

Demo interactive du projet : prediction d'expression faciale (CNN FER+)
et detection de plusieurs visages (YOLOv8-face).

## Lancer

```bash
pip install -r app/requirements.txt
streamlit run app/app.py
```

## Modele requis

L'application charge `models/emotion_cnn.keras`. Exportez-le depuis le notebook,
apres avoir choisi le modele final (partie 6) :

```python
cnn.save('models/emotion_cnn.keras')
```

Vous pouvez pointer vers un autre chemin via la variable d'environnement `EMOTION_MODEL`.

## Fonctionnalites

- **Image** : televerser une photo ou en prendre une avec la webcam -> expression
  predite + histogramme des probabilites.
- **Video / Webcam** : detection multi-visages avec YOLOv8-face, en direct
  (streamlit-webrtc) ou sur une video televersee.

Le detecteur YOLOv8-face est telecharge automatiquement dans `models/` au premier
lancement (repli OpenCV Haar si indisponible).
