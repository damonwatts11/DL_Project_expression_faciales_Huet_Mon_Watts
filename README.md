# Reconnaissance et analyse automatique d'expressions faciales

Projet **Fondamentaux du Deep Learning** — Master, Semestre 1 (2026-2027).

Système de vision par ordinateur qui reconnaît l'expression faciale d'un visage
(8 classes : neutre, joie, surprise, tristesse, colère, dégoût, peur, mépris),
puis l'étend à la **détection de plusieurs visages** (YOLO) sur image et vidéo.

> Classification **multiclasse** : 8 classes → 8 neurones de sortie → **Softmax**.

## Équipe

Projet réalisé en binôme.

| Membre | GitHub |
|--------|--------|
| Daniel Mon Watts | [@damonwatts11](https://github.com/damonwatts11) |
| Alexandre Huet   | [@SpectreAH](https://github.com/SpectreAH) |

**Formatrice :** Hanane Zerdoum

## Résultats

| Modèle | Accuracy test | F1 macro test |
|--------|:-------------:|:-------------:|
| Réseau dense (référence) | 50,6 % | — |
| CNN de référence | 72,6 % | 0,44 |
| **Modèle final** (CNN, learning rate 1e-4) | **73,3 %** | **0,57** |

Le **F1 macro** (moyenne par classe) est notre critère principal car le dataset
est très déséquilibré : l'accuracy seule favorise les classes majoritaires.

Comparaison des expériences (sur la **validation**) :

| # | Modification | Val. accuracy | Val. F1 macro |
|:-:|--------------|:-------------:|:-------------:|
| 0 | CNN de référence | 0,747 | 0,463 |
| 1 | Data augmentation | 0,771 | 0,493 |
| 2 | Poids des classes (`balanced`) | 0,350 | 0,090 |
| 3 | **Learning rate 1e-4** (retenu) | 0,750 | **0,586** |

## Dataset — FER+ (FERPlus)

[FERPlus](https://github.com/microsoft/FERPlus) reprend les images de **FER-2013**
et remplace les labels par une **re-annotation à 10 votes** (plus fiable). Deux fichiers,
alignés ligne à ligne :

| Fichier | Contenu | Source |
|---------|---------|--------|
| `fer2013.csv` | pixels 48×48 niveaux de gris (`emotion, pixels, Usage`) | Kaggle (FER-2013) |
| `fer2013new.csv` | votes des 8 émotions + `unknown`, `NF` | repo microsoft/FERPlus (suivi ici dans `data/`) |

- **35 887 images** ; après **vote majoritaire** (on écarte les votes isolés, on exige
  >50 % des votes restants, on retire `unknown`/`NF`) il reste **31 412 images**.
- Découpage **officiel** via la colonne `Usage` : 25 060 train / 3 199 validation (`PublicTest`) / 3 153 test (`PrivateTest`).
- **Fort déséquilibre** : `neutre`+`joie` ≈ 64 % des images ; `dégoût` et `mépris` ≈ 0,5 % chacune.

## Démarche (notebook `notebooks/expressions_faciales.ipynb`)

| Partie | Contenu | Points clés |
|:------:|---------|-------------|
| 1 | Données & préparation | Chargement + alignement des 2 fichiers, labels par vote majoritaire, split officiel, normalisation `/255`, one-hot |
| 2 | Modèle de référence (dense) | `Flatten → Dense(256) → Softmax` : **contrôle** sans structure spatiale (50,6 %) |
| 3 | CNN | 3 blocs `Conv→BatchNorm→MaxPool` (32/64/128) → `Dense(128)+Dropout` → Softmax ; ~685k paramètres |
| 4 | Entraînement | Adam, `categorical_crossentropy`, EarlyStopping + ReduceLROnPlateau ; lecture des courbes (surapprentissage après l'époque 10) |
| 5 | Évaluation & erreurs | Matrice de confusion + rapport par classe ; confusions typiques `tristesse→neutre`, `peur→surprise` |
| 6 | Expérimentations | 3 expériences (une variable à la fois), choix du modèle final sur le **F1 macro** de validation |
| 7 | Enrichissement | Data augmentation et comparaison des variantes (intégrées à la partie 6) |
| 8 | Détection multi-visages | **YOLOv8-face** (modèle entraîné aux visages, pas COCO) → crop → même prétraitement → CNN |
| 9 | Vidéo (bonus) | `analyser_image` appliqué image par image (~47 images/s) |

**Points défendables.** Le jeu de **test** ne sert qu'à l'évaluation finale ; tous les
choix se font sur la **validation**. L'expérience « poids des classes » a **échoué**
(accuracy 0,35) : les poids extrêmes (26× pour `dégoût`/`mépris`) déstabilisent
l'entraînement — résultat négatif conservé et expliqué.

## Structure du dépôt

```
.
├── notebooks/
│   └── expressions_faciales.ipynb   # notebook complet (parties 0 à 9)
├── data/
│   ├── fer2013new.csv               # labels FER+ (suivi)
│   └── fer2013.csv                  # pixels (NON suivi : ~290 Mo, via Kaggle)
├── models/                          # modèles / poids (non suivis)
├── src/
├── requirements.txt
└── README.md
```

## Exécution

Le notebook télécharge les données automatiquement : `fer2013.csv` via **kagglehub**
(`deadskull7/fer2013`) et `fer2013new.csv` depuis le repo Microsoft. Un `kaggle.json`
peut être nécessaire pour la partie Kaggle.

### Google Colab (recommandé)

*Exécution → Modifier le type d'exécution → GPU*, puis **Exécuter tout**.

### En local

```bash
python -m venv .venv
source .venv/bin/activate      # Windows : .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/expressions_faciales.ipynb
```

## Livrables

1. Notebook Python / Colab structuré, commenté, exécutable de bout en bout.
2. Présentation de la démarche, des architectures, des expériences et des résultats.
3. Démonstration du modèle final (parties 8 et 9).
