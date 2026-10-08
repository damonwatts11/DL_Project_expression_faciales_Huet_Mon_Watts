# Reconnaissance et analyse automatique d'expressions faciales

Projet **Fondamentaux du Deep Learning** — Master, Semestre 1 (2026-2027).

Système de vision par ordinateur qui analyse des images de visages et reconnaît
différentes expressions faciales (ex. : colère, dégoût, peur, joie, tristesse,
surprise, neutre). Le projet part d'un classifieur travaillant sur un visage
déjà extrait, puis évolue vers un système capable d'analyser une image — voire
une vidéo — contenant une ou plusieurs personnes.

> Problème de **classification multiclasse** : `K` classes → `K` neurones de
> sortie → **Softmax**.

## Équipe

Projet réalisé en binôme.

| Membre | GitHub |
|--------|--------|
| Daniel Mon Watts | [@damonwatts11](https://github.com/damonwatts11) |
| Alexandre Huet      | [@SpectreAH](https://github.com/SpectreAH) |

**Formatrice :** Hanane Zerdoum · **Soutenance :** vendredi 9 octobre 2026

## Outils

- Python, Google Colab
- Keras / TensorFlow
- YOLO (extension — détection de visages)

## Organisation du projet

Le projet est découpé en neuf parties. Les parties 1 à 6 constituent le socle
obligatoire ; les parties 7 à 9 sont des enrichissements.

| Partie | Contenu | Statut |
|:------:|---------|--------|
| 1 | Recherche, compréhension et préparation des données | Obligatoire |
| 2 | Modèle de référence (réseau dense) | Obligatoire |
| 3 | Construction d'un CNN avec Keras | Obligatoire |
| 4 | Entraînement du réseau | Obligatoire |
| 5 | Évaluation et analyse des erreurs | Obligatoire |
| 6 | Expérimentation et amélioration (≥ 3 expériences) | Obligatoire |
| 7 | Enrichissement avec les nouvelles notions du cours | Enrichissement |
| 8 | Détection et analyse de plusieurs visages (YOLO) | Extension |
| 9 | Extension à la vidéo | Extension (bonus) |

Le travail se fait dans [`notebooks/expressions_faciales.ipynb`](notebooks/expressions_faciales.ipynb).

## Pipeline visé

```
Image / Vidéo → YOLO (détection des visages) → extraction → CNN → Softmax
              → expression + probabilité
```

## Structure du dépôt

```
.
├── notebooks/      # notebooks Colab (socle + extensions)
├── data/           # datasets (ignoré par git — voir .gitignore)
├── models/         # modèles entraînés / checkpoints (ignoré par git)
├── src/            # code réutilisable (prétraitement, modèles, utils)
├── requirements.txt
└── README.md
```

## Installation

### En local (venv)

```bash
python -m venv .venv
source .venv/bin/activate      # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

### Sur Google Colab

La plupart des dépendances sont préinstallées. Pour les compléter :

```python
!pip install -r requirements.txt
```

## Livrables

1. Un notebook Python / Colab structuré, fonctionnel et commenté, exécutable de
   bout en bout.
2. Une présentation de la démarche, des architectures, des expériences et des
   résultats.
3. Une démonstration du modèle final.
