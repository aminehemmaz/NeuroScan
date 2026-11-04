# Data

🇬🇧 English · [🇫🇷 Français](#données)

## What is in this folder

| File | Content |
|---|---|
| `images/` | 449 MRI slices, 256 × 256 pixels, 8-bit grey level, PNG |
| `index.csv` | One row per slice: slice id, patient id, tumour class, cross-validation fold, tumour area |

`index.csv` is the file the code reads. Its columns:

| Column | Meaning |
|---|---|
| `slice_id` | Number of the slice in the original dataset (1 to 3064) |
| `patient_id` | Identifier of the patient the slice comes from |
| `label` | 1 meningioma, 2 glioma, 3 pituitary tumour |
| `class` | The same, written out |
| `fold` | Cross-validation fold, 1 to 5 |
| `tumour_pixels` | Area of the tumour in pixels, from the mask of the original dataset |
| `file` | Path of the image, relative to `data/` |

## Where the images come from

> **Jun Cheng, "brain tumor dataset", figshare (2017).**
> <https://doi.org/10.6084/m9.figshare.1512427> — licence **CC BY 4.0**

The original dataset holds **3064 T1-weighted contrast-enhanced slices from 233 patients**,
acquired at Nanfang Hospital (Guangzhou) and at the General Hospital of Tianjin Medical
University, between 2005 and 2010. It is distributed as MATLAB files, each holding the
image, the tumour class, the patient identifier and a tumour mask drawn by hand.

This folder is a **subset**, built by [`scripts/01_prepare_dataset.py`](../scripts/01_prepare_dataset.py):

- **25 patients per class**, so the three classes carry the same weight;
- **up to 6 slices per patient**, spread across the examination rather than taken in a row;
- images rescaled to 256 × 256 and written as 8-bit PNG;
- the fold of each slice taken from `cvind.mat`, the 5-fold assignment published with the
  dataset.

| Class | Patients | Slices |
|---|---|---|
| Meningioma | 25 | 149 |
| Glioma | 25 | 150 |
| Pituitary | 25 | 150 |
| **Total** | **75** | **449** |

### The folds keep a patient together

The whole point of using the published folds is that every slice of a patient falls in the
same fold. In this subset, **no patient appears in more than one fold** — the code checks
it on every split, and the tests check it too.

| Fold | Meningioma | Glioma | Pituitary |
|---|---|---|---|
| 1 | 35 | 30 | 30 |
| 2 | 36 | 30 | 12 |
| 3 | 24 | 30 | 36 |
| 4 | 18 | 30 | 36 |
| 5 | 36 | 30 | 36 |

## Privacy

The images carry no name, no date of birth and no DICOM header. The `patient_id` is the
pseudonymous number already published with the original dataset; it is kept so that a
reader can trace a slice back to its source and, above all, so that the splits can be made
patient by patient. No personal data was added.

## Limits of this subset

- **Small.** 449 slices and 75 patients compare methods fairly, but do not support a
  clinical claim.
- **One source, one protocol.** Two hospitals in the same country, T1-weighted with
  contrast. A different scanner or population could change the ranking of the methods.
- **Slices, not examinations.** Six slices of a patient are not six independent cases, and
  the evaluation is built around that fact.
- **The tumour mask is not used.** Its area is recorded in `index.csv`, but the masks
  themselves are not redistributed here. They are in the original archives, for anyone who
  wants to work on segmentation.

## Rebuilding this folder

Download the four archives and `cvind.mat` from
<https://doi.org/10.6084/m9.figshare.1512427>, then:

```bash
python scripts/01_prepare_dataset.py \
    --archives path/to/figshare/archives \
    --folds path/to/cvind.mat \
    --patients-per-class 25 --slices-per-patient 6
```

Without `--folds`, the script builds its own 5 folds, still patient by patient.

## Citation and terms

The images are distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/),
which allows redistribution with attribution. If you use them, cite:

> Cheng, J., Huang, W., Cao, S., Yang, R., Yang, W., Yun, Z., Wang, Z. and Feng, Q. (2015).
> *Enhanced performance of brain tumor classification via tumor region augmentation and
> partition.* PLoS ONE 10(10): e0140381.

> Cheng, J., Yang, W., Huang, M., Huang, W., Jiang, J., Zhou, Y., Yang, R., Zhao, J., Feng,
> Y., Feng, Q. and Chen, W. (2016). *Retrieval of brain tumors by adaptive spatial pooling
> and Fisher vector representation.* PLoS ONE 11(6): e0157112.

---

# Données

[🇬🇧 English](#data) · 🇫🇷 Français

## Contenu du dossier

| Fichier | Contenu |
|---|---|
| `images/` | 449 coupes IRM, 256 × 256 pixels, niveaux de gris 8 bits, PNG |
| `index.csv` | Une ligne par coupe : identifiant de coupe, de patient, classe, pli de validation croisée, aire de la tumeur |

`index.csv` est le fichier que lit le code. Ses colonnes :

| Colonne | Signification |
|---|---|
| `slice_id` | Numéro de la coupe dans le jeu de données d'origine (1 à 3064) |
| `patient_id` | Identifiant du patient dont provient la coupe |
| `label` | 1 méningiome, 2 gliome, 3 tumeur pituitaire |
| `class` | Le même, en toutes lettres |
| `fold` | Pli de validation croisée, 1 à 5 |
| `tumour_pixels` | Aire de la tumeur en pixels, d'après le masque du jeu d'origine |
| `file` | Chemin de l'image, relatif à `data/` |

## Origine des images

> **Jun Cheng, « brain tumor dataset », figshare (2017).**
> <https://doi.org/10.6084/m9.figshare.1512427> — licence **CC BY 4.0**

Le jeu d'origine contient **3064 coupes T1 avec injection, provenant de 233 patients**,
acquises à l'hôpital Nanfang (Canton) et à l'hôpital général de l'université médicale de
Tianjin, entre 2005 et 2010. Il est distribué sous forme de fichiers MATLAB contenant
chacun l'image, la classe de la tumeur, l'identifiant du patient et un masque de tumeur
tracé à la main.

Ce dossier en est un **sous-ensemble**, construit par
[`scripts/01_prepare_dataset.py`](../scripts/01_prepare_dataset.py) :

- **25 patients par classe**, pour que les trois classes aient le même poids ;
- **jusqu'à 6 coupes par patient**, réparties sur l'examen plutôt que prises à la suite ;
- images redimensionnées en 256 × 256 et écrites en PNG 8 bits ;
- le pli de chaque coupe repris de `cvind.mat`, la répartition en 5 plis publiée avec le
  jeu de données.

| Classe | Patients | Coupes |
|---|---|---|
| Méningiome | 25 | 149 |
| Gliome | 25 | 150 |
| Pituitaire | 25 | 150 |
| **Total** | **75** | **449** |

### Les plis gardent un patient ensemble

Tout l'intérêt d'utiliser les plis publiés est que toutes les coupes d'un patient tombent
dans le même pli. Dans ce sous-ensemble, **aucun patient n'apparaît dans plus d'un pli** :
le code le vérifie à chaque séparation, et les tests le vérifient aussi.

| Pli | Méningiome | Gliome | Pituitaire |
|---|---|---|---|
| 1 | 35 | 30 | 30 |
| 2 | 36 | 30 | 12 |
| 3 | 24 | 30 | 36 |
| 4 | 18 | 30 | 36 |
| 5 | 36 | 30 | 36 |

## Vie privée

Les images ne portent ni nom, ni date de naissance, ni en-tête DICOM. Le `patient_id` est
le numéro pseudonyme déjà publié avec le jeu d'origine ; il est conservé pour qu'un lecteur
puisse remonter à la source et, surtout, pour que les séparations se fassent patient par
patient. Aucune donnée personnelle n'a été ajoutée.

## Limites de ce sous-ensemble

- **Petit.** 449 coupes et 75 patients permettent de comparer des méthodes équitablement,
  pas d'étayer une affirmation clinique.
- **Une seule source, un seul protocole.** Deux hôpitaux du même pays, T1 avec injection.
  Un autre appareil ou une autre population pourrait changer le classement des méthodes.
- **Des coupes, pas des examens.** Six coupes d'un patient ne sont pas six cas
  indépendants, et toute l'évaluation est construite autour de ce fait.
- **Le masque de tumeur n'est pas exploité.** Son aire figure dans `index.csv`, mais les
  masques eux-mêmes ne sont pas redistribués ici. Ils sont dans les archives d'origine,
  pour qui veut travailler sur la segmentation.

## Reconstruire ce dossier

Téléchargez les quatre archives et `cvind.mat` depuis
<https://doi.org/10.6084/m9.figshare.1512427>, puis :

```bash
python scripts/01_prepare_dataset.py \
    --archives chemin/vers/les/archives \
    --folds chemin/vers/cvind.mat \
    --patients-per-class 25 --slices-per-patient 6
```

Sans `--folds`, le script construit ses propres 5 plis, toujours patient par patient.

## Citation et conditions

Les images sont distribuées sous [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/
), qui autorise la redistribution avec attribution. En cas d'utilisation, citer :

> Cheng, J., Huang, W., Cao, S., Yang, R., Yang, W., Yun, Z., Wang, Z. et Feng, Q. (2015).
> *Enhanced performance of brain tumor classification via tumor region augmentation and
> partition.* PLoS ONE 10(10): e0140381.

> Cheng, J., Yang, W., Huang, M., Huang, W., Jiang, J., Zhou, Y., Yang, R., Zhao, J., Feng,
> Y., Feng, Q. et Chen, W. (2016). *Retrieval of brain tumors by adaptive spatial pooling
> and Fisher vector representation.* PLoS ONE 11(6): e0157112.
