# Methods

🇬🇧 English · [🇫🇷 Français](fr/methodes.md) · [⬅ Back to README](../README.md)

What each step does, why it is done that way, and what the numbers mean on a medical
image.

## 1. The three tumour types

An MRI of the head shows the brain as a map of grey levels. The three tumours studied here
are the three most common primary brain tumours, and they differ in ways a classifier can
pick up.

| Type | Where it grows | How it looks on a contrast-enhanced T1 |
|---|---|---|
| **Meningioma** | In the membranes around the brain, pushing on it from outside | Well defined, bright after contrast, attached to the skull or the falx |
| **Glioma** | Inside the brain tissue itself | Irregular, often with a diffuse edge and swelling around it |
| **Pituitary tumour** | At the base of the skull, in the pituitary gland | Small, central, at a very constant position |

These differences are the signal. A pituitary tumour is almost always at the same place,
which is why it is the easiest class. A meningioma touches the edge of the brain, a glioma
sits in the middle of the tissue — and a glioma with a well defined edge can look like a
meningioma, which is exactly where the model makes most of its mistakes.

The images are **T1-weighted with gadolinium contrast**: the contrast agent leaks into
tissue where the blood-brain barrier is damaged, so tumours appear brighter.

## 2. Why the split matters more than the model

A patient does not produce one image. An MRI examination produces a stack of slices, a few
millimetres apart. Two neighbouring slices of the same tumour are nearly the same picture.

So if the slices are shuffled and 30 % are set aside as a test set, the test set contains
slices of patients that are also in the training set. The model is then partly graded on
images it has already seen. It is not learning what a glioma looks like; it is recognising
a patient.

> In this dataset, a random split puts **68 of the 75 patients on both sides at once**.

The fix is to split on the patient. Every slice of a patient goes to the same side, so the
test set only holds people the model has never seen — which is the question that matters:
*does it work on the next patient?*

This project does it three ways at once:

1. The cross-validation uses the **5 folds published with the dataset**, which were built
   patient by patient by its authors.
2. Every split passes through `check_no_patient_overlap`, which **raises** if a patient
   appears twice.
3. A test asserts it on every fold, so the guarantee cannot be lost by accident.

The wrong split is kept in the code, as `leaky_split`, for one purpose: to measure how
much it would have inflated the result. See [results](results.md#the-cost-of-a-wrong-split).

### The quieter version of the same mistake

Cutting an image into 16 regions and treating each region as a training sample multiplies
the dataset by 16 — and leaks just as badly, since 16 regions of one image are not 16
independent observations. The first version of this project did that.

Here, regions are a way of **describing** an image. One image produces **one row**.

## 3. Describing an image

A classifier needs numbers. Four descriptions are compared, from the simplest to the most
expensive.

### Regional histograms — 512 numbers

The image is cut into a 4 × 4 grid, and each cell is described by the histogram of its
grey levels: what share of its pixels is dark, mid-grey, bright. The grid is what makes it
work — a histogram of the whole image loses where the bright pixels are, and position is
most of the signal here.

Two details matter. The histogram is computed over 32 bins on the range 0–255: asking for
more bins than there are grey levels, as a first version did with 512 bins on an 8-bit
image, leaves half of them always empty. And the grid edges are computed with
`np.linspace`, so a side that is not divisible by 4 does not lose its last row.

### Intensity statistics — 80 numbers

Five numbers per cell instead of a histogram: mean, spread, skewness, kurtosis and
entropy. Much shorter, and it says something a histogram does not — entropy measures how
disordered the region is, which separates smooth tissue from a textured lesion.

### Texture: local binary patterns and gradients — 1 924 numbers

A histogram says how many pixels of each brightness a region holds, but nothing about how
they are arranged. Two very different tissues can share a histogram.

**Local binary patterns** compare each pixel with its neighbours on a small circle and
record the pattern of "brighter / darker" answers. The result describes the local
structure, and it is unchanged when the whole image gets brighter — useful, because MRI
intensity has no absolute meaning and varies between examinations.

**Histograms of oriented gradients** record the direction of the edges across the image.
The rounded, sharply defined border of a meningioma and the diffuse border of a glioma
produce different edge distributions.

### Deep features — 512 numbers

Training a convolutional network from scratch needs far more than 449 images. The usual
answer is **transfer learning**: take VGG16 trained on millions of photographs, drop its
classifier, and use what its last convolutional block computes as a description of the
image. A small classifier is then trained on those vectors.

One choice matters: the activations are **averaged over each feature map**, which gives
512 numbers instead of 25 088. On a few hundred images, the long vector overfits; the
short one generalises.

## 4. The classifiers

| Classifier | How it decides |
|---|---|
| **Nearest class mean** | Averages the feature vectors of each class; answers the closest average |
| **k-NN** | Answers the majority class among the 5 closest training images |
| **SVM (RBF)** | Finds the boundary with the widest margin, in a transformed space |
| **Logistic regression** | Fits a linear boundary, giving calibrated probabilities |
| **Random forest** | Averages 300 decision trees, each on a random part of the data |

The nearest class mean is the rule this project started from, kept on purpose. It is the
weakest model in the table, and keeping it makes the gain of everything else measurable.

Three of the five are set to `class_weight="balanced"`, which makes a mistake on a rarer
class cost more — the right default when the classes are not equally frequent.

No hyper-parameter search is run. Every model gets a reasonable default, identically, so
the comparison measures the methods rather than how long each was tuned. Tuning properly
would need a second cross-validation loop inside each fold.

## 5. Measuring it

**Accuracy is not reported alone.** With three classes, a model can look acceptable while
missing one type almost entirely. The metrics used:

| Metric | What it answers |
|---|---|
| **Balanced accuracy** | Accuracy if the three classes were equally frequent — the headline number |
| **Macro F1** | Precision and recall combined, averaged over classes with equal weight |
| **Sensitivity** (per class) | Of all the real cases of this type, how many were found |
| **Specificity** (per class) | Of all the cases that are not this type, how many were correctly left out |
| **AUC** (one vs rest) | How well the model ranks a case of the class above a case that is not |

Sensitivity and specificity are the pair clinicians read. A model with high specificity and
low sensitivity is cautious: when it says "glioma" it is usually right, but it misses
cases. That is precisely the behaviour of the best model here, and no single number would
have shown it.

Everything is compared against the **majority-class baseline**: always answer the most
frequent class, which scores 33.4 % here. A result is only meaningful above it.

## 6. What this does not do

- It classifies **slices**, not patients. A clinical tool would combine every slice of an
  examination into one decision per person.
- It does not **localise** the tumour. The dataset includes a hand-drawn tumour mask for
  every slice; the area is kept in the index, but segmentation is left for later.
- It has no **external validation**. All the images come from two hospitals with one
  acquisition protocol, and a model that works there usually loses several points
  elsewhere.
- It has **no confidence interval** beyond the spread across folds. With 75 patients the
  true interval around 84.2 % is several points wide.

**This is a study project, not a medical device.** No part of it has been clinically
validated.
