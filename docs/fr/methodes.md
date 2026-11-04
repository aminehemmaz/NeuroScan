# Méthodes

[🇬🇧 English](../methods.md) · 🇫🇷 Français · [⬅ Retour au README](../../README.fr.md)

Ce que fait chaque étape, pourquoi elle est faite ainsi, et ce que signifient les chiffres
sur une image médicale.

## 1. Les trois types de tumeur

Une IRM de la tête montre le cerveau sous forme d'une carte de niveaux de gris. Les trois
tumeurs étudiées ici sont les trois tumeurs cérébrales primitives les plus fréquentes, et
elles diffèrent d'une façon qu'un classifieur peut saisir.

| Type | Où elle se développe | Aspect sur une T1 avec injection |
|---|---|---|
| **Méningiome** | Dans les membranes autour du cerveau, qu'elle comprime de l'extérieur | Bien délimité, brillant après injection, accolé au crâne ou à la faux |
| **Gliome** | Dans le tissu cérébral lui-même | Irrégulier, souvent à bord flou, avec un œdème autour |
| **Tumeur pituitaire** | À la base du crâne, dans l'hypophyse | Petite, centrale, à une position très constante |

Ces différences sont le signal. Une tumeur pituitaire est presque toujours au même
endroit : c'est la classe la plus facile. Un méningiome touche le bord du cerveau, un
gliome siège au milieu du tissu — et un gliome bien délimité peut ressembler à un
méningiome, ce qui est précisément là que le modèle commet le plus d'erreurs.

Les images sont des **T1 avec injection de gadolinium** : le produit de contraste diffuse
là où la barrière hémato-encéphalique est altérée, ce qui fait apparaître les tumeurs plus
brillantes.

## 2. Pourquoi la séparation compte plus que le modèle

Un patient ne produit pas une image. Un examen IRM produit une pile de coupes, espacées de
quelques millimètres. Deux coupes voisines de la même tumeur sont presque la même image.

Si l'on mélange les coupes et qu'on en met 30 % de côté comme jeu de test, ce jeu de test
contient des coupes de patients présents aussi à l'entraînement. Le modèle est alors noté
en partie sur des images qu'il a déjà vues. Il n'apprend pas à quoi ressemble un gliome ;
il reconnaît un patient.

> Dans ce jeu de données, une séparation aléatoire place **68 des 75 patients des deux
> côtés à la fois**.

La correction consiste à séparer par patient. Toutes les coupes d'un patient vont du même
côté, et le jeu de test ne contient que des personnes jamais vues — ce qui est la question
qui compte : *est-ce que ça marche sur le patient suivant ?*

Ce projet s'en assure de trois façons simultanées :

1. La validation croisée utilise les **5 plis publiés avec le jeu de données**, construits
   patient par patient par ses auteurs.
2. Chaque séparation passe par `check_no_patient_overlap`, qui **lève une erreur** si un
   patient apparaît deux fois.
3. Un test l'affirme sur chaque pli, pour que la garantie ne puisse pas se perdre par
   accident.

La mauvaise séparation est conservée dans le code, sous le nom `leaky_split`, dans un seul
but : mesurer de combien elle aurait gonflé le résultat. Voir les
[résultats](resultats.md#5-le-coût-dune-mauvaise-séparation).

### La version plus discrète de la même erreur

Découper une image en 16 régions et traiter chaque région comme un échantillon
d'entraînement multiplie le jeu de données par 16 — et fuit tout autant, puisque 16 régions
d'une même image ne sont pas 16 observations indépendantes. La première version de ce
projet faisait exactement cela.

Ici, les régions sont une façon de **décrire** une image. Une image produit **une ligne**.

## 3. Décrire une image

Un classifieur a besoin de nombres. Quatre descriptions sont comparées, de la plus simple à
la plus coûteuse.

### Histogrammes par région — 512 nombres

L'image est découpée en une grille 4 × 4, et chaque case est décrite par l'histogramme de
ses niveaux de gris : quelle part de ses pixels est sombre, moyenne, claire. C'est la
grille qui rend la méthode efficace — un histogramme de l'image entière perd l'information
de position, or la position est l'essentiel du signal ici.

Deux détails comptent. L'histogramme est calculé sur 32 classes dans l'intervalle 0–255 :
demander plus de classes qu'il n'y a de niveaux de gris, comme le faisait une première
version avec 512 classes sur une image 8 bits, en laisse la moitié toujours vides. Et les
bords de la grille sont calculés avec `np.linspace`, pour qu'un côté non divisible par 4 ne
perde pas sa dernière ligne.

### Statistiques d'intensité — 80 nombres

Cinq nombres par case au lieu d'un histogramme : moyenne, dispersion, asymétrie,
aplatissement et entropie. Bien plus court, et cela dit quelque chose qu'un histogramme ne
dit pas — l'entropie mesure le désordre de la région, ce qui sépare un tissu lisse d'une
lésion texturée.

### Texture : motifs binaires locaux et gradients — 1 924 nombres

Un histogramme dit combien de pixels de chaque luminosité contient une région, mais rien de
leur agencement. Deux tissus très différents peuvent avoir le même histogramme.

Les **motifs binaires locaux** comparent chaque pixel à ses voisins sur un petit cercle et
enregistrent la suite des réponses « plus clair / plus sombre ». Le résultat décrit la
structure locale, et il ne change pas quand toute l'image s'éclaircit — ce qui est utile,
car l'intensité en IRM n'a pas de valeur absolue et varie d'un examen à l'autre.

Les **histogrammes de gradients orientés** enregistrent la direction des contours dans
l'image. Le bord arrondi et net d'un méningiome et le bord flou d'un gliome produisent des
distributions de contours différentes.

### Caractéristiques profondes — 512 nombres

Entraîner un réseau convolutif de zéro demande bien plus que 449 images. La réponse usuelle
est le **transfert d'apprentissage** : reprendre VGG16 entraîné sur des millions de
photographies, retirer son classifieur, et utiliser ce que calcule son dernier bloc
convolutif comme description de l'image. Un petit classifieur est ensuite entraîné sur ces
vecteurs.

Un choix compte : les activations sont **moyennées sur chaque carte de caractéristiques**,
ce qui donne 512 nombres au lieu de 25 088. Sur quelques centaines d'images, le vecteur
long surapprend ; le court généralise.

## 4. Les classifieurs

| Classifieur | Comment il décide |
|---|---|
| **Plus proche moyenne de classe** | Moyenne les vecteurs de chaque classe ; répond la moyenne la plus proche |
| **k-NN** | Répond la classe majoritaire parmi les 5 images d'entraînement les plus proches |
| **SVM (RBF)** | Cherche la frontière à la marge la plus large, dans un espace transformé |
| **Régression logistique** | Ajuste une frontière linéaire, avec des probabilités calibrées |
| **Forêt aléatoire** | Moyenne 300 arbres de décision, chacun sur une partie aléatoire des données |

La plus proche moyenne de classe est la règle d'où est parti ce projet, conservée
volontairement. C'est le modèle le plus faible du tableau, et la garder rend mesurable
l'apport de tout le reste.

Trois des cinq utilisent `class_weight="balanced"`, qui rend une erreur sur une classe rare
plus coûteuse — le bon réglage par défaut quand les classes ne sont pas également
fréquentes.

Aucune recherche d'hyperparamètres n'est menée. Chaque modèle reçoit des valeurs par défaut
raisonnables, à l'identique, pour que la comparaison mesure les méthodes et non le temps
passé à régler chacune. Un réglage correct demanderait une seconde boucle de validation
croisée à l'intérieur de chaque pli.

## 5. Le mesurer

**L'accuracy n'est jamais donnée seule.** Avec trois classes, un modèle peut sembler
acceptable tout en ratant presque entièrement un type. Les métriques utilisées :

| Métrique | Ce qu'elle répond |
|---|---|
| **Accuracy équilibrée** | L'accuracy si les trois classes étaient également fréquentes — le chiffre principal |
| **F1 macro** | Précision et rappel combinés, moyennés à poids égal sur les classes |
| **Sensibilité** (par classe) | Sur tous les vrais cas de ce type, combien ont été trouvés |
| **Spécificité** (par classe) | Sur tous les cas qui ne sont pas ce type, combien ont été correctement écartés |
| **AUC** (un contre tous) | À quel point le modèle classe un cas de la classe au-dessus d'un cas qui ne l'est pas |

Sensibilité et spécificité sont le couple que lisent les cliniciens. Un modèle à forte
spécificité et faible sensibilité est prudent : quand il dit « gliome », il a presque
toujours raison, mais il en manque. C'est exactement le comportement du meilleur modèle
ici, et aucun chiffre unique ne l'aurait montré.

Tout est comparé à la **référence de la classe majoritaire** : toujours répondre la classe
la plus fréquente, soit 33,4 % ici. Un résultat n'a de sens qu'au-dessus.

## 6. Ce que cela ne fait pas

- On classe des **coupes**, pas des patients. Un outil clinique combinerait toutes les
  coupes d'un examen en une seule décision par personne.
- La tumeur n'est pas **localisée**. Le jeu de données contient un masque de tumeur tracé à
  la main pour chaque coupe ; son aire figure dans l'index, mais la segmentation est
  laissée pour plus tard.
- Il n'y a pas de **validation externe**. Toutes les images viennent de deux hôpitaux avec
  un seul protocole, et un modèle qui marche là perd en général plusieurs points ailleurs.
- Il n'y a pas d'**intervalle de confiance** au-delà de la dispersion entre plis. Avec 75
  patients, l'intervalle réel autour de 84,2 % couvre plusieurs points.

**Il s'agit d'un projet d'étude, pas d'un dispositif médical.** Rien ici n'a fait l'objet
d'une validation clinique.
