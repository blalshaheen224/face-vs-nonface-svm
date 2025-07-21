#  Face vs Non-Face Classification using SVM

This project performs binary classification to distinguish between **face** and **non-face** images using a **Support Vector Machine (SVM)** model. The face images are taken from the `Olivetti Faces` dataset, and the non-face images are from the `Digits` dataset — both are publicly available via `scikit-learn`.

---

##  Datasets Used

- **Olivetti Faces**: Contains 400 grayscale face images (64x64), accessible via `sklearn.datasets.fetch_olivetti_faces`.
- **Digits**: Contains 1797 images of handwritten digits (8x8), accessible via `sklearn.datasets.load_digits`.

These datasets are part of the open-source **scikit-learn** library and are used here for educational and research purposes.

---

##  Project Workflow & Code Explanation

### 1.  Importing Libraries
```python
from sklearn.datasets import fetch_olivetti_faces, load_digits
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt
```

### 2.  Loading and Preparing the Face Images
```python
data = fetch_olivetti_faces()
X_faces = data.data
X_faces_resize = np.array([cv2.resize(img, (8, 8)) for img in X_faces])
x_faces_resize = X_faces_resize.reshape(len(X_faces_resize), -1)
y_faces = np.ones(len(x_faces_resize))
```

### 3.  Loading and Preparing the Non-Face Images
```python
digits = load_digits()
X_non_faces = digits.images.reshape(len(digits.images), -1)
X_non_faces = X_non_faces / 16.0
y_non_faces = np.zeros(len(X_non_faces))
```

### 4.  Combining and Scaling the Data
```python
x = np.vstack((x_faces_resize, X_non_faces))
y = np.hstack((y_faces, y_non_faces))
scalar = StandardScaler()
x_scalar = scalar.fit_transform(x)
```

### 5.  Splitting Data
```python
X_train, X_test, y_train, y_test = train_test_split(x_scalar, y, test_size=0.2, random_state=42)
```

### 6.  PCA for Visualization (2D Projection)
```python
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_scalar)

plt.scatter(...)
```

### 7.  Training the Classifier
```python
model = LinearSVC(C=10, max_iter=5000, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)
```

### 8.  Evaluation: Accuracy and Confusion Matrix
```python
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
```

### 9.  Evaluation: ROC Curve
```python
y_scores = model.decision_function(X_test)
fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

plt.plot(...)
```







##  Requirements

- Python 3.8+
- `scikit-learn`
- `numpy`
- `matplotlib`
- `opencv-python`

Install using:
```bash
pip install -r requirements.txt
```

---

##  License

This project uses datasets that are publicly distributed with `scikit-learn` and is intended for educational and research purposes only.
