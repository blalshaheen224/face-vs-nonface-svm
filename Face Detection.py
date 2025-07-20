from sklearn.datasets import fetch_olivetti_faces
import numpy as np
from sklearn.datasets import load_digits
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

data = fetch_olivetti_faces()
X_faces = data.data             # (400, 4096)
X_faces_resize =  np.array([ cv2.resize(img,(8,8)) for img in X_faces])
x_faces_resize = X_faces_resize.reshape(len(X_faces_resize),-1)
y_faces = np.ones(len(x_faces_resize))  # كلهم وجوه → 1

digits = load_digits()
X_non_faces = digits.images.reshape(len(digits.images), -1)  # تحويل لـ نفس الشكل (n_samples, 64x64 لو عايز تعمل resize)
X_non_faces = X_non_faces / 16.0  # توحيد القيم لتكون من 0 إلى 1
y_non_faces = np.zeros(len(X_non_faces))  # كلهم مش وجوه → 0

x = np.vstack((x_faces_resize, X_non_faces))     # دمج الصور
y = np.hstack((y_faces, y_non_faces))  
   # دمج التصنيفات
scalar = StandardScaler()
x_scalar = scalar.fit_transform(x)

X_train,X_test,y_train,y_test = train_test_split(x_scalar,y,test_size=0.2,random_state=42)

pca = PCA(n_components = 2)
x_pca = pca.fit_transform(x_scalar)

plt.figure(figsize=(8,6))
plt.scatter(x_pca[y==1 ,0] , x_pca[y==1 , 1],label ="face(1)",alpha=0.6)
plt.scatter(x_pca[y==0 ,0] , x_pca[y==0 , 1],label ="face(0)",alpha=0.6)
plt.title('PCA Projection to 2D')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()

model = LinearSVC(C=10, max_iter=5000, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)



y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


y_scores = model.decision_function(X_test)
fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
