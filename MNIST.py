import numpy as np
import matplotlib.pyplot as plt
import joblib
from PIL import Image

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    classification_report
)

# Load MNIST Dataset

mnist = fetch_openml("mnist_784", version=1, as_frame=False)

# Features and Target

X = mnist.data[:3000]
y = mnist.target[:3000].astype(int)

# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Preserve Original Test Images

X_test_images = X_test.copy()

# Standardize Features

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameter Tuning

param_grid = {
    "C": [0.1, 1, 10, 100],
    "gamma": [0.1, 0.01, 0.001],
    "kernel": ["rbf"]
}

grid = GridSearchCV(
    estimator=SVC(),
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=1
)

# Train Model

grid.fit(X_train, y_train)

print(f"Best Parameters : {grid.best_params_}")
print(f"Best CV Score   : {grid.best_score_:.4f}")

# Calibrate Best Model

model = CalibratedClassifierCV(
    estimator=grid.best_estimator_,
    ensemble=False
)

model.fit(X_train, y_train)

# Predictions

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# Evaluation Metrics

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="macro")
recall = recall_score(y_test, y_pred, average="macro")
f1 = f1_score(y_test, y_pred, average="macro")

print(f"\nAccuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(8, 8))

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
).plot(
    ax=ax,
    cmap="Blues",
    values_format="d"
)

plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()

# ROC Curve

classes = model.classes_

y_test_bin = label_binarize(
    y_test,
    classes=classes
)

plt.figure(figsize=(8, 8))

for i, digit in enumerate(classes):

    fpr, tpr, _ = roc_curve(
        y_test_bin[:, i],
        y_proba[:, i]
    )

    plt.plot(
        fpr,
        tpr,
        label=f"Digit {digit}"
    )

plt.plot([0, 1], [0, 1], "k--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.savefig("roc_curve.png")
plt.close()

roc_auc = roc_auc_score(
    y_test_bin,
    y_proba,
    average="macro",
    multi_class="ovr"
)

print(f"ROC-AUC   : {roc_auc:.4f}")

# Sample Predictions

fig, axes = plt.subplots(3, 3, figsize=(8, 8))

for ax, image, actual, prediction in zip(
    axes.ravel(),
    X_test_images[:9],
    y_test[:9],
    y_pred[:9]
):

    ax.imshow(
        image.reshape(28, 28),
        cmap="gray"
    )

    ax.set_title(
        f"Actual: {actual}\nPred: {prediction}"
    )

    ax.axis("off")

plt.tight_layout()

plt.savefig("sample_predictions.png")

plt.close()

# Save Model

joblib.dump(model, "svm_mnist_model.pkl")
joblib.dump(scaler, "scaler.pkl")

def predict_custom_image(image_path):  # Best results are achieved with images that resemble the MNIST dataset i.e a 28 x 28 grayscale

    # Load Image
    image = Image.open(image_path).convert("L")

    # Resize to MNIST dimensions
    image = image.resize((28, 28))

    # Convert to NumPy Array
    image = np.array(image)

    # Invert colors if background is white
    if image.mean() > 127:
        image = 255 - image

    # Display Original Processed Image
    plt.figure(figsize=(3, 3))
    plt.imshow(image, cmap="gray")
    plt.title("Processed Image")
    plt.axis("off")
    plt.show()

    # Flatten Image
    image_flat = image.reshape(1, -1)

    # Apply Standardization
    image_scaled = scaler.transform(image_flat)

    # Predict
    prediction = model.predict(image_scaled)[0]
    probabilities = model.predict_proba(image_scaled)[0]

    print(f"\nPredicted Digit : {prediction}")
    print(f"Confidence       : {probabilities.max() * 100:.2f}%")

image_path = input("\nEnter image path: ")

predict_custom_image(image_path)



# Accuracy  : 0.9117
# Precision : 0.9170
# Recall    : 0.9114
# F1 Score  : 0.9126 for reference