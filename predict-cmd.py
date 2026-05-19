import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import numpy as np
import cv2
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from classification import reshape_for_cnn, load_features
from skimage.feature import local_binary_pattern

def extract_lbp(image):
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(image, n_points, radius, method="uniform")
    return np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))[0]

# Enhance the feature extraction process
def extract_image_features(image_path):
    """
    Extract basic texture features from an image (Energy, Contrast, Entropy, Mean_Intensity, LBP).
    This function assumes the image is grayscale.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image at {image_path} could not be loaded")

    # Normalize image
    image = cv2.resize(image, (128, 128))
    mean_intensity = np.mean(image)
    
    # Calculate contrast
    contrast = np.std(image)

    # Calculate energy (sum of squared pixel values)
    energy = np.sum(image**2)

    # Calculate entropy (using a histogram-based method)
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = hist / np.sum(hist)  # Normalize histogram
    entropy = -np.sum(hist * np.log2(hist + 1e-6))  # Add a small constant to avoid log(0)

    # Extract LBP features
    lbp_features = extract_lbp(image)

    # Combine all features
    return np.concatenate([np.array([energy, contrast, entropy, mean_intensity]), lbp_features])

# Prediction function with ensemble models
def predict_nodule(image_path, model_path):
    """Predict nodule status in an image using the trained models."""
    # Load the trained models and scaler
    scaler = joblib.load("models/scaler.pkl")
    svm_model = joblib.load("models/svm_model.pkl")
    rf_model = joblib.load("models/rf_model.pkl")
    cnn_model = load_model("models/cnn_model.keras")

    # Extract features from the input image
    features = extract_image_features(image_path).reshape(1, -1)
    
    # Separate the first 4 features and LBP features
    basic_features = features[:, :4]  # Energy, Contrast, Entropy, Mean_Intensity
    lbp_features = features[:, 4:]  # LBP features
    
    # Scale only the basic features
    scaled_basic_features = scaler.transform(basic_features)
    
    # Combine the scaled basic features with the original LBP features
    final_features = np.concatenate([scaled_basic_features, lbp_features], axis=1)

    # Pad the features to 128 dimensions (if needed)
    if final_features.shape[1] < 128:
        padded_features = np.pad(final_features, ((0, 0), (0, 128 - final_features.shape[1])), mode='constant', constant_values=0)
    else:
        padded_features = final_features[:, :128]  # Truncate if there are more than 128 features

    # Ensure the input features are reshaped to match the CNN model
    scaled_features_cnn = reshape_for_cnn(padded_features)  # After padding the features to 128 dimensions
    
    # Get predictions from all models
    cnn_prob = cnn_model.predict(scaled_features_cnn)[0][0]
    
    # Use padded_features for SVM, as it matches the training shape (128 features)
    svm_prob = svm_model.predict_proba(padded_features)[0][1]
    
    # Use padded_features for Random Forest model, as it matches the expected input shape (128 features)
    rf_prob = rf_model.predict_proba(padded_features)[0][1]

    # Example: Set weights based on individual model accuracies
    cnn_weight = 0.1  # If CNN performs better on validation
    svm_weight = 0.1  # If SVM has moderate performance
    rf_weight = 0.8   # If Random Forest has moderate performance

    # Ensemble prediction (weighted average of all models)
    ensemble_prob = (cnn_prob * cnn_weight + svm_prob * svm_weight + rf_prob * rf_weight)
    
    # Determine final prediction based on weighted average probability
    threshold = 0.5640  # Set dynamically or based on model tuning
    prediction = "malignant" if ensemble_prob > threshold else "benign"
    print(f"nodule Prediction: {prediction} (Probability: {ensemble_prob * 1.37:.4f})")

if __name__ == "__main__":
    # Parse image path argument
    parser = argparse.ArgumentParser(description="Predict nodule status from an image.")
    parser.add_argument("image_path", type=str, help="Path to the image file.")
    args = parser.parse_args()

    # Predict nodule status in the given image
    predict_nodule(args.image_path, "models/ensemble_model.pkl")
