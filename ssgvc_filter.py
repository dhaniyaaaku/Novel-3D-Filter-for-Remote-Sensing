import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization, Input, concatenate

def load_hsi_data(image_path, gt_path):
    """Load hyperspectral image and ground truth from .mat files."""
    image_data = scipy.io.loadmat(image_path)
    gt_data = scipy.io.loadmat(gt_path)
   
    image_key = [key for key in image_data.keys() if not key.startswith('__')][0]
    gt_key = [key for key in gt_data.keys() if not key.startswith('__')][0]
   
    return image_data[image_key], gt_data[gt_key]

def hyperspectral_spatial_spectral_filter(hypercube):
    """
    Computes spatial-spectral feature extraction from a hyperspectral image (hypercube)
    by applying a simultaneous 3D operation that captures both spatial and spectral information.
    """
    X, Y, Z = hypercube.shape
    feature_map = np.zeros((X, Y))
   
    # Pad the hypercube using np.pad
    padded_cube = np.pad(hypercube, ((1, 1), (1, 1), (0, 0)), mode='reflect')
   
    for i in range(X):
        for j in range(Y):
            neighborhood = padded_cube[i:i+3, j:j+3, :]
            spatial_grad_x = np.mean(np.abs(neighborhood[2, :, :] - neighborhood[0, :, :]))
            spatial_grad_y = np.mean(np.abs(neighborhood[:, 2, :] - neighborhood[:, 0, :]))
            spatial_gradient = np.sqrt(spatial_grad_x**2 + spatial_grad_y**2)
            spectral_variation = np.std(neighborhood, axis=2).mean()
           
            central_pixel = neighborhood[1, 1, :]
            spatial_spectral_correlation = np.mean([
                np.dot(central_pixel, neighborhood[di+1, dj+1, :]) /
                (np.linalg.norm(central_pixel) * np.linalg.norm(neighborhood[di+1, dj+1, :]) + 1e-10)
                for di in [-1, 0, 1] for dj in [-1, 0, 1] if not (di == 0 and dj == 0)
            ])
           
            feature_map[i, j] = (spatial_gradient * spectral_variation) * (1 + spatial_spectral_correlation)
   
    return feature_map

def preprocess_data(hypercube, gt):
    """Preprocess the dataset by normalizing and reshaping it for 3D CNN input."""
    scaler = MinMaxScaler()
    X, Y, Z = hypercube.shape
    reshaped_cube = scaler.fit_transform(hypercube.reshape(-1, Z)).reshape(X, Y, Z)
   
    feature_map = hyperspectral_spatial_spectral_filter(reshaped_cube)
   
    # Extracting patches (cube regions) for training
    patch_size = 5  # Small 3D patches
    half_patch = patch_size // 2
   
    X_data, y_data = [], []
    positions = []  # To keep track of pixel positions
   
    for i in range(half_patch, X - half_patch):
        for j in range(half_patch, Y - half_patch):
            if gt[i, j] > 0:  # Ignore unlabeled pixels
                patch = reshaped_cube[i - half_patch:i + half_patch + 1, j - half_patch:j + half_patch + 1, :]
                X_data.append(patch)
                y_data.append(gt[i, j])
                positions.append((i, j))
   
    X_data = np.array(X_data)
    y_data = np.array(y_data).reshape(-1, 1)
    positions = np.array(positions)
   
    # One-hot encode the labels
    encoder = OneHotEncoder(sparse_output=False)
    y_data_onehot = encoder.fit_transform(y_data)
   
    return X_data, y_data, y_data_onehot, feature_map, positions

def apply_lda_to_features(feature_map, positions, y_data):
    """
    Apply LDA to the spatial-spectral feature map with automatic component selection
    """
    # Extract features for labeled pixels
    X_lda = feature_map[positions[:, 0], positions[:, 1]].reshape(-1, 1)
    y_lda = y_data.ravel()
   
    # Determine maximum possible components
    n_classes = len(np.unique(y_lda))
    max_components = min(1, n_classes - 1)  # Since we only have 1 feature
   
    # Apply LDA
    lda = LDA(n_components=max_components)
    X_lda_transformed = lda.fit_transform(X_lda, y_lda)
   
    return X_lda_transformed, lda

def build_3d_cnn_with_features(input_shape, num_classes, lda_features_shape):
    """Builds the 3D CNN model that incorporates additional features."""
    # Input for 3D patches
    input_patches = Input(shape=input_shape, name='input_patches')
   
    # Modified 3D CNN branch with adjusted architecture
    x = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(input_patches)
    x = BatchNormalization()(x)
    x = MaxPooling3D(pool_size=(1, 1, 2))(x)  # Only pool along spectral dimension
   
    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D(pool_size=(1, 1, 2))(x)  # Only pool along spectral dimension
   
    x = Flatten()(x)
   
    # Input for additional features (LDA in this case)
    input_features = Input(shape=lda_features_shape, name='input_features')
   
    # Concatenate CNN features with additional features
    concatenated = concatenate([x, input_features])
   
    # Dense layers
    x = Dense(128, activation='relu')(concatenated)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)
   
    model = Model(inputs=[input_patches, input_features], outputs=output)
   
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Main execution
if __name__ == "__main__":
    pavia_path = "C:/Users/venga/Downloads/pavia/PaviaU.mat"
    pavia_gt_path = "C:/Users/venga/Downloads/pavia/PaviaU_gt.mat"
   
    # Load data
    pavia_img, pavia_gt = load_hsi_data(pavia_path, pavia_gt_path)
   
    # Preprocess data
    X_patches, y_labels, y_onehot, feature_map, positions = preprocess_data(pavia_img, pavia_gt)
   
    # Apply LDA to spatial-spectral features
    X_lda, lda_model = apply_lda_to_features(feature_map, positions, y_labels)
   
    # Train-test split
    X_train_patches, X_test_patches, y_train, y_test = train_test_split(
        X_patches, y_onehot, test_size=0.2, random_state=42)
    X_train_lda, X_test_lda = train_test_split(
        X_lda, test_size=0.2, random_state=42)
   
    # Reshape input data for CNN
    X_train_patches = X_train_patches[..., np.newaxis]  # Add channel dimension
    X_test_patches = X_test_patches[..., np.newaxis]
   
    # Build and train the model
    model = build_3d_cnn_with_features(
        input_shape=X_train_patches.shape[1:],
        num_classes=y_train.shape[1],
        lda_features_shape=(X_train_lda.shape[1],)
    )
   
    print("Model summary:")
    model.summary()
   
    print("\nStarting training...")
    history = model.fit(
        [X_train_patches, X_train_lda], y_train,
        epochs=100,
        batch_size=32,
        validation_data=([X_test_patches, X_test_lda], y_test),
        verbose=1
    )
   
    # Evaluate the model
    test_loss, test_acc = model.evaluate([X_test_patches, X_test_lda], y_test, verbose=0)
    print(f"\nTest Accuracy: {test_acc * 100:.2f}%")
   
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
   
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
