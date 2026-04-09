import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Config
IMAGE_SIZE = 128
DATASET_PATH = 'dataset'
MODEL_PATH = 'model/brain_model.h5'

CLASSES = {
    'no_tumor': 0,
    'glioma': 1,
    'meningioma': 2,
    'pituitary': 3
}

def load_data():
    X = []
    y = []
    print("Loading data...")
    for class_name, label in CLASSES.items():
        class_path = os.path.join(DATASET_PATH, class_name)
        if not os.path.exists(class_path):
            print(f"Warning: Path {class_path} not found.")
            continue
        
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
                    X.append(img)
                    y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    # Normalize pixel values
    X = X.astype('float32') / 255.0
    # One-hot encode the labels
    if len(y) > 0:
        y = to_categorical(y, num_classes=4)
        
    return X, y

def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_metrics(history):
    # Plot accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    print("Training metrics saved to training_metrics.png")

if __name__ == '__main__':
    X, y = load_data()
    
    if len(X) == 0:
        print("Error: No data loaded. Please check the dataset path and structure.")
        exit(1)
        
    print(f"Loaded {len(X)} images.")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = build_model()
    model.summary()
    
    print("Starting training...")
    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)
    
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    
    # Save the model
    os.makedirs('model', exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    # Plot training metrics
    plot_metrics(history)
