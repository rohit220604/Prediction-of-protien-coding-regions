import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
%matplotlib inline

# Function to preprocess data
def preprocess_data(data, max_length=200):
    sequences = data["sequence"].values
    labels = data["label"].values

    # Convert sequences to numerical form
    X = np.array([encode_sequence(seq, max_length) for seq in sequences])
    y = LabelEncoder().fit_transform(labels)
    return X, y

def plot_roc_curve(y_test, y_pred_prob):
    # Compute ROC curve and ROC area
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

# Function to encode sequences into numerical format
def encode_sequence(sequence, max_length):
    mapping = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
    encoded = [mapping.get(base, 0) for base in sequence]
    if len(encoded) < max_length:
        encoded.extend([0] * (max_length - len(encoded)))
    return encoded[:max_length]

# Function to create the CNN-BRNN model
def create_model(input_shape):
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        Bidirectional(LSTM(64, return_sequences=True)),
        Bidirectional(LSTM(32)),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def plot_confusion_matrix(y_test, y_pred):
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Create a plot
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Coding", "Coding"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.show()

# Function to save results to CSV
def save_results(results, file_name):
    pd.DataFrame([results]).to_csv(file_name, index=False)

# Main workflow
if __name__ == "__main__":
    # Load datasets
    train_data = pd.read_csv("/train.csv")
    test_data = pd.read_csv("/test.csv")

    # Preprocess data
    X_train, y_train = preprocess_data(train_data)
    X_test, y_test = preprocess_data(test_data)

    # Reshape data for CNN
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    # Create and train the model
    model = create_model(input_shape=X_train.shape[1:])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.2f}")

    # Save results
    results = {"accuracy": accuracy}
    save_results(results, "result.csv")
    print("Results saved to result.csv")

    y_pred_prob = model.predict(X_test).flatten()  # Flatten to match shape
    plot_roc_curve(y_test, y_pred_prob)

    # Generate predictions (class labels)
    y_pred = (model.predict(X_test).flatten() > 0.5).astype(int)

    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred)
