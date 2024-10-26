import tensorflow as tf
import librosa
import numpy as np
import matplotlib.pyplot as plt

# Load audio file and compute spectrogram
def load_audio_and_spectrogram(file_path):
    y, sr = librosa.load(file_path)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    return spectrogram

# New function to display the spectrogram
def display_spectrogram(spectrogram, sr, title='Mel Spectrogram'):
    plt.figure(figsize=(12, 8))
    librosa.display.specshow(spectrogram, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def process_and_display_mp3(file_path):
    y, sr, spectrogram = load_audio_and_spectrogram(file_path)
    display_spectrogram(spectrogram, sr, title=f'Spectrogram of {file_path}')


# Load and preprocess data
# ... (load spectrograms and labels)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_spectrograms, train_labels, epochs=10, validation_data=(val_spectrograms, val_labels))

# Evaluate the model
test_loss, test_acc = model.evaluate(test_spectrograms, test_labels)
print('Test accuracy:', test_acc)

mp3_file_path = 'eastern_birds/02 Mourning Dove Song.mp3'
process_and_display_mp3(mp3_file_path)