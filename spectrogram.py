import librosa
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import threading
import time

# Load audio file and compute spectrogram
def load_audio_and_spectrogram(file_path):
    y, sr = librosa.load(file_path)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    return y, sr, spectrogram_db

# New function to display the spectrogram
def display_spectrogram(spectrogram, sr, title='Mel Spectrogram'):
    plt.figure(figsize=(12, 8))
    librosa.display.specshow(spectrogram, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()

# New function to play audio in a separate thread
def play_audio_thread(y, sr):
    def play():
        print("Playing audio...")
        sd.play(y, sr)
        sd.wait()
        print("Audio playback finished.")

    audio_thread = threading.Thread(target=play)
    audio_thread.start()
    return audio_thread

# Load, display, and play a single MP3 file
def process_display_and_play_mp3(file_path):
    y, sr, spectrogram = load_audio_and_spectrogram(file_path)
    display_spectrogram(spectrogram, sr, title=f'Spectrogram of {file_path}')
    
    # Start audio playback in a separate thread
    audio_thread = play_audio_thread(y, sr)
    
    # Keep the plot window open and update it
    plt.ion()
    plt.show()
    
    # Loop to keep the program running
    while audio_thread.is_alive():
        plt.pause(0.1)
    
    # Wait for audio playback to finish
    audio_thread.join()
    
    # Keep the plot open until manually closed
    print("Audio finished. Close the plot window to exit.")
    plt.ioff()
    plt.show()

# Example usage
if __name__ == "__main__":
    mp3_file_path = '../data/eastern_birds/02 Mourning Dove Song.mp3'
    process_display_and_play_mp3(mp3_file_path)
    


# To recap, the key elements that made this work are:
#
# 1.  Using a separate thread for audio playback, which allows 
#     the program to continue running and display the spectrogram 
#     while the audio plays.
# 2. Implementing a loop that keeps the program active and updates the plot while 
#    the audio is playing.
# 3. Using matplotlib's interactive mode (plt.ion()) to keep the plot responsive.
# 4. Properly handling the transition from interactive mode to blocking mode after 
#    the audio finishes, ensuring the plot stays open until manually closed.
#
# This approach provides a good balance between functionality and user experience. 
# You can now visually analyze the spectrogram of the birdsong while simultaneously 
# listening to it, which can be very useful for bird vocalization studies 
# or audio analysis in general.
# If you want to extend this further, you could consider:
# 1. Adding controls to pause/resume the audio playback.
# 2. Implementing a feature to highlight the current position in the spectrogram
#    as the audio plays.
# 3. Adding functionality to load and analyze multiple audio files in sequence.

# Feel free to ask if you have any questions about the code or if you'd like 
# to explore any additional features or improvements!