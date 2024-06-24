import librosa
import numpy as np
import os
from pydub.generators import WhiteNoise
from pydub import AudioSegment
import soundfile as sf 
import numpy as np
class AudioUtils:
    def __init__(self):
        self.count=0
    
    def convert_all_to_wav(self,root_directory):
        for directory, _, filenames in os.walk(root_directory):
            for filename in filenames:
                try:
                    filepath = os.path.join(directory, filename)
                    audio = AudioSegment.from_file(filepath)

                    new_filename = os.path.splitext(filename)[0] + ".wav"
                    new_filepath = os.path.join(directory, new_filename)
                    audio.export(new_filepath, format="wav")

                except Exception as e:
                    print(f"Error converting {filepath}: {e}")

    
    def preprocess_all(self,root_directory, output_directory):
        try:
            # Ensure the output directory exists
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)

            for dirpath, _, filenames in os.walk(root_directory):
                for filename in filenames:
                    try:
                        # Construct full path to the current file
                        filepath = os.path.join(dirpath, filename)

                        # Load the audio file
                        y, sr = librosa.load(filepath)

                        # Detect non-silent segments
                        non_silent_intervals = librosa.effects.split(y, top_db=20)  # Adjust top_db as needed

                        # Find the effective 1 second segment
                        effective_duration = 1  # Desired duration in seconds
                        effective_samples = int(effective_duration * sr)  # Convert to number of samples

                        # Initialize variables to store the effective segment
                        effective_segment = np.array([], dtype=y.dtype)

                        # Extract non-silent segments until we reach 1 second
                        for start, end in non_silent_intervals:
                            segment = y[start:end]
                            effective_segment = np.concatenate((effective_segment, segment))
                            if len(effective_segment) >= effective_samples:
                                effective_segment = effective_segment[:effective_samples]
                                break

                        # If the effective segment is still less than 1 second, pad with silence
                        if len(effective_segment) < effective_samples:
                            padding = np.zeros(effective_samples - len(effective_segment), dtype=y.dtype)
                            effective_segment = np.concatenate((effective_segment, padding))

                        # Determine output subdirectory relative to root_directory
                        relative_dir = os.path.relpath(dirpath, root_directory)
                        output_subdirectory = os.path.join(output_directory, relative_dir)

                        # Ensure the output subdirectory exists
                        if not os.path.exists(output_subdirectory):
                            os.makedirs(output_subdirectory)

                        # Get the base filename without extension
                        base_filename, _ = os.path.splitext(filename)

                        # Save the effective segment as a WAV file in the output subdirectory
                        output_filename = f"{base_filename}.wav"
                        output_filepath = os.path.join(output_subdirectory, output_filename)
                        sf.write(output_filepath, effective_segment, sr)

                    except Exception as e:
                        print(f"Error converting {filepath}: {e}")

        except Exception as e:
            print(f"Error processing directory {root_directory}: {e}")

        
    def load_audio(self,file_path):
        return AudioSegment.from_file(file_path)

    def save_augmented(self,audio, file_path, suffix):
        output_file_path = f"{os.path.splitext(file_path)[0]}_{suffix}.wav"
        audio.export(output_file_path, format='wav')

    def pad_audio(self,augmented_audio, original_length):
        if len(augmented_audio) < original_length:
            padding = AudioSegment.silent(duration=original_length - len(augmented_audio))
            return augmented_audio + padding
        return augmented_audio[:original_length]



    def increase_volume(self,song, original_length, file_path):
        louder_song = self.pad_audio(song + 50, original_length)
        self.save_augmented(louder_song, file_path, "louder")

    def decrease_volume(self,song, original_length, file_path):
        quieter_song = self.pad_audio(song -20, original_length)
        self.save_augmented(quieter_song, file_path, "quieter")

    def change_speed_up(self,song, original_length, file_path):
        speed_up = self.pad_audio(song.speedup(playback_speed=2), original_length)
        self.save_augmented(speed_up, file_path, "faster")

    def change_speed_down(self,song, original_length, file_path):
        slow_down = self.pad_audio(song.speedup(playback_speed=0.6), original_length)
        self.save_augmented(slow_down, file_path, "slower")

    
    def add_noise(self,song, original_length, file_path):
        noise = WhiteNoise().to_audio_segment(duration=original_length)
        noisy_song = self.pad_audio(song.overlay(noise - 30), original_length)
        self.save_augmented(noisy_song, file_path, "noisy")

    
    def fade_in_out(self,song, original_length, file_path):
        fade_in_out_song = self.pad_audio(song.fade_in(2000).fade_out(2000), original_length)
        self.save_augmented(fade_in_out_song, file_path, "fade_in_out")

    

    def process_file(self,file_path):
        song = self.load_audio(file_path)
        original_length = len(song)

        self.increase_volume(song, original_length, file_path)
        self.decrease_volume(song, original_length, file_path)
        self.change_speed_up(song, original_length, file_path)
        self.change_speed_down(song, original_length, file_path)
        self.add_noise(song, original_length, file_path)
        self.fade_in_out(song, original_length, file_path)
        
    def process_main_directory(self,main_dir):
        for subdir in os.listdir(main_dir):
            subdir_path = os.path.join(main_dir, subdir)
            if os.path.isdir(subdir_path):
                for root, _, files in os.walk(subdir_path):
                    for file in files:
                        if file.endswith(".wav"):
                            file_path = os.path.join(root, file)
                            self.process_file(file_path)
                            # print(f"Processed: {file_path}")

