from pydub import AudioSegment
from datetime import datetime
import os

# Function to convert HH:MM:SS time format to milliseconds
def convert_to_millis(time):
    # Convert time string to milliseconds
    h, m, s = str(datetime.utcfromtimestamp(time).strftime('%H:%M:%S')).split(':')
    return int(h) * 3600000 + int(m) * 60000 + int(float(s) * 1000)

# Function to process and cut audio files
def cut_audio_files(audio_path, timestamp_pairs, output_path):
    # Load the audio file
    audio = AudioSegment.from_wav(audio_path)
    name = audio_path.split('/')[-1].split('.')[0]
    # output_path = os.path.join(output_path, name)

    # Process each pair of timestamps
    for pair in timestamp_pairs:
        start_time, end_time = pair
        start_time_millis = convert_to_millis(start_time)
        end_time_millis = convert_to_millis(end_time)

        # Cut the audio file
        cut_audio = audio[start_time_millis:end_time_millis]

        # Create the output file name
        output_filename = f"{str(name)}_{str(datetime.utcfromtimestamp(start_time).strftime('%M:%S')).replace(':', '')}-{str(datetime.utcfromtimestamp(end_time).strftime('%M:%S')).replace(':', '')}.wav"
        output_file_path = os.path.join(output_path, output_filename)

        # Save the cut audio file
        cut_audio.export(output_file_path, format="wav")