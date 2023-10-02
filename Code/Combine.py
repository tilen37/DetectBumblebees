import os
import wave

def combine_wav_files(input_folder, output_file):
    # List all files in the input folder
    file_list = [f for f in os.listdir(input_folder) if f.lower().endswith('.wav')]

    if not file_list:
        print("No .wav files found in the input folder.")
        return

    # Sort the files to combine them in a specific order, if needed
    file_list.sort()

    # Initialize the output wave file
    output_wave = wave.open(output_file, 'wb')

    for file_name in file_list:
        input_path = os.path.join(input_folder, file_name)
        with wave.open(input_path, 'rb') as input_wave:
            # If this is the first file, set the output file's parameters based on it
            if output_wave.getnframes() == 0:
                output_wave.setparams(input_wave.getparams())

            # Append the frames of the current input file to the output file
            output_wave.writeframes(input_wave.readframes(input_wave.getnframes()))

    output_wave.close()
    print("Combined wav file created:", output_file)


input_folder = r""
output_file = r""
combine_wav_files(input_folder, output_file)
