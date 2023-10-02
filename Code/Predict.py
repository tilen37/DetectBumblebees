import numpy as np
import ModelPredict as mp
import tensorflow as tf
import var
import os
import datetime
import Preprocessing as prep
import AudioCut as ac
from datetime import datetime, timedelta

# Open directory  => Split Audio Y  => Create Spectrogram Y => Cut Spectrogram Y
# => Predict     (=> Cut Positive   => Save Positive)

PATH = r""
POSITIVE_SAVE_PATH = r""
SAVE = False

def predict(PATH):
    print("Working on:", PATH)
    images = prep.create_spectrogram_tensors(PATH)
    print("Number of images:", len(images))
    prediction = np.array([])
    for tensor in images:
        prediction = np.append(prediction, mp.predict(tf.reshape(tensor, (1, var.IMG_WIDTH, var.IMG_HEIGHT, 1)), confidence=True))

    # running_average = 3
    # copy = prediction.copy()
    # for i in range(len(prediction) + 1 - running_average):
    #     prediction[i] = np.average(copy[i:i-1+running_average])

    # print(prediction)
    
    j = 0
    BumblebeeTimes = []
    segment_width = var.SECONDS_PER_IMG * 2
    overlap_fraction = var.OVERLAP

    for i in range(len(prediction)):
        if prediction[i] > var.THRESHOLD:
            # Calculate time stamp considering overlap
            time_stamp = i * (1 - overlap_fraction) * segment_width
            BumblebeeTimes.append(time_stamp)
            j += 1

    ranges = []
    if len(BumblebeeTimes) > 0:
        start = BumblebeeTimes[0]
        end = start + 2

        for i in range(1, len(BumblebeeTimes)):
            if BumblebeeTimes[i] == end:
                end += 2
            else:
                ranges.append((start, end))
                start = BumblebeeTimes[i]
                end = start + 2
            
        ranges.append((start, end))

    for i in ranges:
        print(datetime.utcfromtimestamp(i[0]).strftime('%M:%S'), datetime.utcfromtimestamp(i[1]).strftime('%M:%S'))

    if SAVE and len(BumblebeeTimes) > 0:
        ac.cut_audio_files(PATH, ranges, POSITIVE_SAVE_PATH)
        
    print("Number of detections:", len(ranges))

for file in os.listdir(PATH):
    predict(os.path.join(PATH,file))
