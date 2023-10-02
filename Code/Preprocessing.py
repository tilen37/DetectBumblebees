import numpy as np
import librosa
import librosa.display
import var
import tensorflow as tf
import matplotlib.pyplot as plt

def create_segments(spectrogram, window_size, overlap):
    overlap = int(overlap * window_size)
    segments = []
    start = 0
    # print(f"window_size:", window_size)
    # print(spectrogram.shape)
    
    while start + window_size <= spectrogram.shape[1]:
        # print(f"start:", start)
        # print(f"start+windows_size:", start+window_size)
        segment = spectrogram[:, start:start + window_size]
        segments.append(segment)
        start += window_size - overlap
    
    # Include the last frame, even if it's less than 'window_size'
    if start < spectrogram.shape[1]:
        segment = spectrogram[:, -window_size:]
        segments.append(segment)
    
    return segments

def create_spectrogram_tensors(f):
    y, sr = librosa.load(f)
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=int(sr/(var.PX_PER_SEC/var.SECONDS_PER_IMG))*4, hop_length=int(sr/(var.PX_PER_SEC/var.SECONDS_PER_IMG)), n_mels=var.IMG_HEIGHT
        , fmin=0, fmax=1500)
    # )
    img = librosa.power_to_db(S, ref=np.max)

    img = img[::-1]

    # print(f)
    # plt.imshow(img)
    # plt.show()
    
    img = tf.convert_to_tensor(img)
    
    # num_splits = int(np.ceil(img.shape[1] / var.IMG_WIDTH))
    # if num_splits * var.IMG_WIDTH > img.shape[1]:
    #     zeroes_tensor = tf.zeros((img.shape[0], var.IMG_WIDTH * num_splits -
    #                               img.shape[1]), dtype=img.dtype)
    #     tensor_image_gray_expanded = tf.concat(
    #         [img, zeroes_tensor], axis=1)
    # else:
    #     tensor_image_gray_expanded = img
    # tensor_image_gray_expanded = tf.expand_dims(tensor_image_gray_expanded, axis=-1)
    if img.shape[1] < var.IMG_WIDTH:
        zeroes_tensor = tf.zeros((img.shape[0], var.IMG_WIDTH - img.shape[1]), dtype=img.dtype)
        img = tf.concat(
            [img, zeroes_tensor], axis=1)
    img_crop = create_segments(spectrogram=img, overlap=var.OVERLAP, window_size=var.IMG_WIDTH)
    
    j = 0
    tensors = []
    for i in img_crop:
        tensors.append(i)
        
        # if noise:
        #     j += 1
        #     tensors.append(add_noise(i, 30))
        #     tensors.append(add_noise(i, 100))
        #     if j%10:
        #         tensors.append(add_noise(i, 250))
    # for i in range(len(tensors)):
    #     print(f"{f} [{i}]")
    #     plt.imshow(tensors[i])
    #     plt.show()
    # print(f"img_crop[0].shape ",img_crop[0].shape)
    # print(f"img_crop.shape ", len(img_crop))

    # plt.imshow(img_crop[int(len(img_crop)/2)])
    # plt.imshow(img)
    # plt.show()

    return tensors
