import random
import shutil
import os

PATH = r""
CHOSEN_PATH = r""
N = len(os.listdir(PATH))
print(N)

def pick_random_integers(N, a, b):
    return random.sample(range(a, b+1), N)

def choose_files():
    array = pick_random_integers(50, 0, N)
    
    j = 0
    for i, file in enumerate(os.listdir(PATH)):
        if i in array:
            j += 1
            new_name = f"Audiofile{j}.wav"
            print(file, new_name)
            # shutil.move(os.path.join(PATH, file), os.path.join(CHOSEN_PATH, new_name))    #### DANGER ZONE ####
            # continue

choose_files()
