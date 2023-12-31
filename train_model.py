import numpy as np
import cv2
from image_processing import convert_to_greyscale, extract_face, pipeline
import os
import re
import matplotlib.pyplot as plt
import time

def lfw_paths():
    data_folder      = 'lfw_funneled'
    data_subfolders = os.listdir(data_folder)
    picture_paths    = [[f'{data_folder}/{subfolder}/{path}' for path in os.listdir(data_folder+'/'+subfolder)][0] for subfolder in data_subfolders
                        if re.findall(pattern='^[_a-zA-Z]+$', string=subfolder)]
    return picture_paths

def female_paths():
    folder_name = 'ashwingupta/Male and Female face dataset/Female Faces'
    filenames = os.listdir(folder_name)
    absolute_paths = [f"{folder_name}/{filename}" for filename in filenames]
    return absolute_paths

def male_paths():
    folder_name = 'ashwingupta/Male and Female face dataset/Male Faces'
    filenames = os.listdir(folder_name)
    absolute_paths = [f"{folder_name}/{filename}" for filename in filenames]
    return absolute_paths

picture_paths = female_paths()
rng = np.random.RandomState(seed=1)
n_samples = 600
sampling_index = rng.randint(low = 0, high = len(picture_paths), size = n_samples)
picture_length = 250
picture_width  = 250

#flattened_pictures = np.zeros(shape=(picture_length*picture_width, n_samples))
face_shapes    = np.zeros(shape=(n_samples,2))
pic_size       = (116,116)
flat_face_list = []

normalize = lambda pic: (pic-pic.min()) * 256/(pic.max()-pic.min()) 

start = time.time()
for face_number, sample_number in enumerate(sampling_index):
    pic_array = cv2.imread(picture_paths[sample_number])
    greyscale_array = convert_to_greyscale(pic_array)
    greyscale_array = np.uint8(greyscale_array)
    try:
        zoom_in_face     =  extract_face(greyscale_array)
    except ValueError: #if no face is present
        continue
    resized_face = cv2.resize(src = zoom_in_face, dsize=pic_size)
    resized_face = normalize(resized_face)
    flat_face_list.append(resized_face.flatten())
    #flattened_pictures[:,column_number] = greyscale_array.flatten()
print(time.time()-start)

flat_faces = np.array(flat_face_list).T
#Check that 3rd column is a picture plt.imshow(flat_faces[:,3].reshape(116,116));plt.show()

average_face = flat_faces.mean(axis=1)[:,np.newaxis]
centered_faces = flat_faces - average_face
normalized_faces = centered_faces/np.sum(centered_faces**2,axis=0)**0.5 

flat_eigenfaces,importances,vh = np.linalg.svd(normalized_faces, full_matrices=False)
best_eigenfaces = flat_eigenfaces[:,:50]
new_face = cv2.imread(picture_paths[12])
new_face = pipeline(new_face, pic_size=pic_size)
reconstruct = best_eigenfaces @ best_eigenfaces.T @ new_face.flatten()
reconstruct = reconstruct.reshape(pic_size)   
fig, ax = plt.subplots(ncols=2)
ax[0].imshow(new_face)
ax[1].imshow(reconstruct)
plt.show()
np.save(open('female_eigenfaces_600.npy','wb'), flat_eigenfaces)
print()



