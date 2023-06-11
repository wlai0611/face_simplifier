from flask import Flask, render_template, request, make_response
from werkzeug.utils import secure_filename
from image_processing import convert_to_greyscale, extract_face, pipeline, compress_face
import cv2
import numpy as np

app = Flask(__name__)
root_url    = ''
pic_folder  = 'static/'
pic_size    = (116, 116)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/display_image', methods=['GET','POST'])
def display_image():
    if request.method=='POST':
        file = request.files['file']
        file.save(pic_folder+secure_filename(file.filename))
        pic_array = cv2.imread(pic_folder+secure_filename(file.filename))
        pic_array = pipeline(pic_array, pic_size=pic_size)
        eigenfaces= np.load(f'{root_url+pic_folder}/best_eigenfaces.npy')
        reconstruct_face = compress_face(new_face = pic_array, eigenfaces = eigenfaces, n_components=50)
        cv2.imwrite(pic_folder+"reconstruct"+secure_filename(file.filename), reconstruct_face)
        image_display = f'''
        <!doctype html>
        <img src="{root_url+pic_folder}/reconstruct{secure_filename(file.filename)}">
        '''
        return image_display


if __name__ == '__main__':
   app.run(debug = True)