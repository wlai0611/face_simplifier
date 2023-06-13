
# A very simple Flask Hello World app for you to get started with...
from EigenfaceProjection import EigenfaceProjection
from flask import Flask, render_template, request, flash
from werkzeug.utils import secure_filename
from image_processing import convert_to_greyscale, extract_face, pipeline, compress_face
import cv2
import numpy as np
from pathlib import Path
import os
import base64
import re

app = Flask(__name__)
app.secret_key = os.urandom(1)
THIS_FOLDER    = Path(__file__).parent.resolve()

eigenfaces = {
  'neither': np.load(THIS_FOLDER / "best_eigenfaces.npy"),
  'female' : np.load(THIS_FOLDER / "female_eigenfaces.npy"),
  'male'   : np.load(THIS_FOLDER / "male_eigenfaces.npy"),
}

static_folder  = THIS_FOLDER / "static"

pic_size    = (116, 116)

production = False
if production:
  retrieval_path = 'https://www.pythonanywhere.com/user/walterlai/files/home/walterlai/mysite/static'
else:
  retrieval_path = 'static'

@app.route('/')
def index():
    return render_template('index.html')

def image_to_html(image_path):
   image_file   = open(image_path, 'rb')
   image_bytes  = base64.b64encode(image_file.read())
   bytes_string = re.findall("b'(.+)'", str(image_bytes))[0]
   file_extension = re.findall('\.([a-zA-Z]+)$', image_path)[0]
   image_element= f'<img src="data:image/{file_extension};base64, {bytes_string}" width="500" height="500">'
   return image_element

def show_image_in_html():
   original_image_element = image_to_html(projector.original_filepath)
   reconstructed_image_element = image_to_html(projector.reconstruct_filepath)
   return f'''
        <!doctype html>
        Original<br>
        {original_image_element}<br>
        Compressed<br>
        To simplify your face further, click Remove Features.
        <form action="/remove_features" method="post">
        <button type="submit">Remove Features</button>
        </form>
        If the face is too simple and does not resemble your face, click add features.
        <form action="/add_features" method="post">
        <button type="submit">Add Features</button>
        </form>

        {reconstructed_image_element}<br>
        
    '''

@app.route('/display_image', methods=['GET','POST'])
def display_image():
    if request.method=='POST':
        file       = request.files['file']
        gender     = request.form['gender']
        sexed_eigenfaces = eigenfaces[gender]
        img_path   = static_folder/secure_filename(file.filename)
        file.save(img_path.as_posix())

        pic_array  = cv2.imread(img_path.as_posix())
        try:
            pic_array  = pipeline(pic_array, pic_size=pic_size)
        except ValueError:
            flash("Please Upload only JPG or PNG files with at least 1 human face that is front facing.")
            return render_template('index.html')
        global projector
        projector = EigenfaceProjection(original_image = pic_array, 
                                         eigenfaces     = sexed_eigenfaces,
                                         n_components   = 50)
        projector.project_face()
        reconstruct_face = projector.projection

        compressed_face_path     = static_folder / secure_filename(file.filename)
        projector.set_filepath(compressed_face_path)

        cv2.imwrite(projector.reconstruct_filepath, reconstruct_face)
        return show_image_in_html()

@app.route("/remove_features", methods=['POST'])
def remove_features():
   if request.method=='POST':
      projector.add_components(-10)
      cv2.imwrite(projector.reconstruct_filepath, projector.projection)
      return show_image_in_html()
   
@app.route("/add_features", methods=['POST'])
def add_features():
   if request.method=='POST':
      projector.add_components(10)
      cv2.imwrite(projector.reconstruct_filepath, projector.projection)
      return show_image_in_html()

if __name__ == '__main__':
   app.run(debug = True)