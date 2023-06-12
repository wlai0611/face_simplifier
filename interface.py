
# A very simple Flask Hello World app for you to get started with...

from flask import Flask, render_template, request, flash
from werkzeug.utils import secure_filename
from image_processing import convert_to_greyscale, extract_face, pipeline, compress_face
import cv2
import numpy as np
from pathlib import Path
import os

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

def show_image_in_html(image_path):
   return f'''
        <!doctype html>
        Original<br>
        <img src="{retrieval_path}/{image_path}" width="500" height="500"><br>
        Compressed<br>
        <form action="/remove_features" method="post">
        <button type="submit">Remove Features</button>
        <input type="hidden" name="filename" value="{image_path}">
        </form>
        <img src="{retrieval_path}/reconstruct{image_path}" width="500" height="500"><br>
        
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


        reconstruct_face = compress_face(new_face = pic_array, eigenfaces = sexed_eigenfaces, n_components=50)

        compressed_face_filename = f"reconstruct{secure_filename(file.filename)}"
        compressed_face_path     = static_folder / compressed_face_filename
        cv2.imwrite(compressed_face_path.as_posix(), reconstruct_face)
        image_display = show_image_in_html(secure_filename(file.filename))
        return image_display

@app.route("/remove_features", methods=['POST'])
def remove_features():
   if request.method=='POST':
      filename = request.form['filename']
      return show_image_in_html(filename)

if __name__ == '__main__':
   app.run(debug = True)