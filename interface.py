
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
  'female' : np.load(THIS_FOLDER / "female_eigenfaces_1000.npy"),
  'male'   : np.load(THIS_FOLDER / "male_eigenfaces_1000.npy"),
}

static_folder  = THIS_FOLDER / "static"

pic_size    = (116, 116)

@app.route('/')
def index():
    return render_template('index.html')

def image_to_html(image_path):
   image_file   = open(image_path, 'rb')
   image_bytes  = base64.b64encode(image_file.read())
   bytes_string = re.findall("b'(.+)'", str(image_bytes))[0]
   file_extension = re.findall('\.([a-zA-Z]+)$', image_path)[0]
   src_string     = f"data:image/{file_extension};base64, {bytes_string}"
   image_element  = f'<img src="{src_string}" width="500" height="500">'
   return file_extension, src_string, image_element

def show_image_in_html():
   og_extension, og_src, original_image_element = image_to_html(projector.original_filepath)
   new_extension, new_src, reconstructed_image_element = image_to_html(projector.reconstruct_filepath)
   n_eigenfaces = projector.eigenfaces.shape[1]
   n_ticks = 5
   eigenfaces_ticks = np.arange(start = 0, stop = n_eigenfaces, step = n_eigenfaces//n_ticks)
   return f'''
        <!doctype html>
        <style>
        datalist {{
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        writing-mode: vertical-lr;
        width: 200px;
        }}

        option {{
          padding: 0;
        }}

        input[type="range"] {{
          width: 200px;
          margin: 0;
        }}

        </style>
        <!-- <script>window.onbeforeunload=function(){{fetch('/close');}};</script> -->
        <h2>Original</h2>
        {original_image_element}<br>
        <br>
        Below your face is compressed using {projector.n_components} eigenfaces.<br>
        To compress the face and make it simpler, drag the slider to the left to a smaller number of eigenfaces and click Update Number Eigenfaces.<br>
        If the face is too simple and does not look like your face, drag the slider to the right to bigger number of eigenfaces and click Update Number Eigenfaces.<br>
        
        <form action="/add_features" method="post">
        <label for="n_eigenfaces">Choose the number of eigenfaces:</label><br />
        <input type="range" id="n_eigenfaces" name="num_eigenfaces" value="{projector.n_components}" list="values" min = "0" max = "{projector.eigenfaces.shape[1]}"/>

        <datalist id="values">
          <option value="{eigenfaces_ticks[0]}" label="{eigenfaces_ticks[0]}"></option>
          <option value="{eigenfaces_ticks[1]}" label="{eigenfaces_ticks[1]}"></option>
          <option value="{eigenfaces_ticks[2]}" label="{eigenfaces_ticks[2]}"></option>
          <option value="{eigenfaces_ticks[3]}" label="{eigenfaces_ticks[3]}"></option>
          <option value="{eigenfaces_ticks[4]}" label="{eigenfaces_ticks[4]}"></option>
          <option value="{projector.eigenfaces.shape[1]}" label="{projector.eigenfaces.shape[1]}"></option>
        </datalist>
        <button type="submit">Update the Number of Eigenfaces</button>
        </form>
        <h2>Compressed Face</h2><br>
        {reconstructed_image_element}<br>
        <a download="compressed.{new_extension}" href="{new_src}"><button>Download Image</button></a>
        <form action="/">
        <button type="submit">Try Another Face</button>
        </form>
    '''

@app.errorhandler(500)
def internal_error(error):

    return "500 error"

@app.route('/close')
def delete_image():
   os.remove(projector.original_filepath)
   os.remove(projector.reconstruct_filepath)
   return f"Deleted {projector.original_filepath} and {projector.reconstruct_filepath}"

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
            flash("Please Upload only JPG or PNG files with at least 1 human face that is front facing.\nTry removing glasses and turning off camera flash.")
            return render_template('index.html')
        global projector
        projector = EigenfaceProjection(original_image = pic_array, 
                                         eigenfaces     = sexed_eigenfaces,
                                         n_components   = sexed_eigenfaces.shape[1]//2)
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
      n_eigenfaces_update = int(request.form["num_eigenfaces"]) - projector.n_components
      projector.add_components(n_eigenfaces_update)
      cv2.imwrite(projector.reconstruct_filepath, projector.projection)
      return show_image_in_html()

if __name__ == '__main__':
   app.run()