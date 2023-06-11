from flask import Flask, render_template, request, make_response
from werkzeug.utils import secure_filename
from image_processing import convert_to_greyscale, extract_face
import cv2
import numpy as np

app = Flask(__name__)
root_url    = ''
pic_folder  = 'static/'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/display_image', methods=['GET','POST'])
def display_image():
    if request.method=='POST':
        file = request.files['file']
        file.save(pic_folder+secure_filename(file.filename))
        pic_array = cv2.imread(pic_folder+secure_filename(file.filename))
        greyscale_array = convert_to_greyscale(pic_array)
        zoom_in_face    = extract_face(np.uint8(greyscale_array))
        cv2.imwrite(pic_folder+"greyscale"+secure_filename(file.filename), zoom_in_face)
        image_display = f'''
        <!doctype html>
        <img src="{root_url+pic_folder}/greyscale{secure_filename(file.filename)}">
        '''
        return image_display


if __name__ == '__main__':
   app.run(debug = True)