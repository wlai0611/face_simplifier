from flask import Flask, render_template, request, make_response
from werkzeug.utils import secure_filename
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/display_image', methods=['GET','POST'])
def display_image():
    if request.method=='POST':
        file = request.files['file']
        file.save('static/'+secure_filename(file.filename))
        image_display = f'''
        <!doctype html>
        <img src="static/{secure_filename(file.filename)}">
        '''
        return image_display

if __name__ == '__main__':
   app.run(debug = True)