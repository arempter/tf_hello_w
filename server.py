import os
from flask import (
    Flask,
    request,
    render_template,
    Response,
    redirect,
    flash,
    url_for
)
from werkzeug.utils import secure_filename
from tensor_model import load_model, predict
from plot_def import generate_pred_graph

app = Flask(__name__)
UPLOAD_FOLDER = '/tmp'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


@app.route('/')
def main():
    return render_template('not_found.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(path)
            label = predict(model, path)
            generate_pred_graph(label)    
            return render_template('uploaded.html', label=label)
    elif request.method == 'GET':
        return render_template('upload.html')

if __name__ == 'server':
    model = load_model('./simple_model.h5')
    app.run()

