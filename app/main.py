import os
from evodiff.pretrained import OA_DM_38M
from flask import Flask, render_template, request, jsonify
from evodiff.generate import generate_oaardm
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'uploads')
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route("/")
def home():
    return render_template("home.html", current_page="home", content=render_template('home.html'))

@app.route("/results")
def results():
    return render_template("results.html", current_page="results", content=render_template('results.html'))

@app.route("/about")
def about():
    return render_template("about.html", current_page="about", content=render_template('about.html'))

@app.route('/get_data', methods=['GET'])
def get_data():
    data = {'message': 'Hello from Flask!'}
    return jsonify(data)

@app.route('/post_data', methods=['POST'])
def post_data():

    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return 'File uploaded successfully', 200
    # data = request.json['data']
    # # process data
    # processed_data = process_data(data)
    # return jsonify({'result': processed_data})

@app.route('/generation', methods=['POST'])
def generation():

    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    message = request.form.get('message')


    if file.filename == '':
        return 'No selected file', 400
    
    if file and allowed_file(file.filename):
        # 处理文件,例如保存
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        # 使用message
        print(f"Received message: {message}")
        
        # 进行后续处理...
        
        return 'File uploaded successfully', 200
    
    return 'Invalid file', 400



def process_data(data):
    # 在这里处理数据
    return f"Processed: {data}"

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8000))
    app.run(debug=True,port = port)