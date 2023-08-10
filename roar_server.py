from flask import Flask, request, render_template, send_from_directory, jsonify
import os

app = Flask(__name__)

UPLOAD_FOLDER = "/home/roar-nexus/Downloads"
parent_folder = os.path.dirname(os.abspath(__file__))
OUTPUT_FOLDER = os.path.join(parent_folder, "roar_annotations")
ANN_OUT = os.path.join("output", "annotations_output", "annotation.zip")
IMAGES = ["image1.jpg", "image2.jpg", "image3.jpg"]
queue = []
current_image_index = 0

@app.route('/')
def index():
    return render_template('index.html', image_url=f'/uploads/{IMAGES[current_image_index]}')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    if file:
        filename = file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        if not os.path.exists(UPLOAD_FOLDER):
            return 'Specified UPLOAD_FOLDER in server does not exist', 400
        file.save(filepath)
        return 'File uploaded successfully'


@app.route('/uploads/<filename>')
def serve_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/forward', methods=['GET'])
def forward_image():
    global current_image_index
    current_image_index = (current_image_index + 1) % len(IMAGES)
    return jsonify({"image_url": f'/uploads/{IMAGES[current_image_index]}'})

@app.route('/backward', methods=['GET'])
def backward_image():
    global current_image_index
    current_image_index = (current_image_index - 1) % len(IMAGES)
    return jsonify({"image_url": f'/uploads/{IMAGES[current_image_index]}'})

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
