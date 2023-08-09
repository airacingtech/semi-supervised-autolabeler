from flask import Flask, request, render_template, send_from_directory, jsonify
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
IMAGES = ["image1.jpg", "image2.jpg", "image3.jpg"]
queue = []
current_image_index = 0

@app.route('/')
def index():
    return render_template('index.html', image_url=f'/uploads/{IMAGES[current_image_index]}')

@app.route('/upload', methods=['POST'])
def upload_file():
    #... [rest unchanged]

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
