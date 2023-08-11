from flask import Flask, request, render_template, send_from_directory, jsonify
import os
from roar_main import arg_main
import re

app = Flask(__name__)

UPLOAD_FOLDER = "/home/roar-nexus/Downloads"
parent_folder = os.path.dirname(os.abspath(__file__))
OUTPUT_FOLDER = os.path.join(parent_folder, "roar_annotations")
ANN_OUT = os.path.join("output", "annotations_output", "annotation.zip")
IMAGES = ["image1.jpg", "image2.jpg", "image3.jpg"]
QUEUE = []
current_image_index = 0

@app.route('/')
def index():
    return render_template('index.html', image_url=f'/uploads/{IMAGES[current_image_index]}')

@app.route('/upload', methods=['POST', 'GET'])
def upload_file():
    try:
        if 'file' not in request.files:
            return 'No file part', 400
        file = request.files['file']
        if file.filename == '':
            return 'No selected file', 400
        if file:
            r = request.get_json(force=True)
            job_id = int(r['jobId'])
            threads = int(r['threads'])
            reseg_bool = not (r['jobType'] == "initial segmentation")
            on_pattern = r'([O|o][n|N])'
            reuse_annotation_output = bool(re.match(on_pattern, r['reuseAnnotation']))
            delete_zip = bool(re.match(on_pattern, r['delete_zip']))
            frames = []
            if reseg_bool:
                frames = r['frames'].split(",")
            
            filename = file.filename
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            if not os.path.exists(UPLOAD_FOLDER):
                return 'Specified UPLOAD_FOLDER in server does not exist', 400
            file.save(filepath)
            
            arg_main(job_id=job_id, reseg_bool=reseg_bool, reuse_output=reuse_annotation_output,
                    threads=threads, reseg_frames=frames, delete_zip=delete_zip)
    except Exception as e:
        print(f"Error while uploading with error: {e}")
        # return 'File uploaded successfully'
        
        


@app.route('/segment', methods=['POST'])
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
    app.run(port=5000, debug=True)
