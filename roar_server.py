from flask import Flask, request, render_template, send_from_directory, jsonify
from flask_cors import CORS
import os
from roar_main import arg_main, MainHub, create_main_hub
import re

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
UPLOAD_FOLDER = "/home/roar-nexus/Downloads"
# UPLOAD_FOLDER = "C:/Users/chowm/Downloads"
parent_folder = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FOLDER = os.path.join(parent_folder, "roar_annotations")
ANN_OUT = os.path.join("output", "annotations_output")
IMAGES = ["image1.jpg", "image2.jpg", "image3.jpg"]
QUEUE = []
CLIENTS = {}
current_image_index = 0

@app.route('/')
def index():
    return render_template('roar_webpage.html', image_url=f'/uploads/{IMAGES[current_image_index]}')

@app.route('/upload', methods=['POST', 'GET'])
def upload_file():
    try:
        if request.method == 'GET':
            return "Nice try uploading..."
        r = request.get_json(force=True)
        job_id = int(r['jobId'])
        threads = int(r['threads'])
        reseg_bool = not (r['jobType'] == "initial segmentation")
        # on_pattern = r'([O|o][n|N])'
        reuse_annotation_output = bool(r.get('reuseAnnotation'))
        delete_zip = bool(r.get('delete_zip'))
        frames = []
        
        if reseg_bool:
            frames = r['frames'].split(",")
                
        if 'file' not in request.files and not reuse_annotation_output and reseg_bool:
            return 'No file part', 400
        if 'file' in request.files:
            file = request.files['file']
            
            if file.filename == '' and not reuse_annotation_output:
                return 'No selected file', 400
            if file:
                
                
                filename = str(file.filename)
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                if not os.path.exists(UPLOAD_FOLDER):
                    return 'Specified UPLOAD_FOLDER in server does not exist', 400
                file.save(filepath)
            
        arg_main(job_id=job_id, reseg_bool=reseg_bool, reuse_output=reuse_annotation_output,
                threads=threads, reseg_frames=frames, delete_zip=delete_zip)
        job_folder = os.path.join(OUTPUT_FOLDER, str(job_id))
        annotation_output = os.path.join(job_folder, ANN_OUT)
        return send_from_directory(annotation_output, "annotation.zip", as_attachment=True)
    except Exception as e:
        return f"Error while uploading with error: {e}", 400
        # return 'File uploaded successfully'
        
        


@app.route('/segment', methods=['POST'])
def serve_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/get_frame', methods=['POST'])
def get_frame():
    r = request.get_json()
    job_id = int(r['jobId'])
    #TODO: add funtionality for giving frame images to client
    return
@app.route('/forward', methods=['GET'])
def forward_image():
    global current_image_index
    current_image_index = (current_image_index + 1) % len(IMAGES)
    return jsonify({"image_url1": f'/uploads/{IMAGES[current_image_index]}'})

@app.route('/backward', methods=['GET'])
def backward_image():
    global current_image_index
    current_image_index = (current_image_index - 1) % len(IMAGES)
    return jsonify({"image_url1": f'/uploads/{IMAGES[current_image_index]}'})

def start_client(job_id: int = 0):
    main_hub = create_main_hub(job_id=job_id, reseg_bool=True, reuse_output=True)
    main_hub.set_tracker()
    main_hub.track_key_frame_mask_objs = \
        main_hub.roarsegtracker.get_key_frame_to_masks()
    return main_hub

def get_frame_for_client(main_hub, frame: int = 0):
    end_frame_idx = main_hub.roarsegtracker.get_end_frame_idx()
    start_frame_idx = main_hub.roarsegtracker.get_start_frame_idx()
    img, img_mask = main_hub.get_frame(frame, end_frame_idx=end_frame_idx, 
                                       start_frame_idx=start_frame_idx)
    return img, img_mask

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(port=5000, debug=True)
