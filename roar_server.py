import eventlet

eventlet.monkey_patch()
import base64
import os
import os.path as osp
# import re

from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit

from cvat_listener import CVAT_PATH
from roar_main import MainHub, arg_main, create_main_hub, save_main_hub
from tool.roar_tools import numpy_to_base64
from flask_celery import make_celery

app = Flask(__name__)
parent_folder = os.path.dirname(os.path.abspath(__file__))
keypath = osp.join(parent_folder, "agent.key")
secret_key = "no key found"
with open(keypath, "r") as f:
    secret_key = f.read()
app.config["SECRET_KEY"] = secret_key  # Change this to a random and secure value
socketio = SocketIO(app, cors_allowed_origins="https://label.roarart.online:5000")
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"
UPLOAD_FOLDER = "/home/roar-apex/cvat/downloads"

OUTPUT_FOLDER = os.path.join(parent_folder, "roar_annotations")
ANN_OUT = os.path.join("output", "annotations_output")
IMAGES = ["image1.jpg", "image2.jpg", "image3.jpg"]
TRACKERS = {}
CLIENTS = {}

current_image_index = 0


app.config.update(
    CELERY_BROKER_URL = "amqp://localhost//",
    CELERY_RESULT_BACKEND = "rpc://"
)
socketio = SocketIO(app, message_queue='amqp://')
celery = make_celery(app)


def remove_job_from_file(filepath, job_id):
    with open(filepath, "r") as f:
        lines = f.readlines()
    filename = f"{job_id}.zip"
    with open(filepath, "w") as f:
        for line in lines:
            if filename not in line:
                f.write(line)


@app.route("/")
def index():
    return render_template(
        "roar_webpage_updated.html", image_url=f"/uploads/{IMAGES[current_image_index]}"
    )


@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        r = request.form

        job_id = int(r.get("jobId"))
        if TRACKERS.get(job_id) is not None:
            return f"Tracking job for {job_id} already in progress", 400
        else:
            TRACKERS[job_id] = job_id
        if type(r.get("threads")) is str or type(r.get("threads")) is int:
            threads = r["threads"]
            if threads == "":
                threads = 1
            else:
                threads = int(r["threads"])
        else:
            threads = 1
        reseg_bool = not (r["jobType"] == "initial segmentation")
        # on_pattern = r'([O|o][n|N])'
        reuse_annotation_output = bool(r.get("reuseAnnotation"))
        delete_zip = bool(r.get("delete_zip"))
        frames = []

        if reseg_bool:
            frames = (
                r["frames"].split(",")
                if r.get("frames") is not None and r.get("frames") != ""
                else []
            )
            frames = [int(frame) for frame in frames]
        # if not request.files.get('file') and not reuse_annotation_output and reseg_bool:
        #     return 'No file part', 400
        else:
            file = request.files.get("file")
            if file is None or file.filename == "":
                filename = str("{}.zip".format(job_id))
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                if not os.path.exists(UPLOAD_FOLDER):
                    return "Specified UPLOAD_FOLDER in server does not exist", 400
                # file.save(filepath)

            # elif file.filename == '' and not reuse_annotation_output:
            #     return 'No selected file', 400
            else:
                filename = str(file.filename)
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                if not os.path.exists(UPLOAD_FOLDER):
                    return "Specified UPLOAD_FOLDER in server does not exist", 400
                file.save(filepath)

        do_arg_main.delay(job_id, reseg_bool, reuse_annotation_output, threads, frames, delete_zip)
        return "Started job", 200

    except Exception as e:
        return f"Error while uploading with error: {e}", 400

@celery.task(name="upload")
def do_arg_main(job_id, reseg_bool, reuse_annotation_output, threads, frames, delete_zip):
    print("inside do_arg_main")
    socketio = SocketIO(message_queue="amqp://")
    arg_main(
        job_id=job_id,
        reseg_bool=reseg_bool,
        reuse_output=reuse_annotation_output,
        threads=threads,
        reseg_frames=frames,
        delete_zip=delete_zip,
    )
    job_folder = os.path.join(OUTPUT_FOLDER, str(job_id))
    annotation_output = os.path.join(job_folder, ANN_OUT)
    remove_job_from_file(CVAT_PATH, job_id)
    TRACKERS.pop(job_id)
    zip_file = os.path.join(annotation_output, "annotation.zip")
    output = (
        send_from_directory(annotation_output, "annotation.zip", as_attachment=True)
        if os.path.exists(zip_file)
        else "No File"
    )
    socketio.emit("upload_response", {"output": output})
    # emit("upload_response", {"output": output})

    

@app.route("/segment", methods=["POST"])
def serve_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/getUpdate", methods=["GET"])
def get_update():
    try:
        with open(CVAT_PATH, "r") as file:
            content = file.read()
        return jsonify(content=content)
    except Exception as e:
        return jsonify(error=str(e)), 500


def start_client(job_id: int = 0):
    main_hub = create_main_hub(job_id=job_id, reseg_bool=True, reuse_output=True)
    main_hub.set_tracker()
    main_hub.track_key_frame_mask_objs = (
        main_hub.roarsegtracker.get_key_frame_to_masks()
    )
    return main_hub


def get_frame_for_client(main_hub, frame: int = 0):
    end_frame_idx = main_hub.roarsegtracker.get_end_frame_idx()
    start_frame_idx = main_hub.roarsegtracker.get_start_frame_idx()
    img, img_mask = main_hub.get_frame(
        frame, end_frame_idx=end_frame_idx, start_frame_idx=start_frame_idx
    )
    return img, img_mask


# #socketio
@socketio.on("frame_track_start")
def assign_tracker(formData):
    # file_test = request.files.get('file')
    # r = request.get_json(force=True)
    r = formData
    print(f"r: {r}")
    job_id = int(r.get("jobId"))

    if type(r.get("threads")) is str or type(r.get("threads")) is int:
        threads = r["threads"]
        if threads == "":
            threads = 1
        else:
            threads = int(r["threads"])
    else:
        threads = 1
    reseg_bool = not (r["jobType"] == "initial segmentation")
    # on_pattern = r'([O|o][n|N])'
    reuse_annotation_output = bool(r.get("reuseAnnotation"))
    delete_zip = bool(r.get("delete_zip"))
    frames = []

    if reseg_bool:
        frames = (
            r["frames"].split(",")
            if r.get("frames") is not None and r.get("frames") != ""
            else []
        )
        frames = [int(frame) for frame in frames]

    if TRACKERS.get(job_id) is not None:
        return

    tracker_object = create_main_hub(
        job_id=job_id, reseg_bool=reseg_bool, reuse_output=reuse_annotation_output
    )
    main_hub = tracker_object
    main_hub.set_tracker()
    main_hub.track_key_frame_mask_objs = (
        main_hub.get_roar_seg_tracker().get_key_frame_to_masks()
    )
    end_frame_idx = main_hub.get_roar_seg_tracker().get_end_frame_idx()
    start_frame_idx = main_hub.get_roar_seg_tracker().get_start_frame_idx()
    TRACKERS[job_id] = tracker_object
    CLIENTS[request.sid] = job_id
    emit(
        "post_frame_range",
        {"type": "int", "start_frame": start_frame_idx, "end_frame": end_frame_idx},
        room=request.sid,
    )


@socketio.on("disconnect")
def handle_disconnect():
    jobId = CLIENTS.get(request.sid)
    if jobId is not None:
        save_tracker(int(jobId))


@socketio.on("save_job")
def save_tracker(jobId):
    assert type(jobId) == int
    if TRACKERS.get(jobId) is not None:
        main_hub = TRACKERS[jobId]
        if type(main_hub) == MainHub:
            save_main_hub(main_hub)
        TRACKERS.pop(jobId)
        CLIENTS.pop(request.sid)
    job_folder = os.path.join(OUTPUT_FOLDER, str(jobId))
    annotation_output = os.path.join(job_folder, ANN_OUT)
    remove_job_from_file(CVAT_PATH, jobId)
    zip_file = os.path.join(annotation_output, "annotation.zip")
    if os.path.exists(zip_file):
        # Convert the file into a blob or a data URL (Base64 encoding) and send it
        with open(zip_file, "rb") as f:
            file_data = f.read()
            b64_encoded_data = base64.b64encode(file_data).decode("utf-8")
            emit(
                "post_annotation",
                {"type": "blob", "content": b64_encoded_data},
                room=request.sid,
            )
    else:
        emit(
            "post_annotation", {"type": "text", "content": "No File"}, room=request.sid
        )


@socketio.on("frame_value")
def get_frame(response):
    job_id = response["job_id"]
    frame = response["frame"]
    tracker = TRACKERS.get(job_id)
    if tracker is not None:
        end_frame_idx = tracker.get_roar_seg_tracker().get_end_frame_idx()
        start_frame_idx = tracker.get_roar_seg_tracker().get_start_frame_idx()
        img, img_mask = tracker.get_frame(
            frame, end_frame_idx=end_frame_idx, start_frame_idx=start_frame_idx
        )
        img_string = numpy_to_base64(img)
        img_mask_string = numpy_to_base64(img_mask)
        emit(
            "post_images",
            {"type": "image", "img": img_string, "img_mask": img_mask_string},
            room=request.sid,
        )


if __name__ == "__main__":
    PORT = os.environ.get("PORT", 5000)
    HOST = os.environ.get("HOST", "label.roarart.online")
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    print(f"Running on {HOST}:{PORT}")
    socketio.run(app, host=HOST, port=PORT, debug=False)
