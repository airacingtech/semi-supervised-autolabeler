import eventlet
eventlet.monkey_patch()

from dotenv import load_dotenv
load_dotenv()

import base64
import os
import os.path as osp
from itertools import groupby
from operator import itemgetter

import dataset # https://dataset.readthedocs.io/en/latest/

from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit

from cvat_listener import CVAT_PATH, remove_job_from_file
from roar_main import MainHub, arg_main, create_main_hub, save_main_hub
from tool.roar_tools import numpy_to_base64
from flask_celery import make_celery

PORT = os.environ.get("FLASK_RUN_PORT", 5000)
HOST = os.environ.get("FLASK_RUN_HOST", "label.roarart.online")
UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "/home/roar-apex/cvat/downloads")


app = Flask(__name__)

app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY")

app.config.update(
    CELERY_TASK_SERIALIZER = 'json',
    CELERY_RESULT_SERIALIZER = 'json',
    CELERY_ACCEPT_CONTENT=['json'],
    CELERY_ENABLE_UTC = True)
socketio = SocketIO(app, message_queue='amqp://', cors_allowed_origins=f"http://{HOST}:{PORT}")
celery = make_celery(app)
cors = CORS(app, expose_headers=["Content-Disposition"])
app.config["CORS_HEADERS"] = "Content-Type"


parent_folder = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FOLDER = os.path.join(parent_folder, "roar_annotations")
ANN_OUT = os.path.join("output", "annotations_output")
IMAGES = ["image1.jpg", "image2.jpg", "image3.jpg"]
TRACKERS = {}
CLIENTS = {}
current_image_index = 0

STATUS_READY = 1
STATUS_QUEUED = 2
STATUS_IN_PROGRESS = 3
STATUS_DONE = 4
db = dataset.connect(os.environ.get('DB_URL'))
jobs_db = db['jobs']

def get_jobs_from_cvat():
    try:
        with open(CVAT_PATH, "r") as file:
            return [int(line.strip()[:-4]) for line in file.readlines()]
    except Exception as e:
        print("Error reading " + CVAT_PATH)
        return []

for jobid in get_jobs_from_cvat():
    jobs_db.upsert(dict(id=jobid, status=STATUS_READY), ['id'])


@app.route("/")
def index():
    return render_template(
        "index.html", image_url=f"/uploads/{IMAGES[current_image_index]}"
    )

@app.route("/celery-status")
def celery_status():
    inspection = celery.control.inspect()
    scheduled = inspection.scheduled()
    active = inspection.active()
    reserved = inspection.reserved()
    return jsonify({'active': active, 'scheduled': scheduled, 'reserved':reserved})

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
        else:
            file = request.files.get("file")
            if file is None or file.filename == "":
                filename = str("{}.zip".format(job_id))
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                if not os.path.exists(UPLOAD_FOLDER):
                    return "Specified UPLOAD_FOLDER in server does not exist", 400

            else:
                filename = str(file.filename)
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                if not os.path.exists(UPLOAD_FOLDER):
                    return "Specified UPLOAD_FOLDER in server does not exist", 400
                file.save(filepath)

        task = do_arg_main.delay(job_id, reseg_bool, reuse_annotation_output, threads, frames, delete_zip)

        jobs_db.update(dict(id=job_id, status=STATUS_QUEUED), ['id'])
        return jsonify({"message": f"Queued job {job_id}", "task_id": task.id })

    except Exception as e:
        return f"Error while uploading with error: {e}", 400

@celery.task(name="upload")
def do_arg_main(job_id, reseg_bool, reuse_annotation_output, threads, frames, delete_zip):
    jobs_db.update(dict(id=job_id, status=STATUS_IN_PROGRESS), ['id'])
    try:
        out = arg_main(
            job_id=job_id,
            reseg_bool=reseg_bool,
            reuse_output=reuse_annotation_output,
            threads=threads,
            reseg_frames=frames,
            delete_zip=delete_zip,
        )

        if job_id in TRACKERS:
            TRACKERS.pop(job_id)

        # socketio.emit("upload_response", {"status": "success", "job_id": job_id})
    except Exception as e:
        pass
        # socketio.emit("upload_response", {"status": "fail", "job_id": job_id})

    jobs_db.update(dict(id=job_id, status=STATUS_DONE), ['id'])
    return job_id


@app.route("/download-annotation/<job_id>")
def download_annotation(job_id):
    job_folder = os.path.join(OUTPUT_FOLDER, job_id)
    annotation_output = os.path.join(job_folder, ANN_OUT)
    remove_job_from_file(job_id)

    jobs_db.delete(id=job_id, status=STATUS_DONE)

    return send_from_directory(annotation_output, "annotation.zip", as_attachment=True, download_name=f"annotation-{job_id}.zip")
    

@app.route("/segment", methods=["POST"])
def serve_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/jobs-status", methods=["GET"])
def get_update():
    jobs = jobs_db.all()
    status_map = {STATUS_READY: 'ready', STATUS_IN_PROGRESS: 'in_progress', STATUS_DONE: 'done', STATUS_QUEUED: 'queued'}
    grouped_jobs = {'ready': [], 'done': [], 'in_progress': [], 'queued':[]}

    curr_uploaded_jobs = jobs_db.find(status=STATUS_READY)
    updated_uploaded_jobs = get_jobs_from_cvat()
    new_jobs = [job for job in updated_uploaded_jobs if job not in curr_uploaded_jobs]
    for jobid in new_jobs:
        if not jobs_db.find_one(id=jobid):
            jobs_db.upsert(dict(id=jobid, status=STATUS_READY), ['id'])

    for job in jobs:
        if job['status'] in status_map:
            grouped_jobs[status_map[job['status']]].append(job['id'])
    return jsonify(grouped_jobs)



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


@socketio.on("frame_track_start")
def assign_tracker(formData):
    r = formData
    try: 
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
    except Exception as e:
        print(f"Error in frame_track_start: {e}")



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
    # remove_job_from_file(CVAT_PATH, jobId)
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
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    print(f"Running on {HOST}:{PORT}")
    socketio.run(app, host=HOST, port=PORT, debug=False)
