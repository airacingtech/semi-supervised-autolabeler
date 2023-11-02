
###comment out if debugging in vscode is enabled
import eventlet

eventlet.monkey_patch()

from dotenv import load_dotenv

load_dotenv()

import traceback

import os

import dataset  # https://dataset.readthedocs.io/en/latest/

from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit

from cvat_listener import CVAT_PATH, remove_job_from_file
from roar_main import MainHub, arg_main, create_main_hub, save_main_hub
from tool.roar_tools import numpy_to_base64
from flask_celery import make_celery

# For the script that clears roar_annotations every couple hours.
import subprocess


PORT = os.environ.get("FLASK_RUN_PORT", 5000)
HOST = os.environ.get("FLASK_RUN_HOST", "label.roarart.online")
UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "/home/roar-apex/cvat/downloads")


app = Flask(__name__)

app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY")

app.config.update(
    CELERY_TASK_SERIALIZER="json",
    CELERY_RESULT_SERIALIZER="json",
    CELERY_ACCEPT_CONTENT=["json"],
    CELERY_ENABLE_UTC=True,
)
socketio = SocketIO(
    app, message_queue="amqp://", cors_allowed_origins=f"http://{HOST}:{PORT}"
)
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
STATUS_FAIL = 5
db = dataset.connect(os.environ.get("DB_URL"))
jobs_db = db["jobs"]


def get_jobs_from_cvat():
    job_files = []
    try:
        with open(CVAT_PATH, "r") as file:
            for line in file.readlines():
                if len(line) < 3:
                    continue
                job_zip = line.strip() # eg. 123.zip
                filepath = os.path.join(UPLOAD_FOLDER, job_zip)
                if os.path.exists(filepath):
                    job_files.append(job_zip)
    except Exception as e:
        print("Error reading " + CVAT_PATH)
        return []
    
    try:
        with open(CVAT_PATH, "w") as file:
            file.writelines(job_files)
    except Exception as e:
        print("Error reading " + CVAT_PATH)

    return [int(j[:-4]) for j in job_files]


for jobid in get_jobs_from_cvat():
    jobs_db.upsert(dict(id=jobid, status=STATUS_READY, msg=""), ["id"])


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
    return jsonify({"active": active, "scheduled": scheduled, "reserved": reserved})


@app.route("/clean-jobs")
def clean_jobs():
    jobs_db.delete()
    return "Cleaned up all jobs"


@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        r = request.form

        job_id = int(r.get("jobId"))
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
        frames = []

        if reseg_bool:
            frames = (
                r["frames"].split(",")
                if r.get("frames") is not None and r.get("frames") != ""
                else []
            )
            frames = [int(frame) for frame in frames]

            frames.sort()
        # if not request.files.get('file') and not reuse_annotation_output and reseg_bool:
        #     return 'No file part', 400

        else:
            file = request.files.get("file")
            if file is None or file.filename == "":
                filename = str("{}.zip".format(job_id))
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                if not os.path.exists(UPLOAD_FOLDER):
                    return jsonify(
                        {
                            "message": f"Specified UPLOAD_FOLDER in server does not exist {job_id}",
                            "task_id": -1,
                        }
                    )
            else:
                filename = str(file.filename)
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                if not os.path.exists(UPLOAD_FOLDER):
                    return jsonify(
                        {
                            "message": f"Specified UPLOAD_FOLDER in server does not exist {job_id}",
                            "task_id": -1,
                        }
                    )
                file.save(filepath)

        task = do_arg_main.delay(
            job_id, reseg_bool, reuse_annotation_output, threads, frames, socketroom=True
        )

        jobs_db.update(dict(id=job_id, status=STATUS_QUEUED, msg=r["jobType"]), ["id"])
        return jsonify({"message": f"Queued job {job_id}", "task_id": task.id})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"message": f"Failed to queue {job_id}: {e}", "task_id": -1})


@celery.task(name="upload")
def do_arg_main(
    job_id, reseg_bool, reuse_annotation_output, threads, frames, socketroom=None
):
    jobs_db.update(dict(id=job_id, status=STATUS_IN_PROGRESS), ["id"])
    try:
        arg_main(
            job_id=job_id,
            reseg_bool=reseg_bool,
            reuse_output=reuse_annotation_output,
            threads=threads,
            reseg_frames=frames,
            socketroom=socketroom
        )

        if job_id in TRACKERS:
            TRACKERS.pop(job_id)

        jobs_db.update(dict(id=job_id, status=STATUS_DONE, msg=""), ["id"])

    except Exception as e:
        print(f"Task failed: {e}")
        jobs_db.update(dict(id=job_id, status=STATUS_FAIL, msg=str(e)), ["id"])

    return job_id


@app.route("/download-annotation/<job_id>")
def download_annotation(job_id):
    job_folder = os.path.join(OUTPUT_FOLDER, job_id)
    annotation_output = os.path.join(job_folder, ANN_OUT)
    remove_job_from_file(job_id)

    jobs_db.delete(id=job_id, status=STATUS_DONE)

    return send_from_directory(
        annotation_output,
        "annotation.zip",
        as_attachment=True,
        download_name=f"annotation-{job_id}.zip",
    )


@app.route("/segment", methods=["POST"])
def serve_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/jobs-status", methods=["GET"])
def get_update():
    jobs = jobs_db.all()
    status_map = {
        STATUS_READY: "ready",
        STATUS_IN_PROGRESS: "in_progress",
        STATUS_DONE: "done",
        STATUS_QUEUED: "queued",
        STATUS_FAIL: "failed",
    }
    grouped_jobs = {
        "ready": [],
        "done": [],
        "in_progress": [],
        "queued": [],
        "failed": [],
    }

    curr_uploaded_jobs = jobs_db.find(status=STATUS_READY)
    updated_uploaded_jobs = get_jobs_from_cvat()
    new_jobs = [job for job in updated_uploaded_jobs if job not in curr_uploaded_jobs]
    for jobid in new_jobs:
        if not jobs_db.find_one(id=jobid):
            jobs_db.upsert(dict(id=jobid, status=STATUS_READY), ["id"])

    for job in jobs:
        if job["status"] in status_map:
            grouped_jobs[status_map[job["status"]]].append([job["id"], job["msg"]])
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
    job_id = -1

    try:
        job_id = int(r.get("jobId"))
        jobs_db.update(dict(id=job_id, status=STATUS_QUEUED, msg=""), ["id"])
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
        # delete_zip = bool(r.get('delete_zip'))
        frames = []

        if reseg_bool:
            frames = (
                r["frames"].split(",")
                if r.get("frames") is not None and r.get("frames") != ""
                else []
            )
            frames = [int(frame) for frame in frames]
            frames.sort()
        if TRACKERS.get(job_id) is not None:
            return

        tracker_object = create_main_hub(
            job_id=job_id, reseg_bool=reseg_bool, reuse_output=reuse_annotation_output,
          new_frames=frames
        )
        main_hub = tracker_object
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

        jobs_db.update(dict(id=job_id, status=STATUS_IN_PROGRESS, msg="0%"), ["id"])
    except Exception as e:
        print(f"Error in frame_track_start: {e}")
        jobs_db.update(dict(id=job_id, status=STATUS_FAIL, msg=str(e)), ["id"])


@socketio.on("disconnect")
def handle_disconnect():
    jobId = CLIENTS.get(request.sid)
    if jobId is not None:
        save_tracker(int(jobId))


@socketio.on("save_job")
def save_tracker(job_id):
    assert type(job_id) == int
    if TRACKERS.get(job_id) is not None:
        main_hub = TRACKERS[job_id]
        if type(main_hub) == MainHub:
            save_main_hub(main_hub)
        TRACKERS.pop(job_id)
        CLIENTS.pop(request.sid)
    job_folder = os.path.join(OUTPUT_FOLDER, str(job_id))

    jobs_db.update(dict(id=job_id, status=STATUS_DONE, msg=""), ["id"])


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

# Function that kills a subprocess
def kill_process(proc):
    print("Killing subprocess")
    proc.terminate()


if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    try:
        # Run the script that would clear the roar_annotations folder once in awhile.
        proc = subprocess.Popen(["/bin/bash", "server_cleanup.sh"])
    
        print(f"Running on {HOST}:{PORT}")
        socketio.run(app, host=HOST, port=PORT, debug=True)

    finally:
        kill_process(proc)