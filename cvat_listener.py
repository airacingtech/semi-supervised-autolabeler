import os
root = "/home/kevinchow/Downloads"
CVAT_PATH = os.path.join(root, 'updates.txt')

def remove_job_from_file(job_id):
    with open(CVAT_PATH, "r") as f:
        lines = f.readlines()
    filename = f"{job_id}.zip"
    with open(CVAT_PATH, "w") as f:
        for line in lines:
            if filename not in line:
                f.write(line)