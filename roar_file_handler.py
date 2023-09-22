from operator import contains
import sys
import os
import zipfile
import re
class RoarFileHandler():
    """File handler to setup and move files from and for CVAT
    """
    def __init__(self, roar_path: str = "", downloads_path: str = ""):
        """init

        Args:
            roar_path (str, optional): base path to roar_annotations folder. Defaults to "".
            downloads_path (str, optional): path to downloads folder. Defaults to "".
        """
        self.roar_path = roar_path
        self.resegment_path = ""
        self.output_path = ""
        self.folder_path = ""
        self.annotations_output_path = ""
        self.downloads_path = downloads_path
    def make_folder(self, job_id: int = 0):
        """Make a folder for a job
        """
        folder_path = os.path.join(self.roar_path, str(job_id))
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        self.folder_path = folder_path
        resegment_folder_path = os.path.join(folder_path, "resegment_annotations")
        if not os.path.exists(resegment_folder_path):
            os.makedirs(resegment_folder_path)
        self.resegment_path = resegment_folder_path
        output_folder = os.path.join(folder_path, "output")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        self.output_path = output_folder
        annotations_output_folder = os.path.join(output_folder, "annotations_output")
        if not os.path.exists(annotations_output_folder):
            os.makedirs(annotations_output_folder)
        self.annotations_output_path = annotations_output_folder
        
        return folder_path
    def move_download_to_resegment(self, job_id: int = 0):
        """Move downloaded files to resegment folder
        """
        # download_folder = os.path.join(self.downloads_path, str(job_id))
        # resegment_folder = os.path.join(self.resegment_path, str(job_id))
        with zipfile.ZipFile(os.path.join(self.downloads_path, 
                                          "{}.zip".format(str(job_id))), "r") as zip_ref:
            for member in zip_ref.namelist():
                zip_ref.extract(member, self.resegment_path)
    def delete_zip(self, job_id: int = 0):
        rm = os.path.join(self.downloads_path, 
                                          "{}.zip".format(str(job_id)))
        if os.path.exists(rm):
            os.remove(rm)
    
    def move_download_to_init_segment(self, job_id: int = 0):
        """Move downloaded files to new job folder for first video segmentation tracking.
        Looks for folder called JOB_ID.zip in downloads folder.
        """
        # download_folder = os.path.join(self.downloads_path, str(job_id))
        # resegment_folder = os.path.join(self.resegment_path, str(job_id))
        pattern = re.compile(r"images/.")
        with zipfile.ZipFile(os.path.join(self.downloads_path, 
                                          "{}.zip".format(str(job_id))), "r") as zip_ref:
            for member in zip_ref.namelist():
                if pattern.match(member):
                    images_folder = os.path.join(self.folder_path, "images")
                    os.makedirs(images_folder, exist_ok=True)
                    assert os.path.exists(images_folder)
                    zip_ref.extract(member, self.folder_path)
                elif member == "annotations.xml":
                    annotation_file = os.path.join(self.folder_path, member)
                    # if not os.path.exists(annotation_file):
    
                    zip_ref.extract(member, self.folder_path)