# Startup
python roar_server.py

# Installation
1. Install Docker Engine (https://docs.docker.com/desktop/install/ubuntu/) NOT Docker Desktop.

# Notes
1. Always save the annotations in the CVAT server before exporting them.
2. Make sure to to use CVAT for video 1.1 and NOT CVAT for images 1.1 when exporting the annotations from the CVAT server.
3. Always reupload the zip folder to the seg and track server after re-exporting the annotations from the CVAT server as the annotations are not updated automatically even if the zip folder has the same name.
4. When uploading the annotations back onto CVAT uncheck "Convert masks to polygons".
5. When exporting a regsementation you don't have to save the images, just the annotations.
6. When exporting the dataset out of CVAT for fine-tuning, select the "COCO 1.0" format.

# Debugging
When debugging be aware that errors thrown will be shown in the web console of the browser (by pressing F12 and going to the console tab). The errors will be shown in the console tab of the browser and not in the terminal where the server is running. 

# TODO
- [ ] Remove BERT
- [ ] Add error message if user uploads a CVAT Images 1.1 file instead of a CVAT Video 1.1 file