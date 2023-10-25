import gc
import json
import os
import threading
import time
from concurrent.futures.thread import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import torch
from matplotlib.sankey import DOWN
from tqdm import tqdm

import tool.roar_tools as rt
from model_args import aot_args, sam_args, segtracker_args
from roar_file_handler import RoarFileHandler
from RoarSegTracker import RoarSegTracker

# from PIL import Image
# from aot_tracker import _palette
# import cv2
# import imageio
# import matplotlib.pyplot as plt
# from scipy.ndimage import binary_dilation


DOWNLOADS_PATH = "/home/roar-apex/cvat/downloads"
# DOWNLOADS_PATH = "C:/Users/chowm/Downloads"
sam_args["generator_args"] = {
    "points_per_side": 30,
    "pred_iou_thresh": 0.8,
    "stability_score_thresh": 0.9,
    "crop_n_layers": 1,
    "crop_n_points_downscale_factor": 2,
    "min_mask_region_area": 200,
}

segtracker_args = {
    "sam_gap": 7860,  # the interval to run sam to segment new objects
    "min_area": 200,  # minimal mask area to add a new mask as a new object
    "max_obj_num": 255,  # maximal object number to track in a video
    "min_new_obj_iou": 0.8,  # the area of a new object in the background should > 80%
}


class MainHub:
    """
    Main hub class for RoarSegTracker

    Class structure of main for use of RoarSegTracker
    """

    MAX_WORKERS = 3

    def __init__(
        self,
        segtracker_args={},
        sam_args={},
        aot_args={},
        photo_dir="",
        annotation_dir="",
        output_dir="",
    ):
        self.segtracker_args = segtracker_args
        self.sam_args = sam_args
        self.aot_args = aot_args
        self.photo_dir = photo_dir
        self.annotation_dir = annotation_dir
        self.output_dir = output_dir
        # key frame to dictionary of obj id to obj
        self.track_key_frame_mask_objs: dict[int : dict[int, rt.MaskObject]] = {}
        self.roarsegtracker = RoarSegTracker(
            self.segtracker_args, self.sam_args, self.aot_args
        )
        self.roarsegtracker.restart_tracker()
        # toggle use sam gap here
        self.use_sam_gap = False
        self.store = False
        # multithreading
        self.max_workers = MainHub.MAX_WORKERS
        self.lock = threading.Lock()
        # reseg frame-by-frame track option
        self.new_frames: list[int] = []
        self.reseg_idx: int = 0
        self.key_frame_path: str = ""
        self.root_path: str = ""
        # options
        self.id_countour = False  # countour masks for img w/ mask overlay feature

    def setup(self):
        # TODO: add setup to modulate main tracking methods?
        return

    def get_segmentations(self, key_frame_idx=0):
        origin_merged_mask = self.roarsegtracker.create_origin_mask(key_frame_idx)
        return origin_merged_mask

    def set_tracker(self, annontation_dir=""):
        # TODO: set tracker values with anything it needs before segmentation tracking

        if annontation_dir != "":
            self.annotation_dir = annontation_dir
        self.roarsegtracker.start_seg_tracker_for_cvat(
            annotation_dir=self.annotation_dir
        )

    def set_root(self, path: str):
        self.root_path = path

    def get_root(self) -> str:
        return self.root_path

    def store_key_frames(
        self, key_frames: list[int] = [], reseg_idx: int = 0, output_dir: str = ""
    ):
        list_vars = [key_frames, reseg_idx]
        with open(output_dir, "w") as f:
            json.dump(list_vars, f, indent=4)

    def get_key_frames(self, output_dir: str = ""):
        with open(output_dir, "r") as f:
            list_vars = json.load(f)
            return list_vars

    def set_key_frame_path(self, key_frame_path: str):
        self.key_frame_path = key_frame_path

    def get_key_frame_path(self):
        return self.key_frame_path

    def set_reseg_idx(self, reseg_idx: int):
        self.reseg_idx = reseg_idx

    def get_reseg_idx(self) -> int:
        return self.reseg_idx

    def set_new_frames(self, new_frames: list[int]):
        self.new_frames = new_frames

    def get_new_frames(self):
        return self.new_frames

    def store_tracker(self, frame=""):
        tracker_serial_data = RoarSegTracker.store_data(self.roarsegtracker)
        folder_name = "tracker_data"
        file_name = "tracker_data_frame_" + str(frame) + "_time_" + str(datetime.now())
        folder_path = os.path.join(self.output_dir, folder_name)
        file_path = os.path.join(folder_path, file_name)

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        with open(file_path, "wb") as outfile:
            outfile.write(tracker_serial_data)

    def get_tracker(self, frame="", time=""):
        folder_name = "tracker_data"
        file_name = "tracker_data_frame_" + str(frame) + "_time_" + time
        folder_path = os.path.join(self.output_dir, folder_name)
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "rb") as outfile:
            tracker_serial_data = outfile.read()
        self.roarsegtracker = RoarSegTracker.load_data(tracker_serial_data)

    def get_roar_seg_tracker(self) -> RoarSegTracker:
        return self.roarsegtracker

    def setup_reseg_key_frames(self, new_frames: list[int], past_frames: list[int]):
        try:
            combined_frames = past_frames + new_frames
            dupes = {}
            for frame in combined_frames:
                dupes[frame] = 1
            frames = list(dupes.keys())
            frames.sort()
            return frames
        except Exception as e:
            print(f"Error: {e}")

    def remake_key_frames(
        self,
        roarsegtracker: RoarSegTracker,
        new_frames: list[int],
        past_frames: list[int],
    ):
        """
        Remakes the key_frame to mask_object dictionary by removing frame that would be generated again by new_frames
        declared for resegmentation
        Arguments:
            roarsegtracker (RoarSegTracker): The tracker associated with this tracking run
            new_frames (list[int]): list of new key frames declared for resegmentation
            past_frames (list[int]): old key frames from previous tracking runs
        Returns:
            combined_frames (list[int]): arr of combined_frames
        """
        combined_frames = self.setup_reseg_key_frames(new_frames, past_frames)
        end_frame_idx = roarsegtracker.get_end_frame_idx()
        new_frame_queue = new_frames[:]
        for i in range(len(combined_frames)):
            if len(new_frame_queue) == 0:
                break
            elif combined_frames[i] == new_frame_queue[0]:
                new_frame_queue.pop(0)
                if i + 1 < len(combined_frames):
                    end = combined_frames[i + 1]
                else:
                    end = end_frame_idx
                for frame in range(combined_frames[i] + 1, end + 1):
                    if self.track_key_frame_mask_objs.get(frame) is not None:
                        self.track_key_frame_mask_objs.pop(frame)

        return combined_frames

    def get_frame(
        self, frame: int = -1, end_frame_idx: int = -1, start_frame_idx: int = -1
    ):
        """given desired frame, return image at desired frame with generated masks

        Args:
            frame (int, optional): int value of desired frame. Defaults to -1.
            end_frame_idx (int, optional): edge case for when to stop looking for frames. Defaults to -1.
            start_frame_idx (int): edge case for when to start looking for frames. Defaults to -1
            reseg (bool) : tells object if its a resegmentation task.
        Returns:
            [img, img_mask]: list of 3 dimensional img (RGB) for given frame as well as the image with mask overlay
        """
        img_dim = self.roarsegtracker.get_img_dim()
        if start_frame_idx > frame or frame > end_frame_idx:
            img = np.array([np.zeros(img_dim) for i in range(3)])
            return img, img
        img = rt.get_image(self.photo_dir, frame)
        # see if frame is already made or exists
        tWalker: int = self.roarsegtracker.get_curr_key_frame()
        min_dist = float("inf") if frame - tWalker < 0 else frame - tWalker

        if self.track_key_frame_mask_objs.get(frame) is None:
            # find closest frame to desired frame
            if frame - 1 != tWalker:
                for k, v in self.track_key_frame_mask_objs.items():
                    dist = frame - k
                    if dist < min_dist and dist >= 0:
                        min_dist = dist
                        tWalker = k

            # set key frame
            roartracker = RoarSegTracker(
                self.segtracker_args, self.sam_args, self.aot_args
            )
            key_frame = tWalker
            key_frame_arr = [key_frame]
            label_to_color = self.roarsegtracker.get_label_to_color()
            roartracker.restart_tracker()
            roartracker.setup_tracker_by_values(
                key_frame_to_masks={
                    key_frame: self.track_key_frame_mask_objs[key_frame]
                },
                start_frame_idx=key_frame,
                end_frame_idx=end_frame_idx,
                img_dim=img_dim,
                label_to_color=label_to_color,
                key_frame_arr=key_frame_arr,
            )
            self.track_set_frames(
                roar_seg_tracker=roartracker,
                key_frames=key_frame_arr,
                end_frame_idx=frame,
            )

        self.roarsegtracker.set_curr_key_frame(frame)
        img_mask = rt.make_img_with_masks(
            id_to_mask_objs=self.track_key_frame_mask_objs[frame],
            img=img,
            id_countour=self.id_countour,
        )

        # set everything up given last generated frame

        # while the desired frame still none, generate frames

        # return list of two images one

        return img, img_mask

    def track_set_frames(
        self, roar_seg_tracker, key_frames: list[int] = [], end_frame_idx: int = 0
    ):
        """Given end frame index, as well as a list of
        key frames, track only the portion starting from first key frame idx to end_frame_index.
        Destructive method so copy key frame list if needed.

        Args:
            roar_seg_tracker: RoarSegTracker w/ custom presets for given frame iteration
            key_frames :List of key frames in sorted increasing order(manually annotated). Defaults to 0.
            end_frame_idx (int, optional): ending frame index; must be equal to or greater than
            greatest key frame index value. Defaults to 0.
        """
        next_key_frame = key_frames.pop(0)
        assert next_key_frame < end_frame_idx
        frames = list(range(next_key_frame, end_frame_idx + 1))
        curr_frame = frames[0]
        with torch.cuda.amp.autocast():
            for curr_frame in tqdm(
                frames, "Processing frames {} to {}".format(curr_frame, frames[-1])
            ):
                frame = rt.get_image(self.photo_dir, curr_frame)
                if curr_frame == next_key_frame:
                    # start with new tracker for every keyframe to reset weights
                    roar_seg_tracker.new_tracker()
                    roar_seg_tracker.restart_tracker()
                    # cuda
                    torch.cuda.empty_cache()
                    gc.collect()

                    pred_mask = roar_seg_tracker.create_origin_mask(
                        key_frame_idx=curr_frame
                    )
                    roar_seg_tracker.set_curr_key_frame(next_key_frame)
                    with self.lock:
                        self.track_key_frame_mask_objs[
                            curr_frame
                        ] = roar_seg_tracker.get_key_frame_to_masks()[curr_frame]

                    # cuda
                    torch.cuda.empty_cache()
                    gc.collect()

                    roar_seg_tracker.add_reference_with_label(frame, pred_mask)

                    if len(key_frames) > 0:
                        next_key_frame = key_frames.pop(0)

                elif curr_frame % self.roarsegtracker.sam_gap == 0 and self.use_sam_gap:
                    pass

                else:
                    # TODO: create mask object from pred_mask
                    pred_mask = roar_seg_tracker.track(frame, update_memory=True)

                    test_pred_mask = np.unique(pred_mask)
                    with self.lock:
                        self.track_key_frame_mask_objs[
                            curr_frame
                        ] = roar_seg_tracker.create_mask_objs_from_pred_mask(
                            pred_mask, curr_frame
                        )

                # cuda
                torch.cuda.empty_cache()
                gc.collect()

    def resegment_track(
        self,
        past_key_frames: list[int] = [],
        new_frames: list[int] = [],
        multithreading: bool = False,
        custom_end_frame_idx: int = 0,
    ):
        """Resegmentation tracker function. Expects that multi_track or
        track has been run on CVAT job before calling this function.
        Takes new frames inputed by user which are resegmented by user in
        CVAT (mask objects for specified frame is different from previously
        generated mask objects generated by TRACK or MULTI_TRACK).

        Args:
            past_key_frames (list[int], optional): past key_frame_arr
            generated by TRACK or MULTI_TRACK. Defaults to [].
            new_frames (list[int], optional): List of new resegmented
            frames from CVAT specified by User Input. Defaults to [].
            multithreading (bool, optional): Utilize multiple threads
            according to SELF.MAX_WORKERS. Defaults to False.
        """
        assert len(new_frames) > 0
        if not multithreading:
            self.max_workers = 1
        self.set_tracker(annontation_dir=self.annotation_dir)
        # key_frame_queue = self.roarsegtracker.get_key_frame_arr()[:]
        end_frame_idx = (
            self.roarsegtracker.end_frame_idx
            if custom_end_frame_idx == 0
            else custom_end_frame_idx
        )
        # past_key_frames.append(end_frame_idx) #last key frame goes to end of video
        self.track_key_frame_mask_objs = self.roarsegtracker.get_key_frame_to_masks()
        # new_frame = new_frames.pop(0)
        # for frame in past_key_frames:
        #     if frame > new_frame:
        #         try:
        #             key_frame_arr = [new_frame]
        #             roartracker = RoarSegTracker(self.segtracker_args, self.sam_args, self.aot_args)
        #             roartracker.restart_tracker()
        #             roartracker.setup_tracker_by_values(key_frame_to_masks=\
        #                 {new_frame: self.track_key_frame_mask_objs[new_frame]},
        #                                                 start_frame_idx=new_frame, end_frame_idx=frame,
        #                                                 img_dim=self.roarsegtracker.get_img_dim(),
        #                                                 label_to_color=self.roarsegtracker.get_label_to_color(),
        #                                                 key_frame_arr=key_frame_arr)
        #             self.track_set_frames(roartracker,
        #                             key_frames=key_frame_arr, end_frame_idx=frame)
        #             if len(new_frames) > 0:
        #                 new_frame = new_frames.pop(0)
        #             else:
        #                 break
        #         except Exception as e:
        #             print(e)
        #             # print(traceback.format_exc())
        #             print(f"Skipping frame {frame} due to exception")
        #             continue

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            try:
                new_frame = new_frames.pop(0)
                for frame in tqdm(
                    past_key_frames, "Processing new frame to past key frames: "
                ):
                    if frame > new_frame:
                        key_frame_arr = [new_frame]
                        roartracker = RoarSegTracker(
                            self.segtracker_args, self.sam_args, self.aot_args
                        )
                        roartracker.restart_tracker()
                        roartracker.setup_tracker_by_values(
                            key_frame_to_masks={
                                new_frame: self.track_key_frame_mask_objs[new_frame]
                            },
                            start_frame_idx=new_frame,
                            end_frame_idx=frame,
                            img_dim=self.roarsegtracker.get_img_dim(),
                            label_to_color=self.roarsegtracker.get_label_to_color(),
                            key_frame_arr=key_frame_arr,
                        )

                        executor.submit(
                            self.track_set_frames, roartracker, key_frame_arr, frame
                        )
                        if len(new_frames) > 0:
                            new_frame = new_frames.pop(0)
                        else:
                            new_frame = None
                            break
                if new_frame is not None:
                    key_frame_queue = [new_frame] + new_frames
                    assert end_frame_idx > key_frame_queue[-1]
                    key_frame_queue.append(end_frame_idx)
                    for i in tqdm(
                        range(len(key_frame_queue) - 1),
                        "Processing new frames to end frame: ",
                    ):
                        key_frame = key_frame_queue[i]
                        end_frame_idx = key_frame_queue[i + 1]
                        if end_frame_idx != key_frame_queue[-1]:
                            end_frame_idx -= 1
                        key_frame_arr = [key_frame]
                        roartracker = RoarSegTracker(
                            self.segtracker_args, self.sam_args, self.aot_args
                        )
                        roartracker.restart_tracker()
                        roartracker.setup_tracker_by_values(
                            key_frame_to_masks={
                                key_frame: self.track_key_frame_mask_objs[key_frame]
                            },
                            start_frame_idx=key_frame,
                            end_frame_idx=end_frame_idx,
                            img_dim=self.roarsegtracker.get_img_dim(),
                            label_to_color=self.roarsegtracker.get_label_to_color(),
                            key_frame_arr=key_frame_arr,
                        )

                        # thread = threading.Thread(target=self.track_set_frames, args=(key_frame_arr, end_frame_idx, roartracker))
                        executor.submit(
                            self.track_set_frames,
                            roartracker,
                            key_frame_arr,
                            end_frame_idx,
                        )
            except Exception as e:
                print(f"An exception occurred: {e}")

    def multi_trackers(self):
        """Multithreading performance works by starting a thread for
        each given key frame up to MAX_WORKERS at a time.
        """
        self.set_tracker(annontation_dir=self.annotation_dir)
        key_frame_queue = self.roarsegtracker.get_key_frame_arr()[:]
        end_frame_idx = self.roarsegtracker.end_frame_idx
        key_frame_queue.append(end_frame_idx)
        key_frame_to_masks = self.roarsegtracker.get_key_frame_to_masks()

        img_dim = self.roarsegtracker.get_img_dim()
        label_to_color = self.roarsegtracker.get_label_to_color()
        threads = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            try:
                for i in tqdm(
                    range(len(key_frame_queue) - 1),
                    "Making threads {}".format(self.max_workers),
                ):
                    key_frame = key_frame_queue[i]
                    end_frame_idx = key_frame_queue[i + 1]
                    if end_frame_idx != key_frame_queue[-1]:
                        end_frame_idx -= 1
                    key_frame_arr = [key_frame]
                    roartracker = RoarSegTracker(
                        self.segtracker_args, self.sam_args, self.aot_args
                    )
                    roartracker.restart_tracker()
                    roartracker.setup_tracker_by_values(
                        key_frame_to_masks={key_frame: key_frame_to_masks[key_frame]},
                        start_frame_idx=key_frame,
                        end_frame_idx=end_frame_idx,
                        img_dim=img_dim,
                        label_to_color=label_to_color,
                        key_frame_arr=key_frame_arr,
                    )
                    # thread = threading.Thread(target=self.track_set_frames, args=(key_frame_arr, end_frame_idx, roartracker))
                    executor.submit(
                        self.track_set_frames, roartracker, key_frame_arr, end_frame_idx
                    )
                    # thread.start()
                    # threads.append(thread)
            except Exception as e:
                print(f"An exception occurred: {e}")

        # for i in tqdm(range(len(key_frame_queue) - 1), "Making threads {}".format(self.max_workers)):
        #        key_frame = key_frame_queue[i]
        #        end_frame_idx = key_frame_queue[i + 1]
        #        key_frame_arr = [key_frame]
        #        roartracker = RoarSegTracker(self.segtracker_args, self.sam_args, self.aot_args)
        #        roartracker.restart_tracker()
        #        roartracker.setup_tracker_by_values(key_frame_to_masks={key_frame: key_frame_to_masks[key_frame]},
        #                                            start_frame_idx=key_frame, end_frame_idx=end_frame_idx,
        #                                            img_dim=img_dim, label_to_color=label_to_color,
        #                                            key_frame_arr=key_frame_arr)
        #        thread = threading.Thread(target=self.track_set_frames, args=(roartracker, key_frame_arr, end_frame_idx))
        #        thread.start()
        #        threads.append(thread)

        # for thread in threads:
        #    thread.join()

    def track(self):
        """
        Main function for tracking
        """

        # start at first annotated frame
        self.set_tracker(annontation_dir=self.annotation_dir)
        start_frame = self.roarsegtracker.start_frame_idx
        end_frame = self.roarsegtracker.end_frame_idx
        key_frame_queue = self.roarsegtracker.get_key_frame_arr()[:]
        next_key_frame = key_frame_queue.pop(0)
        curr_frame = self.roarsegtracker.get_key_frame_arr()[0]
        if curr_frame != next_key_frame:
            while curr_frame > next_key_frame:
                next_key_frame = key_frame_queue.pop(0)
            curr_frame = next_key_frame
        self.roarsegtracker.set_curr_key_frame(curr_frame)

        # cuda
        torch.cuda.empty_cache()
        gc.collect()
        frames = list(range(curr_frame, end_frame + 1))
        with torch.cuda.amp.autocast():
            for curr_frame in tqdm(frames, "Processing frames... "):
                # if curr_frame % 2187 == 0:

                #     print("day of reckoning")
                frame = rt.get_image(self.photo_dir, curr_frame)
                if curr_frame == next_key_frame:
                    # segment
                    # get new mask and tracking objects
                    self.roarsegtracker.new_tracker()
                    self.roarsegtracker.restart_tracker()
                    torch.cuda.empty_cache()
                    gc.collect()

                    pred_mask = self.get_segmentations(key_frame_idx=curr_frame)
                    # if curr_frame == 2187:
                    #     plt.imshow(pred_mask)
                    #     plt.show()

                    self.roarsegtracker.set_curr_key_frame(next_key_frame)

                    # TODO: create mask object from pred_mask
                    self.track_key_frame_mask_objs[
                        curr_frame
                    ] = self.roarsegtracker.get_key_frame_to_masks()[curr_frame]

                    # TODO: add curr version of tracker to serialized save file in case of mem crash or seg fault
                    # cuda
                    torch.cuda.empty_cache()
                    gc.collect()
                    test_pred = np.unique(pred_mask)
                    self.roarsegtracker.add_reference_with_label(frame, pred_mask)
                    if len(key_frame_queue) > 0:
                        next_key_frame = key_frame_queue.pop(0)
                elif curr_frame % self.roarsegtracker.sam_gap == 0 and self.use_sam_gap:
                    # resegment on sam gap
                    pass
                else:
                    # TODO: create mask object from pred_mask
                    pred_mask = self.roarsegtracker.track(frame, update_memory=True)
                    # if curr_frame > 2187:
                    #     plt.imshow(pred_mask)
                    #     plt.show()

                    test_pred_mask = np.unique(pred_mask)
                    self.track_key_frame_mask_objs[
                        curr_frame
                    ] = self.roarsegtracker.create_mask_objs_from_pred_mask(
                        pred_mask, curr_frame
                    )
                    # if curr_frame == 88:
                    #     print('88')
                if self.store:
                    self.store_tracker(frame=str(curr_frame))
                # cuda
                torch.cuda.empty_cache()
                gc.collect()

    def save_annotations(self) -> str:
        """Save the annotations to designate output dir

        Args: None

        Returns: str of file path to zip file containing annotations.xml"""
        folder_name = "annotations_output"
        folder_path = os.path.join(self.output_dir, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_name = "annotations.xml"
        file_path = os.path.join(folder_path, file_name)
        if not os.path.exists(file_path):
            with open(file_path, "w") as outfile:
                outfile.write("set RoarSegTracker use\n")

        zip_fp = rt.masks_to_xml(
            self.track_key_frame_mask_objs,
            self.roarsegtracker.get_start_frame_idx(),
            self.roarsegtracker.get_end_frame_idx(),
            file_path,
        )
        return zip_fp

    def tune(self):
        # TODO: add tuning on first frame with gui to adjust tuning values for tracking
        return


def save_main_hub(main_hub: MainHub):
    """Saves Main_Hub Object and deletes it

    Arguments:
        mainhub (MainHub): main_hub object to save and delete

    Returns:
        None
    """
    reseg_idx = 1
    if (
        main_hub.get_reseg_idx() != 0
    ):  # reseg_idx declaraed 0 on init, 1 if resegment until saving where it chnaages
        resegment_key_frames, reseg_idx = main_hub.get_key_frames(
            main_hub.get_key_frame_path()
        )
        new_frames = main_hub.get_new_frames()
        main_path = main_hub.get_root()
        output_dir = os.path.join(main_path, "output")
        annotations_output = os.path.join(output_dir, "annotations_output")
        annotation_output_path = os.path.join(annotations_output, "annotations.xml")
        annotation_copy_path = os.path.join(
            annotations_output, "annotations_{}.xml".format(reseg_idx)
        )
        # save previous annotations output
        if os.path.exists(annotation_output_path):
            with open(annotation_output_path, "r") as f:
                with open(annotation_copy_path, "w") as f2:
                    f2.write(f.read())
            reseg_idx += 1
    key_frame_arr = main_hub.get_roar_seg_tracker().get_key_frame_arr()
    main_hub.store_key_frames(
        key_frames=key_frame_arr,
        reseg_idx=reseg_idx,
        output_dir=main_hub.get_key_frame_path(),
    )
    main_hub.save_annotations()
    del main_hub
    torch.cuda.empty_cache()
    gc.collect()


def create_main_hub(
    sam_args=sam_args,
    segtracker_args=segtracker_args,
    aot_args=aot_args,
    job_id: int = -1,
    reseg_bool: bool = False,
    reuse_output: bool = False,
) -> MainHub:
    """Creates MainHub Object

    Args:
        sam_args (_type_, optional): segment anything model args. Defaults to sam_args.
        segtracker_args (_type_, optional): segmentation tracker args. Defaults to segtracker_args.
        aot_args (_type_, optional): tracker engine args. Defaults to aot_args.
        job_id (int, optional): job id according to CVAT. Defaults to -1.
        reseg_bool (bool, optional): define if tracking job is a resegmentation(previously tracked). Defaults to False.
        reuse_output (bool, optional): option to reuse annotation.xml output of previous tracking job. Defaults to False.

    Returns:
        _type_: MainHub Object
    """
    resegment = reseg_bool

    ###Create User Files
    root = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(root, "roar_annotations")

    reuse = reuse_output
    file_handler = RoarFileHandler(roar_path=root, downloads_path=DOWNLOADS_PATH)
    if not resegment and not reuse:
        file_handler.make_folder(job_id=job_id)
        file_handler.move_download_to_init_segment(job_id=job_id)
    elif not resegment and reuse:
        file_handler.make_folder(job_id=job_id)
    elif resegment and not reuse:
        file_handler.make_folder(job_id=job_id)
        file_handler.move_download_to_resegment(job_id=job_id)
    # job_id = 262

    main_path = os.path.join(root, str(job_id))
    photo_dir = os.path.join(main_path, "images")
    annotation_path = os.path.join(main_path, "annotations.xml")
    key_frame_path = os.path.join(main_path, "key_frames_{}".format(job_id))

    if not os.path.exists(annotation_path) or not os.path.exists(photo_dir):
        raise RuntimeError("annotations.xml or images directory not found")
    output_dir = os.path.join(main_path, "output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    reseg_dir = os.path.join(main_path, "resegment_annotations")
    if not os.path.exists(reseg_dir):
        os.makedirs(reseg_dir)
    reseg_path = os.path.join(reseg_dir, "annotations.xml")

    ###If want to reuse previous output annotation
    if reuse:
        annotations_output = os.path.join(output_dir, "annotations_output")
        annotation_output_path = os.path.join(annotations_output, "annotations.xml")
        reseg_path = annotation_output_path

    main_hub = MainHub(
        segtracker_args=segtracker_args,
        sam_args=sam_args,
        aot_args=aot_args,
        photo_dir=photo_dir,
        annotation_dir=(annotation_path if not resegment else reseg_path),
        output_dir=output_dir,
    )
    main_hub.set_key_frame_path(key_frame_path)
    main_hub.set_root(main_path)
    if resegment:
        main_hub.set_reseg_idx(1)
        new_frames = main_hub.get_new_frames()
        past_frames, reseg_idx = main_hub.get_key_frames(main_hub.get_key_frame_path())
        key_frame_arr = main_hub.remake_key_frames(
            main_hub.get_roar_seg_tracker(), new_frames, past_frames
        )
        main_hub.get_roar_seg_tracker().set_key_frame_arr(key_frame_arr)
    return main_hub


def arg_main(
    sam_args=sam_args,
    segtracker_args=segtracker_args,
    aot_args=aot_args,
    job_id: int = -1,
    reseg_bool: bool = False,
    reuse_output: bool = False,
    threads: int = 1,
    reseg_frames: list[int] = [],
    delete_zip: bool = False,
):
    sam_args["generator_args"] = {
        "points_per_side": 30,
        "pred_iou_thresh": 0.8,
        "stability_score_thresh": 0.9,
        "crop_n_layers": 1,
        "crop_n_points_downscale_factor": 2,
        "min_mask_region_area": 200,
    }

    segtracker_args = {
        "sam_gap": 7860,  # the interval to run sam to segment new objects
        "min_area": 200,  # minimal mask area to add a new mask as a new object
        "max_obj_num": 255,  # maximal object number to track in a video
        "min_new_obj_iou": 0.8,  # the area of a new object in the background should > 80%
    }

    resegment = reseg_bool

    ###Create User Files
    root = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(root, "roar_annotations")

    reuse = reuse_output
    file_handler = RoarFileHandler(roar_path=root, downloads_path=DOWNLOADS_PATH)
    if not resegment and not reuse:
        file_handler.make_folder(job_id=job_id)
        file_handler.move_download_to_init_segment(job_id=job_id)
    elif not resegment and reuse:
        file_handler.make_folder(job_id=job_id)
    else:
        file_handler.make_folder(job_id=job_id)
        file_handler.move_download_to_resegment(job_id=job_id)
    start_time = time.time()
    # job_id = 262

    main_path = os.path.join(root, str(job_id))
    photo_dir = os.path.join(main_path, "images")
    annotation_path = os.path.join(main_path, "annotations.xml")
    key_frame_path = os.path.join(main_path, "key_frames_{}".format(job_id))
    if not os.path.exists(annotation_path) or not os.path.exists(photo_dir):
        return RuntimeError("annotations.xml or images directory not found")
    output_dir = os.path.join(main_path, "output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    reseg_dir = os.path.join(main_path, "resegment_annotations")
    if not os.path.exists(reseg_dir):
        os.makedirs(reseg_dir)
    reseg_path = os.path.join(reseg_dir, "annotations.xml")

    ###If want to reuse previous output annotation
    if reuse:
        annotations_output = os.path.join(output_dir, "annotations_output")
        annotation_output_path = os.path.join(annotations_output, "annotations.xml")
        reseg_path = annotation_output_path

    main_hub = MainHub(
        segtracker_args=segtracker_args,
        sam_args=sam_args,
        aot_args=aot_args,
        photo_dir=photo_dir,
        annotation_dir=(annotation_path if not resegment else reseg_path),
        output_dir=output_dir,
    )
    # main_hub.set_key_frame_path(key_frame_path)
    start_time = time.time()

    # start tracking
    main_hub.max_workers = threads
    multithread = threads > 1

    reseg_idx = 1
    if resegment:
        resegment_key_frames, reseg_idx = main_hub.get_key_frames(key_frame_path)
        repeat = True
        new_frames = reseg_frames

        annotations_output = os.path.join(output_dir, "annotations_output")
        annotation_output_path = os.path.join(annotations_output, "annotations.xml")
        annotation_copy_path = os.path.join(
            annotations_output, "annotations_{}.xml".format(reseg_idx)
        )
        # save previous annotations output
        if os.path.exists(annotation_output_path):
            with open(annotation_output_path, "r") as f:
                with open(annotation_copy_path, "w") as f2:
                    f2.write(f.read())
            reseg_idx += 1

        # Add new key frame to key frame arr for saving later
        key_frame_arr = resegment_key_frames[:]
        key_frame_arr.extend(new_frames)

        dupes = {}
        for i in key_frame_arr:
            if dupes.get(i) is None:
                dupes[i] = 1

        key_frame_arr = list(dupes.keys())
        key_frame_arr.sort()
        main_hub.resegment_track(
            past_key_frames=resegment_key_frames,
            new_frames=new_frames,
            multithreading=multithread,
        )

    else:
        if not multithread:
            main_hub.track()
        else:
            main_hub.multi_trackers()
        key_frame_arr = main_hub.roarsegtracker.get_key_frame_arr()

    mid_time = time.time()
    print("storing data...")
    # save annotations
    # main_hub.store_tracker(frame="3093")
    main_hub.store_key_frames(
        key_frames=key_frame_arr, reseg_idx=reseg_idx, output_dir=key_frame_path
    )
    main_hub.save_annotations()
    end_time = time.time()
    print(
        "Finished Tracking in {} secs and finished writing annotations in {} secs".format(
            mid_time - start_time, end_time - mid_time
        )
    )
    if delete_zip:
        file_handler.delete_zip(job_id=job_id)
    del main_hub
    torch.cuda.empty_cache()
    gc.collect()
    print("Done!")


def main():
    sam_args["generator_args"] = {
        "points_per_side": 30,
        "pred_iou_thresh": 0.8,
        "stability_score_thresh": 0.9,
        "crop_n_layers": 1,
        "crop_n_points_downscale_factor": 2,
        "min_mask_region_area": 200,
    }

    segtracker_args = {
        "sam_gap": 7860,  # the interval to run sam to segment new objects
        "min_area": 200,  # minimal mask area to add a new mask as a new object
        "max_obj_num": 255,  # maximal object number to track in a video
        "min_new_obj_iou": 0.8,  # the area of a new object in the background should > 80%
    }
    job_id = -1
    resegment_ans = input("Is this a resegmentation task? (y/n): ")
    resegment = resegment_ans == "y" or resegment_ans == "Y"
    repeat = True
    ###Gather User Data
    while repeat:
        job_id = int(input("Enter job id: "))
        if job_id < 0:
            print("Invalid job id")
            continue
        else:
            check = input("Is the job id: {} correct? (y/n): ".format(job_id))
            repeat = not (check == "y" or check == "Y")

    ###Create User Files
    root = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(root, "roar_annotations")
    q = input("Would you like to reuse output annotation? (y/n): ")
    reuse = q == "y" or q == "Y"
    file_handler = RoarFileHandler(roar_path=root, downloads_path=DOWNLOADS_PATH)
    if not resegment and not reuse:
        file_handler.make_folder(job_id=job_id)
        file_handler.move_download_to_init_segment(job_id=job_id)
    elif not resegment and reuse:
        file_handler.make_folder(job_id=job_id)
    else:
        file_handler.make_folder(job_id=job_id)
        file_handler.move_download_to_resegment(job_id=job_id)
    start_time = time.time()
    # job_id = 262

    main_path = os.path.join(root, str(job_id))
    photo_dir = os.path.join(main_path, "images")
    annotation_path = os.path.join(main_path, "annotations.xml")
    key_frame_path = os.path.join(main_path, "key_frames_{}".format(job_id))

    if not os.path.exists(annotation_path) or not os.path.exists(photo_dir):
        return RuntimeError("annotations.xml or images directory not found")
    output_dir = os.path.join(main_path, "output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    reseg_dir = os.path.join(main_path, "resegment_annotations")
    if not os.path.exists(reseg_dir):
        os.makedirs(reseg_dir)
    reseg_path = os.path.join(reseg_dir, "annotations.xml")

    ###If want to reuse previous output annotation
    if reuse:
        annotations_output = os.path.join(output_dir, "annotations_output")
        annotation_output_path = os.path.join(annotations_output, "annotations.xml")
        reseg_path = annotation_output_path

    main_hub = MainHub(
        segtracker_args=segtracker_args,
        sam_args=sam_args,
        aot_args=aot_args,
        photo_dir=photo_dir,
        annotation_dir=(annotation_path if not resegment else reseg_path),
        output_dir=output_dir,
    )
    main_hub.set_key_frame_path(key_frame_path)
    start_time = time.time()
    reseg_idx = 1
    # start tracking
    multithread_ans = input("Do you want to use multithreading? (y/n): ")
    multithread = multithread_ans == "y" or multithread_ans == "Y"
    check_worker_input = lambda x: (int(x) <= main_hub.max_workers and int(x) > 0)
    convert_func = lambda x: int(x)
    if multithread:
        max_workers = rt.get_correct_input(
            check_worker_input, convert_func, "indicate max threads (int): "
        )
        main_hub.max_workers = max_workers

    if resegment:
        resegment_key_frames, reseg_idx = main_hub.get_key_frames(key_frame_path)
        repeat = True
        new_frames = []

        while repeat:
            new_frame_str = input(
                "What new frames fo u want to segment? (comma separated values):"
            )

            new_frames = [int(x) for x in new_frame_str.split(",")]
            print("Your selected frames are: {}".format(new_frames))
            repeat_ans = input("is this correct? (y/n): ")
            repeat = not (repeat_ans == "y" or repeat_ans == "Y")

        annotations_output = os.path.join(output_dir, "annotations_output")
        annotation_output_path = os.path.join(annotations_output, "annotations.xml")
        annotation_copy_path = os.path.join(
            annotations_output, "annotations_{}.xml".format(reseg_idx)
        )
        # save previous annotations output
        if os.path.exists(annotation_output_path):
            with open(annotation_output_path, "r") as f:
                with open(annotation_copy_path, "w") as f2:
                    f2.write(f.read())
            reseg_idx += 1

        key_frame_arr = resegment_key_frames[:]
        key_frame_arr.extend(new_frames)
        key_frame_arr.sort()
        main_hub.resegment_track(
            past_key_frames=resegment_key_frames,
            new_frames=new_frames,
            multithreading=multithread,
        )

    else:
        if not multithread:
            main_hub.track()
        else:
            main_hub.multi_trackers()
        key_frame_arr = main_hub.roarsegtracker.get_key_frame_arr()

    mid_time = time.time()
    print("storing data...")
    # save annotations
    # main_hub.store_tracker(frame="3093")
    main_hub.store_key_frames(
        key_frames=key_frame_arr, reseg_idx=reseg_idx, output_dir=key_frame_path
    )
    main_hub.save_annotations()
    end_time = time.time()
    print(
        "Finished Tracking in {} secs and finished writing annotations in {} secs".format(
            mid_time - start_time, end_time - mid_time
        )
    )
    delete = input("Would you like to delete the {}.zip? (y/n): ".format(job_id))
    if delete == "y" or delete == "Y":
        file_handler.delete_zip(job_id=job_id)
    del main_hub
    torch.cuda.empty_cache()
    gc.collect()
    print("Done!")


### Other tools

if __name__ == "__main__":
    main()
