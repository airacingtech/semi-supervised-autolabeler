from roar_main import MainHub, create_main_hub
import matplotlib.pyplot as plt

def main():
    main_hub = create_main_hub(job_id=251, reseg_bool=True, reuse_output=True)
    main_hub.set_tracker()
    main_hub.track_key_frame_mask_objs = main_hub.roarsegtracker.get_key_frame_to_masks()
    end_frame_idx = main_hub.roarsegtracker.get_end_frame_idx()
    start_frame_idx = main_hub.roarsegtracker.get_start_frame_idx()
    while(True):
        frame = input("what frame? {}-{}: ".format(start_frame_idx, end_frame_idx))
        frame = int(frame)
        img, img_mask = main_hub.get_frame(frame, end_frame_idx=end_frame_idx, start_frame_idx=start_frame_idx)
        f, axarr = plt.subplots(1,2)
        axarr[0].imshow(img)
        axarr[1].imshow(img_mask)
        plt.show()
if __name__ == '__main__':
    main()

    