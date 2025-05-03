# https://stackoverflow.com/questions/33311153/python-extracting-and-saving-video-frames

import cv2
import time
import os

def video_to_frames(input_loc, output_loc):
    """Function to extract frames from input video file
    and save them as separate frames in an output directory.
    Args:
        input_loc: Input video file.
        output_loc: Output directory to save the frames.
    Returns:
        None
    """
    os.makedirs(output_loc, exist_ok=True)
    # Log the time
    time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #-1
    # print ("Number of frames: ", video_length)
    count = 0
    # print ("Converting video...")
    # Start converting the video
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        if not ret:
            print("breaking")
            # break
            continue
        # Write the results back to output location.
        cv2.imwrite(output_loc + "/img_%#05d.jpg" % (count+1), frame)
        count = count + 1
        # If there are no more frames left
        # if (count > (video_length-1)):
        #     # Log the time again
        #     time_end = time.time()
        #     # Release the feed
        #     cap.release()
        #     # Print stats
        #     print ("Done extracting frames.\n%d frames extracted" % count)
        #     # print ("It took %d seconds for conversion." % (time_end-time_start))
        #     break
    cap.release()
    # Print stats
    print ("Done extracting frames.\n%d frames extracted" % count)
    
def save_last_frame(input_loc, output_loc):
    cap = cv2.VideoCapture(input_loc)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Number of frames: ", video_length)

    # Set position to the last frame (index is zero-based)
    cap.set(cv2.CAP_PROP_POS_FRAMES, video_length - 1)

    ret, frame = cap.read()
    if ret:
        cv2.imwrite(output_loc + "/img_%#05d.jpg" % (video_length), frame)
        print("Saved last frame.")
    else:
        print("Could not read the last frame.")

    cap.release()

if __name__ == "__main__":
    parent_folder = '/nfs/wattrel/data/md0/datasets/EgoProceL/videos/pc_disassembly'  # <-- change this to your top folder

    for root, dirs, files in os.walk(parent_folder):
        for filename in files:
            if filename.lower().endswith((".mp4", ".avi")):
                video_path = os.path.join(root, filename)
                video_name = os.path.splitext(filename)[0]
                output_path = os.path.join(root, video_name + '_frames')
                if not os.path.exists(output_path):
                    print(f"Processing: {video_path}")
                    video_to_frames(video_path, output_path)
                # save_last_frame(video_path, output_path)