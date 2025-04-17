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
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print ("Number of frames: ", video_length)
    count = 0
    print ("Converting video...")
    # Start converting the video
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        if not ret:
            continue
        # Write the results back to output location.
        cv2.imwrite(output_loc + "/img_%#05d.jpg" % (count+1), frame)
        count = count + 1
        # If there are no more frames left
        if (count > (video_length-1)):
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            print ("Done extracting frames.\n%d frames extracted" % count)
            print ("It took %d seconds for conversion." % (time_end-time_start))
            break

if __name__ == "__main__":
    inputs = ["/nfs/wattrel/data/md0/datasets/AE2/AE2_data/pour_milk/train/ego",
              "/nfs/wattrel/data/md0/datasets/AE2/AE2_data/pour_milk/train/exo",

            #   "/nfs/wattrel/data/md0/datasets/AE2/AE2_data/pour_liquid/train/ego",
            #   "/nfs/wattrel/data/md0/datasets/AE2/AE2_data/pour_liquid/train/exo",

            #   "/nfs/wattrel/data/md0/datasets/AE2/AE2_data/pour_liquid/test/ego",
            #   "/nfs/wattrel/data/md0/datasets/AE2/AE2_data/pour_liquid/test/exo",
            #   "/nfs/wattrel/data/md0/datasets/AE2/AE2_data/pour_liquid/val/ego",
            #   "/nfs/wattrel/data/md0/datasets/AE2/AE2_data/pour_liquid/val/exo",

    #           "/nfs/wattrel/data/md0/datasets/AE2/AE2_data/pour_milk/test/ego",
    #           "/nfs/wattrel/data/md0/datasets/AE2/AE2_data/pour_milk/test/exo",
    #           "/nfs/wattrel/data/md0/datasets/AE2/AE2_data/pour_milk/val/ego",
    #           "/nfs/wattrel/data/md0/datasets/AE2/AE2_data/pour_milk/val/exo",

    #           "/nfs/wattrel/data/md0/datasets/AE2/AE2_data/tennis_forehand/ego",
    #           "/nfs/wattrel/data/md0/datasets/AE2/AE2_data/tennis_forehand/exo",
    #           "/nfs/wattrel/data/md0/datasets/AE2/AE2_data/tennis_forehand/ego",
    #           "/nfs/wattrel/data/md0/datasets/AE2/AE2_data/tennis_forehand/exo",
              ]
    
    for input_folder in inputs:    
        print(input_folder)
        for filename in os.listdir(input_folder):
            if filename.lower().endswith(".mp4"):
                video_path = os.path.join(input_folder, filename)
                video_name = os.path.splitext(filename)[0]
                output_path = os.path.join(input_folder, video_name + '_frames')
                video_to_frames(video_path, output_path)