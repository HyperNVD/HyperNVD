import os
import cv2
import sys
import argparse


if __name__ == '__main__':

    # Define paths
    parser = argparse.ArgumentParser(description='Preprocess image sequence')
    parser.add_argument('--path2DAVIS', type=str, help='Path to DAVIS dataset')
    args = parser.parse_args()

    jpeg_images_path = os.path.join(args.path2DAVIS, "JPEGImages", "480p")
    mp4_output_path = os.path.join(args.path2DAVIS, "MP4")

    # Create MP4 output folder if it doesn't exist
    os.makedirs(mp4_output_path, exist_ok=True)

    # Loop through each folder in the JPEGImages/480p directory
    for folder_name in os.listdir(jpeg_images_path):
        folder_path = os.path.join(jpeg_images_path, folder_name)
        
        if not os.path.isdir(folder_path):
            continue

        # Get a list of all image files in the folder and sort them
        image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
        if not image_files:
            continue

        # Read the first image to get the frame size
        first_image_path = os.path.join(folder_path, image_files[0])
        frame = cv2.imread(first_image_path)
        height, width, layers = frame.shape

        # Define the codec and create VideoWriter object
        video_path = os.path.join(mp4_output_path, f"{folder_name}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))

        # Write each image to the video
        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            frame = cv2.imread(image_path)
            out.write(frame)

        # Release the video writer
        out.release()

        print(f"Video saved: {video_path}")

    print("All videos have been created and saved in the MP4 folder.")
