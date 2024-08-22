import cv2
import os

def save_video_frames(video_path, output_folder):
    # Check if output folder exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Check if the video was opened successfully
    if not video.isOpened():
        print("Error: Could not open video.")
        return

    frame_number = 0

    # Loop through the video frames
    while True:
        ret, frame = video.read()

        # Break the loop if there are no frames left to read
        if not ret:
            break

        # Construct the filename for the current frame
        frame_filename = os.path.join(output_folder, f"{video_path}_frame_{frame_number:04d}.jpg")

        # Save the frame as an image
        cv2.imwrite(frame_filename, frame)

        print(f"Saved {frame_filename}")

        # Increment the frame number
        frame_number += 1

    # Release the video capture object
    video.release()

    print("Finished saving all frames.")


save_video_frames('17.mp4', 'test_images')
