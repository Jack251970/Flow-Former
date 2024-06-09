import argparse
import os

import cv2


def visualize_flow_video(data_dir, image_folder, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # read ori image
    ori_images = [img for img in os.listdir(data_dir) if img.endswith(".png")]
    ori_images.sort()
    first_ori_image_path = os.path.join(data_dir, ori_images[0])
    ori_frame = cv2.imread(first_ori_image_path)
    ori_height, ori_width, ori_layers = ori_frame.shape
    print(f"ori_height: {ori_height}, ori_width: {ori_width}, ori_layers: {ori_layers}")

    # read flow image
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort()
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape
    print(f"height: {height}, width: {width}, layers: {layers}")

    if ori_height != height or ori_width != width or len(ori_images) != (len(images) + 1):
        # output flow video
        video_file = os.path.join(out_dir, 'output.mp4')
        frame_rate = 15.0
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_file, fourcc, frame_rate, (width, height))
        for image in images:
            image_path = os.path.join(image_folder, image)
            frame = cv2.imread(image_path)
            video.write(frame)
    else:
        # output contacted video (deprecate the first ori image)
        video_file = os.path.join(out_dir, 'output.mp4')
        frame_rate = 15.0
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_file, fourcc, frame_rate, (2 * width, height))
        for index in range(len(images)):
            ori_image_path = os.path.join(data_dir, ori_images[index+1])
            ori_frame = cv2.imread(ori_image_path)
            image_path = os.path.join(image_folder, images[index])
            frame = cv2.imread(image_path)
            combined_frame = cv2.hconcat([ori_frame, frame])
            video.write(combined_frame)

    video.release()
    print(f"Video saved to {video_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sintel_dir', default='./data/TUBCrowdFlow/images/IM01')
    parser.add_argument('--viz_root_dir', default='./outputs/sintel/data/TUBCrowdFlow/images/IM01')
    parser.add_argument('--out_dir', default='./outputs/sintel/data/TUBCrowdFlow/images/IM01/Video')

    args = parser.parse_args()

    root_dir = args.sintel_dir
    viz_root_dir = args.viz_root_dir
    out_dir = args.out_dir

    visualize_flow_video(root_dir, viz_root_dir, out_dir)
