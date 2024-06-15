import argparse
import os

import cv2


def visualize_flow_video(data_dir, flow_folder):
    """
    Visualize flow video from the given data directory and image folder.
    The images in the data directory will be put on the left side of the video.
    Args:
        data_dir: data directory
        flow_folder: flow image folder
    """
    # create output directory
    out_dir = os.path.join(flow_dir, 'video')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # read ori image
    suffixs = ['.jpg', '.png', '.jpeg']
    ori_images = None
    for suffix in suffixs:
        ori_images = [img for img in os.listdir(data_dir) if img.endswith(suffix)]
        if len(ori_images) > 0:
            break
    ori_images.sort()
    first_ori_image_path = os.path.join(data_dir, ori_images[0])
    ori_frame = cv2.imread(first_ori_image_path)
    ori_height, ori_width, ori_layers = ori_frame.shape
    print(f"ori_height: {ori_height}, ori_width: {ori_width}, ori_layers: {ori_layers}")

    # read flow image
    images = [img for img in os.listdir(flow_folder) if img.endswith(".png")]
    images.sort()
    first_image_path = os.path.join(flow_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape
    print(f"height: {height}, width: {width}, layers: {layers}")

    if ori_height != height or ori_width != width:
        # output flow video
        video_file = os.path.join(out_dir, 'output.mp4')
        frame_rate = 15.0
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_file, fourcc, frame_rate, (width, height))
        for image in images:
            image_path = os.path.join(flow_folder, image)
            frame = cv2.imread(image_path)
            video.write(frame)
    else:
        if len(ori_images) == len(images):
            # output contacted video
            video_file = os.path.join(out_dir, 'output.mp4')
            frame_rate = 15.0
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(video_file, fourcc, frame_rate, (2 * width, height))
            for index in range(len(images)):
                ori_image_path = os.path.join(data_dir, ori_images[index])
                ori_frame = cv2.imread(ori_image_path)
                image_path = os.path.join(flow_folder, images[index])
                frame = cv2.imread(image_path)
                combined_frame = cv2.hconcat([ori_frame, frame])
                video.write(combined_frame)
        elif len(ori_images) == len(images) + 1:
            # output contacted video (deprecate the first ori image)
            video_file = os.path.join(out_dir, 'output.mp4')
            frame_rate = 15.0
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(video_file, fourcc, frame_rate, (2 * width, height))
            for index in range(len(images)):
                ori_image_path = os.path.join(data_dir, ori_images[index+1])
                ori_frame = cv2.imread(ori_image_path)
                image_path = os.path.join(flow_folder, images[index])
                frame = cv2.imread(image_path)
                combined_frame = cv2.hconcat([ori_frame, frame])
                video.write(combined_frame)
        else:
            # output flow video
            video_file = os.path.join(out_dir, 'output.mp4')
            frame_rate = 15.0
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(video_file, fourcc, frame_rate, (width, height))
            for image in images:
                image_path = os.path.join(flow_folder, image)
                frame = cv2.imread(image_path)
                video.write(frame)

    video.release()
    print(f"Video saved to {video_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data/TUBCrowdFlow/gt_flow/IM01')
    parser.add_argument('--flow_dir', default='./outputs/sintel/data/TUBCrowdFlow/images/IM01')

    args = parser.parse_args()

    data_dir = args.data_dir
    flow_dir = args.flow_dir

    visualize_flow_video(data_dir, flow_dir)
