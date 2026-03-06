import cv2
import mediapy
import os
import random
import math
import argparse
from diffusers.models import AutoencoderKL
import mediapy
import torch
import h5py
import numpy as np
import json
from diffusers.models import AutoencoderKL,AutoencoderKLTemporalDecoder
import mediapy


def load_hdf5(dataset_path):
    global compressed

    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    with h5py.File(dataset_path, 'r') as root:
        compressed = root.attrs.get('compress', False)
        arm_qpos = root['/observations/arm_qpos'][()]
        hand_qpos = root['/observations/hand_qpos'][()]
        arm_end_pose = root['/observations/arm_end_pose'][()]
        waist_qpos = root['/observations/waist_qpos'][()]
        neck_qpos = root['/observations/neck_qpos'][()]
        action = root['/action'][()]
        text = root['/text'][()]

        image_dict = dict()
        for cam_name in root[f'/observations/images/'].keys():
            image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]

    cam_names = list(image_dict.keys())
    cam_names = sorted(cam_names)
    all_cam_videos = {}
    if compressed:
        for cam_name in cam_names:
            decompressed_image = np.array([cv2.imdecode(row,1) for row in image_dict[cam_name]])
            all_cam_videos[cam_name] = decompressed_image
    else:
        for cam_name in cam_names:
            all_cam_videos[cam_name] = image_dict[cam_name]
        
    return arm_qpos, hand_qpos, arm_end_pose, waist_qpos, neck_qpos, action, all_cam_videos, text


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare latent data for video prediction')
    parser.add_argument('--raw_data_path', type=str, default='xhand', help='Path to raw data')
    parser.add_argument('--output_dir', type=str, default='./opensource_robotdata/xbot', help='Output base directory')
    parser.add_argument('--vae_model_path', type=str, 
                        default='ckpt/stable_video_diffusion_img2vid',
                        help='Path to VAE model')
    parser.add_argument('--skip_step', type=int, default=5, help='Skip step for downsampling')
    parser.add_argument('--fps', type=int, default=10, help='FPS for output video')
    parser.add_argument('--video_size', type=int, default=256, help='Video resolution (square)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for VAE encoding')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    ############## raw data paths (teleoperation hdf5 files) ###############
    raw_data_path = args.raw_data_path
    output_dir = args.output_dir
    dir = output_dir
    
    print(f"Raw data path: {raw_data_path}")
    print(f"Output directory: {dir}")
    print(f"VAE model path: {args.vae_model_path}")
    print(f"Skip step: {args.skip_step}")

    ############## saved paths ###############
    video_dir = os.path.join(dir, 'videos')
    latent_video_dir = os.path.join(dir, 'latent_videos')
    anno_dir = os.path.join(dir, 'annotation')
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(latent_video_dir, exist_ok=True)
    os.makedirs(anno_dir, exist_ok=True)

    raw_file = []
    all_subfolders = [
        d for d in sorted(os.listdir(raw_data_path))
        if os.path.isdir(os.path.join(raw_data_path, d))
    ]
    for sub in all_subfolders:
        sub_path = os.path.join(raw_data_path, sub)
        sub_raw_file = os.listdir(sub_path)
        sub_raw_file = [f for f in sub_raw_file if f.endswith(".hdf5")]
        sub_raw_file.sort()
        sub_raw_file = [os.path.join(sub_path, f) for f in sub_raw_file]
        raw_file += sub_raw_file

    print(f"Found {len(raw_file)} hdf5 files")

    ####################################################
    # start prepare vae latent data
    print(f"Loading VAE model from {args.vae_model_path}")
    vae = AutoencoderKLTemporalDecoder.from_pretrained(args.vae_model_path, subfolder="vae").to("cuda")

    failed_num = 0
    success_num = 0
    for file_num, file_name in enumerate(raw_file):
        anno_ind_all = file_num
        data_type = 'val' if anno_ind_all%50==4 else 'train'
        with h5py.File(file_name, 'r') as file:
            arm_qpos, hand_qpos, arm_end_pose, waist_qpos, neck_qpos, action_all, image_dict,texts  = load_hdf5(file_name)
            print(file_name)
        text = str(texts[0])[2:-1]

        if 'capybara' in text:
            text = text.replace('capybara', 'pink capybara')
        if 'cube' in text:
            text = text.replace('cube', 'orange cube')
        if 'duck' in text:
            text = text.replace('duck', 'yellow duck')
        if 'mouse' in text:
            text = text.replace('mouse', 'brown mouse')
        if 'seal' in text:
            text = text.replace('seal', 'white seal')
        if 'bamboo' in text:
            p = random.random()
            if p < 0.5:
                text = text.replace('bamboo', 'green bamboo')
        text = 'place white bag in left and black bag in right'

        # split 1 trajectory into 5 trajectories if data is record at 50 hz. since the video model always predict 16 frames with frame intervel=0.1s
        skip_step = args.skip_step

        assert arm_end_pose.shape[-1] == 14
        assert hand_qpos.shape[-1] == 24
        states_all = np.concatenate((arm_end_pose[:,:7],hand_qpos[:,:12],arm_end_pose[:,7:],hand_qpos[:,12:]),axis=1)

        num_traj = 1 if data_type == 'val' else 1
        for j in range(num_traj):
            key_in_order = ['cam_high', 'cam_left', 'cam_right']
            latent_key = ['cam_high', 'cam_left', 'cam_right']
            action = action_all[j:]
            action = action[::skip_step]

            states = states_all[j:]
            states = states[::skip_step]

            anno_ind = skip_step*anno_ind_all+j
            for idx,cam_name in enumerate(key_in_order):
                img_all = image_dict[cam_name]

                frame, h, w, c = img_all.shape
                pad_h = int(w*0.75)
                img_pad = np.zeros((frame, pad_h, w, c), dtype=np.uint8)
                img_pad[:, int(pad_h/2-h/2):int(pad_h/2+h/2), :w, :] = img_all

                img_all = img_pad

                img = img_all[j:]
                img = img[::skip_step]

                # crop
                frames = np.array(img)
                # save latent video
                latent_video_path = f"{dir}/latent_videos/{data_type}/{anno_ind}"
                os.makedirs(latent_video_path, exist_ok=True)
                frames = torch.tensor(frames).permute(0, 3, 1, 2).float().to("cuda") / 255.0*2-1
                # resize to video_size x video_size
                x = torch.nn.functional.interpolate(frames, size=(args.video_size, args.video_size), mode='bilinear', align_corners=False)
                resize_video = ((x / 2.0 + 0.5).clamp(0, 1)*255)
                resize_video = resize_video.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
                # save images to video
                video_path = f"{dir}/videos/{data_type}/{anno_ind}"
                os.makedirs(video_path, exist_ok=True)
                mediapy.write_video(f"{dir}/videos/{data_type}/{anno_ind}/{idx}.mp4", resize_video, fps=args.fps)

                img_path = f"{dir}/imgs/{data_type}/{anno_ind}/{idx}.mp4"

                if cam_name in latent_key:
                    with torch.no_grad():
                        batch_size = args.batch_size
                        latents = []
                        for i in range(0, len(x), batch_size):
                            batch = x[i:i+batch_size]
                            latent = vae.encode(batch).latent_dist.sample().mul_(vae.config.scaling_factor).cpu()
                            latents.append(latent)
                        x = torch.cat(latents, dim=0)
                    
                    torch.save(x, f"{latent_video_path}/{idx}.pt")
            
            success_num += 1

            print("text", text, "num", file_num, "total_num", len(raw_file))
            
            # save anno
            info = {
                "task": "robot_trajectory_prediction",
                "texts": [
                    text
                ],
                "videos": [
                    {
                        "video_path": f"videos/{data_type}/{anno_ind}/0.mp4"
                    },
                    {
                        "video_path": f"videos/{data_type}/{anno_ind}/1.mp4"
                    },
                    {
                        "video_path": f"videos/{data_type}/{anno_ind}/2.mp4"
                    }
                ],
                "episode_id": anno_ind,
                "video_length": len(action),
                "latent_videos": [
                    {
                        "latent_video_path": f"latent_videos/{data_type}/{anno_ind}/0.pt"
                    },
                    {
                        "latent_video_path": f"latent_videos/{data_type}/{anno_ind}/1.pt"
                    },
                    {
                        "latent_video_path": f"latent_videos/{data_type}/{anno_ind}/2.pt"
                    },
                    
                ],
                "states": states.tolist(),
                "actions": action.tolist(),
            }
            
            # write json file
            os.makedirs(f"{dir}/annotation/{data_type}", exist_ok=True)
            with open(f"{dir}/annotation/{data_type}/{anno_ind}.json", "w") as f:
                json.dump(info, f, indent=2)
    
    print(f"\nProcessing completed. Success: {success_num}, Failed: {failed_num}")
