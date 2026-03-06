from collections import Counter, defaultdict
import json
import logging
import os
from pathlib import Path
import sys
import time
from accelerate import Accelerator
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs
sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())
import hydra
import numpy as np
from pytorch_lightning import seed_everything
from termcolor import colored
import torch
from tqdm.auto import tqdm
import torch.distributed as dist
from hydra import compose, initialize
import argparse
from policy_evaluation.multistep_sequences import get_sequences
from policy_evaluation.utils import get_default_beso_and_env, get_env_state_for_initial_condition, join_vis_lang
from policy_models.utils.utils import get_last_checkpoint, get_all_checkpoints
from policy_models.rollout.rollout_video import RolloutVideo


logger = logging.getLogger(__name__)


def get_video_tag(i):
    if dist.is_available() and dist.is_initialized():
        i = i * dist.get_world_size() + dist.get_rank()
    return f"_long_horizon/sequence_{i}"


def get_log_dir(log_dir):
    if log_dir is not None:
        log_dir = Path(log_dir)
        os.makedirs(log_dir, exist_ok=True)
    else:
        log_dir = Path(__file__).parents[3] / "evaluation"
        if not log_dir.exists():
            log_dir = Path("/tmp/evaluation")

    log_dir = log_dir / "logs" / time.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(log_dir, exist_ok=True)
    print(f"logging to {log_dir}")
    return log_dir


def count_success(results):
    count = Counter(results)
    step_success = []
    for i in range(1, 6):
        n_success = sum(count[j] for j in reversed(range(i, 6)))
        sr = n_success / len(results)
        step_success.append(sr)
    return step_success


def print_and_save(total_results, cfg, log_dir=None):
    if log_dir is None:
        log_dir = get_log_dir(cfg.train_folder)

    sequences = get_sequences(cfg.num_sequences)

    current_data = {}
    ranking = {}
    for checkpoint, results in total_results.items():
        epoch = checkpoint.stem
        print(f"Results for Epoch {epoch}:")
        avg_seq_len = np.mean(results)
        ranking[epoch] = avg_seq_len
        chain_sr = {i + 1: sr for i, sr in enumerate(count_success(results))}
        print(f"Average successful sequence length: {avg_seq_len}")
        print("Success rates for i instructions in a row:")
        for i, sr in chain_sr.items():
            print(f"{i}: {sr * 100:.1f}%")

        cnt_success = Counter()
        cnt_fail = Counter()

        for result, (_, sequence) in zip(results, sequences):
            for successful_tasks in sequence[:result]:
                cnt_success[successful_tasks] += 1
            if result < len(sequence):
                failed_task = sequence[result]
                cnt_fail[failed_task] += 1

        total = cnt_success + cnt_fail
        task_info = {}
        for task in total:
            task_info[task] = {"success": cnt_success[task], "total": total[task]}
            print(f"{task}: {cnt_success[task]} / {total[task]} |  SR: {cnt_success[task] / total[task] * 100:.1f}%")

        data = {"avg_seq_len": avg_seq_len, "chain_sr": chain_sr, "task_info": task_info}

        current_data[epoch] = data

        print()
    previous_data = {}
    try:
        with open(log_dir / "results.json", "r") as file:
            previous_data = json.load(file)
    except FileNotFoundError:
        pass
    json_data = {**previous_data, **current_data}
    with open(log_dir / "results.json", "w") as file:
        json.dump(json_data, file, indent=2)
    print(f"Best model: epoch {max(ranking, key=ranking.get)} with average sequences length of {max(ranking.values())}")


def evaluate_policy(model, env, lang_embeddings, cfg, 
                    num_procs, procs_id, checkpoint,
                    save_dir=None):
    task_oracle = hydra.utils.instantiate(cfg.tasks)
    val_annotations = cfg.annotations

    # video stuff
    rollout_video = RolloutVideo(
        logger=logger,
        empty_cache=False,
        log_to_file=True,
        save_dir=save_dir,
        resolution_scale=1,
    )

    eval_sequences = get_sequences(cfg.num_sequences)
    num_seq_per_procs = cfg.num_sequences // num_procs
    eval_sequences = eval_sequences[num_seq_per_procs*procs_id:num_seq_per_procs*(procs_id+1)]

    results = []

    eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    for i, (initial_state, eval_sequence) in enumerate(eval_sequences):
        # TODO 这里控制是否存下来eval的视频
        record = False
        result = evaluate_sequence(
            env, model, task_oracle, initial_state, eval_sequence, lang_embeddings, val_annotations, cfg, record, rollout_video, i
        )
        results.append(result)
        if record:
            rollout_video.write_to_tmp()

        success_rates = count_success(results)
        average_rate = sum(success_rates) / len(success_rates) * 5
        description = " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(success_rates)])
        description += f" Average: {average_rate:.1f} |"
        eval_sequences.set_description(description)

        if record:
            rollout_video._log_currentvideos_to_file(i, save_as_video=True)

    results_dict = {}
    results_dict[checkpoint] = results
    print_and_save(results_dict, cfg, log_dir=save_dir)
    return results


def evaluate_sequence(
    env, model, task_checker, initial_state, eval_sequence, lang_embeddings, val_annotations, cfg, record, rollout_video, i
):
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    if record:
        caption = " | ".join(eval_sequence)
        rollout_video.new_video(tag=get_video_tag(i), caption=caption)
    success_counter = 0

    for subtask in eval_sequence:
        if record:
            rollout_video.new_subtask()
        success = rollout(env, model, task_checker, cfg, subtask, lang_embeddings, val_annotations, record, rollout_video)
        if record:
            rollout_video.draw_outcome(success)
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter


def rollout(env, model, task_oracle, cfg, subtask, lang_embeddings, val_annotations, record=False, rollout_video=None):

    obs = env.get_obs()
    
    # get lang annotation for subtask
    lang_annotation = val_annotations[subtask][0]
    # get language goal embedding
    goal = lang_embeddings.get_lang_goal(lang_annotation)
    goal['lang_text'] = val_annotations[subtask][0]
    
    model.reset()
    start_info = env.get_info()

    for step in range(cfg.ep_len):  # 360
        action = model.step(obs, goal)  # torch.Size([7])

        # 在env中执行action
        obs, _, _, current_info = env.step(action)
        
        if record:
            # update video
            rollout_video.update(obs["rgb_obs"]["rgb_static"])
        
        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            if record:
                rollout_video.add_language_instruction(lang_annotation)
            return True  # 从这里返回就是rollout成功
    
    if record:
        rollout_video.add_language_instruction(lang_annotation)
    return False  # 从这里返回就是rollout失败


def main(cfg):
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))
    acc = Accelerator(kwargs_handlers=[kwargs])
    device = acc.device

    log_wandb = cfg.log_wandb
    seed_everything(0, workers=True)
    checkpoints = get_all_checkpoints(Path(cfg.train_folder))
    lang_embeddings = None
    env = None
    results = {}

    print('train_folder',cfg.train_folder)
    print("\n" + "="*80)
    print(f"Found {len(checkpoints)} checkpoint(s) for evaluation:")
    for i, ckpt in enumerate(checkpoints, 1):
        print(f"  [{i}] {str(ckpt)}")
    print("="*80 + "\n")

    for checkpoint in checkpoints:
        print(device)
        print(f"\n{'*'*80}")
        print(f"Processing checkpoint: {str(checkpoint)}")
        print(f"{'*'*80}")
        env, _, lang_embeddings = get_default_beso_and_env(
            cfg.train_folder,
            cfg.root_data_dir,
            checkpoint,
            env=env,
            lang_embeddings=lang_embeddings,
            eval_cfg_overwrite=cfg.eval_cfg_overwrite,
            device_id=device.index,
            cfg=cfg,
        )

        print(f"Loading model from {str(checkpoint)}")
        state_dict = torch.load(str(checkpoint), map_location="cpu")
        print(f"✓ Successfully loaded checkpoint from: {str(checkpoint)}")

        model = hydra.utils.instantiate(cfg.model)

        model.load_state_dict(state_dict['model'],strict = False)
        model.freeze()

        print(cfg.num_sampling_steps, cfg.sampler_type, cfg.multistep, cfg.sigma_min, cfg.sigma_max, cfg.noise_scheduler)
        model.num_sampling_steps = cfg.num_sampling_steps
        model.sampler_type = cfg.sampler_type
        model.multistep = cfg.multistep
        if cfg.sigma_min is not None:
            model.sigma_min = cfg.sigma_min
        if cfg.sigma_max is not None:
            model.sigma_max = cfg.sigma_max
        if cfg.noise_scheduler is not None:
            model.noise_scheduler = cfg.noise_scheduler

        if cfg.cfg_value != 1:
            raise NotImplementedError("cfg_value != 1 not implemented yet")
        model = acc.prepare(model, device_placement=[True])
        model.process_device()
        model.eval()
        if log_wandb:
            log_dir = get_log_dir(cfg.train_folder)
            os.makedirs(log_dir / "wandb", exist_ok=True)

            results[checkpoint] = evaluate_policy(
                model, env, lang_embeddings, cfg, 
                acc.num_processes, acc.process_index, checkpoint,
                save_dir=Path(log_dir))
            avg_reward = torch.tensor(results[checkpoint]).float().mean().to(device)
            acc.wait_for_everyone()
            avg_reward = acc.gather_for_metrics(avg_reward).mean()
            if acc.is_main_process:
                print('average success rate ', avg_reward)
        
        # 显式清理
        model.to("cpu")
        model.process_device()
        del model
        del env
        del lang_embeddings
        lang_embeddings = None
        env = None
        torch.cuda.empty_cache()


if __name__ == "__main__":
    os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_model_path", type=str, default="")
    parser.add_argument("--action_model_folder", type=str, default="")
    parser.add_argument("--clip_model_path", type=str, default="")
    parser.add_argument("--t5_model_path", type=str, default="")
    parser.add_argument("--language_goal_path", type=str, default="")
    parser.add_argument("--calvin_abc_dir", type=str, default="")
    
    args = parser.parse_args()
    
    with initialize(config_path="../policy_conf", job_name="calvin_evaluate_all.yaml"):
        cfg = compose(config_name="calvin_evaluate_all.yaml")
    cfg.model.pretrained_model_path = args.video_model_path
    cfg.train_folder = args.action_model_folder
    cfg.model.text_encoder_path = args.clip_model_path
    if args.t5_model_path:
        cfg.model.t5_model_path = args.t5_model_path
        print(f"[INFO] Overriding cfg.model.t5_model_path with args.t5_model_path: {cfg.model.t5_model_path}")
    if args.language_goal_path:
        cfg.model.language_goal_path = args.language_goal_path
        print(f"[INFO] Overriding cfg.model.language_goal_path with args.language_goal_path: {cfg.model.language_goal_path}")
    cfg.root_data_dir = args.calvin_abc_dir
    main(cfg)
