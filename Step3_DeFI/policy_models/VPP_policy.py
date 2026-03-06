import logging
from typing import Dict, Optional, Tuple
from functools import partial
from torch import einsum, nn
from einops import rearrange, repeat
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
import einops
from policy_models.edm_diffusion.score_wrappers import GCDenoiser
import omegaconf
import hydra
from pathlib import Path
from policy_models.module.clip_lang_encoder import LangClip
from policy_models.edm_diffusion.gc_sampling import *
from policy_models.utils.lr_schedulers.tri_stage_scheduler import TriStageLRScheduler
from policy_models.module.Video_Former import Video_Former_3D
from diffusers import StableVideoDiffusionPipeline
from transformers import AutoTokenizer, CLIPTextModelWithProjection
from policy_models.m_former_univla.latent_motion_tokenizer_univla import UncontrolledDINOLatentActionModel
# 如果正常的1步diffusion过程，用下面这个
from policy_models.module.diffusion_extract import Diffusion_feature_extractor
# 如果想输出不同步数的SVD预测的video，用下面这个
from policy_models.module.diffusion_extract_outvideo import Diffusion_feature_extractor as Diffusion_feature_extractor_outvideo


logger = logging.getLogger(__name__)


def load_primary_models(pretrained_model_path, eval=False):
    if eval:
        pipeline = StableVideoDiffusionPipeline.from_pretrained(pretrained_model_path, torch_dtype=torch.float16)
    else:
        pipeline = StableVideoDiffusionPipeline.from_pretrained(pretrained_model_path)
    return pipeline, None, pipeline.feature_extractor, pipeline.scheduler, pipeline.video_processor, \
        pipeline.image_encoder, pipeline.vae, pipeline.unet


class VPP_Policy(pl.LightningModule):
    """
    The lightning module used for training.
    """

    def __init__(
            self,
            optimizer: DictConfig,
            lr_scheduler: DictConfig,
            latent_dim: int = 512,
            multistep: int = 10,
            sampler_type: str = 'ddim',
            num_sampling_steps: int = 10,
            sigma_data: float = 0.5,
            sigma_min: float = 0.001,
            sigma_max: float = 80,
            noise_scheduler: str = 'exponential',
            sigma_sample_density_type: str = 'loglogistic',
            use_lr_scheduler: bool = True,
            act_window_size: int = 10,
            use_text_not_embedding: bool = False,
            seed: int = 42,
            pretrained_model_path: str = '/ckpt/svd/checkpoint-100000',
            text_encoder_path: str = '/ckpt/clip-vit-base-patch32',
            t5_model_path: str = '/ckpt/t5_base',
            language_goal_path: str = '/ckpt/ViT-B-32.pt',
            use_position_encoding: bool = True,
            Former_depth: int = 3,
            Former_heads: int = 8,
            Former_dim_head: int = 64,
            Former_num_time_embeds: int = 1,
            num_latents: int = 3,
            use_Former: str = '3d',
            timestep: int = 20,
            max_length: int = 20,
            extract_layer_idx: int = 1,
            use_all_layer: bool = False,
            obs_seq_len: int = 1,
            action_dim: int = 7,
            action_seq_len: int = 10,
    ):
        super(VPP_Policy, self).__init__()
        self.latent_dim = latent_dim
        self.use_all_layer = use_all_layer
        self.use_position_encoding = use_position_encoding
        self.t5_model_path = t5_model_path
        self.language_goal_path = language_goal_path
        self.act_window_size = act_window_size
        self.action_dim = action_dim
        self.timestep = timestep  # 20, 正确的改去噪步骤输出视频从这里改
        self.extract_layer_idx = extract_layer_idx  # 1
        self.use_Former = use_Former  # '3d'
        self.Former_num_time_embeds = Former_num_time_embeds  # 16
        self.max_length = max_length  # 20

        condition_dim_list = [1280,1280,1280,640]
        sum_dim = 0
        for i in range(extract_layer_idx+1):
            sum_dim = sum_dim + condition_dim_list[i+1]
        condition_dim = condition_dim_list[extract_layer_idx+1] if not self.use_all_layer else sum_dim

        self.use_original_diffusion_policy = False
        
        if use_Former=='3d':  # True
            self.Video_Former = Video_Former_3D(
                dim=latent_dim,
                depth=Former_depth,
                dim_head=Former_dim_head,
                heads=Former_heads,
                num_time_embeds=Former_num_time_embeds,
                num_latents=num_latents,
                condition_dim=condition_dim,
                use_temporal=True,
            )

        print('use_Former:', self.use_Former)
        print('use_all_layer',self.use_all_layer)

        self.seed = seed
        self.use_lr_scheduler = use_lr_scheduler

        # TODO whether to use gripper
        self.use_gripper = True

        self.use_univla = True 

        # goal encoders
        # self.language_goal = LangClip(model_name='ViT-B/32').to(self.device)
        self.language_goal = LangClip(
            model_name=self.language_goal_path).to(
                self.device)

        pipeline, tokenizer, feature_extractor, train_scheduler, vae_processor, text_encoder, vae, unet = load_primary_models(
            pretrained_model_path , eval = True)

        text_encoder = CLIPTextModelWithProjection.from_pretrained(text_encoder_path)
        tokenizer = AutoTokenizer.from_pretrained(text_encoder_path, use_fast=False)

        text_encoder = text_encoder.to(self.device).eval()

        for param in pipeline.image_encoder.parameters():
            param.requires_grad = False
        for param in text_encoder.parameters():
            param.requires_grad = False

        for param in pipeline.vae.parameters():
            param.requires_grad = False
        for param in pipeline.unet.parameters():
            param.requires_grad = False

        pipeline = pipeline.to(self.device)
        pipeline.unet.eval()

        self.TVP_encoder = Diffusion_feature_extractor(pipeline=pipeline,
                                                        tokenizer=tokenizer,
                                                        text_encoder=text_encoder,
                                                        position_encoding = self.use_position_encoding)
        self.TVP_encoder = self.TVP_encoder.to(self.device)

        if not self.use_original_diffusion_policy and self.use_univla:
            self.goal_emb = nn.Sequential(
                nn.Linear(condition_dim, 768),
                nn.GELU(),
                nn.Linear(768, 768)
            )
            self.goal_emb = self.goal_emb.to(self.device)
            self.time_pos_emb = nn.Parameter(torch.randn(2, 1, 768))

        # policy network
        if self.use_univla:
            self.model = GCDenoiser(action_dim = action_dim,
                                    obs_dim=latent_dim,
                                    goal_dim=512,
                                    num_tokens=num_latents,
                                    goal_window_size = 1,
                                    obs_seq_len = obs_seq_len,
                                    act_seq_len = action_seq_len,
                                    device=self.device,
                                    use_original_diffusion_policy=False,
                                    sigma_data=0.5).to(self.device)
            self.lam = UncontrolledDINOLatentActionModel(t5_model_path=t5_model_path)

        self.optimizer_config = optimizer
        self.lr_scheduler = lr_scheduler
        self.save_hyperparameters()

        # diffusion stuff
        self.sampler_type = sampler_type  # 'ddim'
        self.num_sampling_steps = num_sampling_steps  # 10
        self.noise_scheduler = noise_scheduler  # 'exponential'
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_sample_density_type = sigma_sample_density_type

        # for inference
        self.rollout_step_counter = 0
        self.multistep = multistep  # 10
        self.latent_goal = None
        self.plan = None
        self.use_text_not_embedding = use_text_not_embedding  # True

        # for clip loss ground truth plot
        self.ema_callback_idx = None

        for param in self.model.inner_model.proprio_emb.parameters():
            param.requires_grad = False
        for param in self.model.inner_model.goal_emb.parameters():
            param.requires_grad = False
        self.model.inner_model.pos_emb.requires_grad = False

    def process_device(self):
        self.TVP_encoder.pipeline = self.TVP_encoder.pipeline.to(self.device)
        self.TVP_encoder.text_encoder = self.TVP_encoder.text_encoder.to(self.device)

    def configure_optimizers(self):
        """
        Initialize optimizers and learning rate schedulers based on model configuration.
        """
        # Configuration for models using transformer weight decay
        '''optim_groups = self.action_decoder.model.inner_model.get_optim_groups(
            weight_decay=self.optimizer_config.transformer_weight_decay
        )'''
        if self.use_univla:  # True
            optim_groups = [
                {"params": self.model.inner_model.parameters(),
                "weight_decay": self.optimizer_config.transformer_weight_decay},
                {"params": self.lam.parameters(),
                "weight_decay": self.optimizer_config.transformer_weight_decay},
                {"params": self.Video_Former.parameters(), 
                "weight_decay": self.optimizer_config.transformer_weight_decay},
                {"params": self.goal_emb.parameters(), 
                "weight_decay": self.optimizer_config.transformer_weight_decay},
                {"params": [self.time_pos_emb], 
                "weight_decay": self.optimizer_config.transformer_weight_decay},
            ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.optimizer_config.learning_rate,
                                      betas=self.optimizer_config.betas)

        # 如果不用 scheduler，那 AdamW 就直接用这个 lr。
        # 一旦用了 TriStageLRScheduler，实际 step 中的 lr 都来自 scheduler。
        
        # Optionally initialize the scheduler
        if self.use_lr_scheduler:  # True
            lr_configs = OmegaConf.create(self.lr_scheduler)
            scheduler = TriStageLRScheduler(optimizer, lr_configs)
            lr_scheduler = {
                "scheduler": scheduler,
                "interval": 'step',
                "frequency": 1,
            }
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        else:
            return optimizer

    def training_step(self, dataset_batch: Dict[str, Dict],) -> torch.Tensor:  # type: ignore
        """
        Compute and return the training loss for the MDT Agent.
        The training loss consists of the score matching loss of the diffusion model
        and the contrastive loss of the CLIP model for the multimodal encoder.

        Args:
            batch: Dictionary containing the batch data for each modality.
            batch_idx: Index of the batch. used for compatibility with pytorch lightning.
            dataloader_idx: Index of the dataloader. used for compatibility with pytorch lightning.

        Returns:
            loss tensor
        """
        total_loss, action_loss = (
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
        )

        predictive_feature, latent_goal= self.extract_predictive_feature(dataset_batch)
        # predictive_feature['state_images'] torch.Size([28, 224, 384])
        # latent_goal torch.Size([28, 1, 512])

        if self.use_univla:
            univla_out = self.lam(
                predictive_feature['state_images'],
                dataset_batch['lang_text'],
            )
            latent_motion_tokens_up = univla_out['video_action_patches'].squeeze(1)
            act_loss, sigmas, noise = self.diffusion_loss(
                predictive_feature,
                latent_goal,
                dataset_batch["actions"],
                latent_motion_tokens_up,
            )

        action_loss += act_loss
        total_loss += act_loss

        total_bs = dataset_batch["actions"].shape[0]

        if self.use_univla:
            q_loss_u = ((univla_out["emb"].detach() - univla_out["z"]) ** 2).mean()
            commit_loss_u = ((univla_out["emb"] - univla_out["z"].detach()) ** 2).mean()

            total_loss = total_loss + q_loss_u + 0.25*commit_loss_u

        self._log_training_metrics(action_loss, total_loss, total_bs)
        return total_loss

    def extract_predictive_feature(self, dataset_batch):
        """
        Compute the required embeddings for the visual ones and the latent goal.
        """
        rgb_static = dataset_batch["rgb_obs"]['rgb_static'].to(self.device)  # torch.Size([28, 1, 3, 256, 256])
        rgb_gripper = dataset_batch["rgb_obs"]['rgb_gripper'].to(self.device)  # torch.Size([28, 1, 3, 256, 256])
        modality = "lang"
        if self.use_text_not_embedding:  # True
            latent_goal = self.language_goal(dataset_batch["lang_text"]).to(rgb_static.dtype)  # torch.Size([28, 1, 512])
        else:
            latent_goal = self.language_goal(dataset_batch["lang"]).to(rgb_static.dtype)

        language = dataset_batch["lang_text"]

        num_frames = self.Former_num_time_embeds  # 16
        rgb_static = rgb_static.to(self.device)
        rgb_gripper = rgb_gripper.to(self.device)
        batch = rgb_static.shape[0]  # 28

        with torch.no_grad():
            input_rgb = torch.cat([rgb_static, rgb_gripper], dim=0)
            language = language + language
            perceptual_features = self.TVP_encoder(input_rgb, language, self.timestep,
                                                           self.extract_layer_idx, all_layer=self.use_all_layer,
                                                           step_time=1, max_length=self.max_length)  # torch.Size([56, 16, 2560, 16, 16])

        perceptual_features = einops.rearrange(perceptual_features, 'b f c h w-> b f c (h w)')  # torch.Size([56, 16, 2560, 256])
        perceptual_features = einops.rearrange(perceptual_features, 'b f c l-> b f l c')  # torch.Size([56, 16, 256, 2560])
        perceptual_features = perceptual_features[:, :num_frames, :, :]  # torch.Size([56, 16, 256, 2560])

        perceptual_features, gripper_feature = torch.split(perceptual_features, [batch, batch], dim=0)
        
        #### use gripper (above) or donotuse gripper (below) ###
        if self.use_gripper:
            perceptual_features_for_videoformer = torch.cat([perceptual_features, gripper_feature], dim=2)
        else:
            perceptual_features_for_videoformer = perceptual_features

        perceptual_features_for_videoformer = perceptual_features_for_videoformer.to(torch.float32)

        perceptual_features = perceptual_features.to(torch.float32)  # torch.Size([28, 16, 512, 2560])

        # TODO 在这儿改frame, train过程
        frame_0 = perceptual_features[:, 0]      # shape: [28, 256, 2560]
        frame_5 = perceptual_features[:, 0+5]      # shape: [28, 256, 2560]
        perceptual_features = torch.stack([frame_0, frame_5], dim=1)  # shape [28, 2, 256, 2560]
        perceptual_features = self.goal_emb(perceptual_features)  # torch.Size([28, 2, 256, 1024])
        time_pos_emb = (
            self.time_pos_emb.unsqueeze(0).expand(perceptual_features.size(0), -1, -1, -1)
        )  # torch.Size([28, 2, 1, 384])
        perceptual_features = perceptual_features + time_pos_emb  # torch.Size([28, 2, 512, 384])

        perceptual_features_for_videoformer = self.Video_Former(perceptual_features_for_videoformer)  # torch.Size([28, 224(16*14), 384]) torch.Size([28, 28, 384])
        if self.use_Former=='linear':  # False
            perceptual_features = rearrange(perceptual_features, 'b T q d -> b (T q) d')
        predictive_feature = {
            'state_images': perceptual_features,
            'state_images_for_videoformer': perceptual_features_for_videoformer,
        }
        predictive_feature['modality'] = modality
        return predictive_feature, latent_goal

    def _log_training_metrics(self, action_loss, total_loss, total_bs):
        """
        Log the training metrics.
        """
        self.log("train/action_loss", action_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=total_bs)
        self.log("train/total_loss", total_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=total_bs)

    def diffusion_loss(
            self,
            perceptual_emb: torch.Tensor,
            latent_goal: torch.Tensor,
            actions: torch.Tensor,
            latent_motion_tokens_up: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Computes the score matching loss given the perceptual embedding, latent goal, and desired actions.
        """
        self.model.train()
        sigmas = self.make_sample_density()(shape=(len(actions),), device=self.device).to(self.device)
        noise = torch.randn_like(actions).to(self.device)  # torch.Size([28, 10, 7])
        loss, _ = self.model.loss(perceptual_emb, actions, latent_goal, noise, sigmas,
                                  latent_motion_tokens_up)
        return loss, sigmas, noise

    def denoise_actions(  # type: ignore
            self,
            latent_plan: torch.Tensor,
            perceptual_emb: torch.Tensor,
            latent_goal: torch.Tensor,
            latent_motion_tokens_up: torch.Tensor = None,
            inference: Optional[bool] = False,
            extra_args={}
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Denoise the next sequence of actions
        """
        if inference:
            sampling_steps = self.num_sampling_steps  # 10
        else:
            sampling_steps = 10
        self.model.eval()
        if len(latent_goal.shape) < len(
                perceptual_emb['state_images'].shape if isinstance(perceptual_emb, dict) else perceptual_emb.shape):
            latent_goal = latent_goal.unsqueeze(1)  # torch.Size([28, 1, 512])
        input_state = perceptual_emb
        sigmas = self.get_noise_schedule(sampling_steps, self.noise_scheduler)

        # 随机初始化action
        x = torch.randn((len(latent_goal), self.act_window_size, self.action_dim), device=self.device) * self.sigma_max
        # torch.Size([28, 10, 7])

        actions = self.sample_loop(sigmas, latent_motion_tokens_up,
                                   x, input_state, latent_goal, latent_plan, self.sampler_type, extra_args)

        return actions  # torch.Size([28, 10, 7])

    def make_sample_density(self):
        """
        Generate a sample density function based on the desired type for training the model
        We mostly use log-logistic as it has no additional hyperparameters to tune.
        """
        sd_config = []
        if self.sigma_sample_density_type == 'lognormal':
            loc = self.sigma_sample_density_mean  # if 'mean' in sd_config else sd_config['loc']
            scale = self.sigma_sample_density_std  # if 'std' in sd_config else sd_config['scale']
            return partial(utils.rand_log_normal, loc=loc, scale=scale)

        if self.sigma_sample_density_type == 'loglogistic':  # True
            loc = sd_config['loc'] if 'loc' in sd_config else math.log(self.sigma_data)
            scale = sd_config['scale'] if 'scale' in sd_config else 0.5
            min_value = sd_config['min_value'] if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(utils.rand_log_logistic, loc=loc, scale=scale, min_value=min_value, max_value=max_value)

        if self.sigma_sample_density_type == 'loguniform':
            min_value = sd_config['min_value'] if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(utils.rand_log_uniform, min_value=min_value, max_value=max_value)

        if self.sigma_sample_density_type == 'uniform':
            return partial(utils.rand_uniform, min_value=self.sigma_min, max_value=self.sigma_max)

        if self.sigma_sample_density_type == 'v-diffusion':
            min_value = self.min_value if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(utils.rand_v_diffusion, sigma_data=self.sigma_data, min_value=min_value, max_value=max_value)
        
        if self.sigma_sample_density_type == 'discrete':
            sigmas = self.get_noise_schedule(self.num_sampling_steps * 1e5, 'exponential')
            return partial(utils.rand_discrete, values=sigmas)
        
        if self.sigma_sample_density_type == 'split-lognormal':
            loc = sd_config['mean'] if 'mean' in sd_config else sd_config['loc']
            scale_1 = sd_config['std_1'] if 'std_1' in sd_config else sd_config['scale_1']
            scale_2 = sd_config['std_2'] if 'std_2' in sd_config else sd_config['scale_2']
            return partial(utils.rand_split_log_normal, loc=loc, scale_1=scale_1, scale_2=scale_2)
        else:
            raise ValueError('Unknown sample density type')

    def sample_loop(
            self,
            sigmas,
            latent_motion_tokens_up,
            x_t: torch.Tensor,
            state: torch.Tensor,
            goal: torch.Tensor,
            latent_plan: torch.Tensor,
            sampler_type: str,
            extra_args={},
    ):
        """
        Main method to generate samples depending on the chosen sampler type. DDIM is the default as it works well in all settings.
        """
        s_churn = extra_args['s_churn'] if 's_churn' in extra_args else 0
        s_min = extra_args['s_min'] if 's_min' in extra_args else 0
        use_scaler = extra_args['use_scaler'] if 'use_scaler' in extra_args else False
        keys = ['s_churn', 'keep_last_actions']
        if bool(extra_args):
            reduced_args = {x: extra_args[x] for x in keys}
        else:
            reduced_args = {}
        if use_scaler:
            scaler = self.scaler
        else:
            scaler = None

        if sampler_type == 'lms':
            x_0 = sample_lms(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True, extra_args=reduced_args)
        elif sampler_type == 'heun':
            x_0 = sample_heun(self.model, state, x_t, goal, sigmas, scaler=scaler, s_churn=s_churn, s_tmin=s_min,
                              disable=True)
        elif sampler_type == 'euler':
            x_0 = sample_euler(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'ancestral':
            x_0 = sample_dpm_2_ancestral(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'euler_ancestral':
            x_0 = sample_euler_ancestral(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'dpm':
            x_0 = sample_dpm_2(self.model, state, x_t, goal, sigmas, disable=True)
        elif sampler_type == 'dpm_adaptive':
            x_0 = sample_dpm_adaptive(self.model, state, x_t, goal, sigmas[-2].item(), sigmas[0].item(), disable=True)
        elif sampler_type == 'dpm_fast':
            x_0 = sample_dpm_fast(self.model, state, x_t, goal, sigmas[-2].item(), sigmas[0].item(), len(sigmas),
                                  disable=True)
        elif sampler_type == 'dpmpp_2s_ancestral':
            x_0 = sample_dpmpp_2s_ancestral(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'dpmpp_2m':
            x_0 = sample_dpmpp_2m(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'dpmpp_2m_sde':
            x_0 = sample_dpmpp_sde(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'ddim':  # True
            x_0 = sample_ddim(self.model, latent_motion_tokens_up,
                              state, x_t, goal, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'dpmpp_2s':
            x_0 = sample_dpmpp_2s(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'dpmpp_2_with_lms':
            x_0 = sample_dpmpp_2_with_lms(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        else:
            raise ValueError('desired sampler type not found!')
        return x_0  # torch.Size([28, 10, 7])

    def get_noise_schedule(self, n_sampling_steps, noise_schedule_type):
        """
        Get the noise schedule for the sampling steps. Describes the distribution over the noise levels from sigma_min to sigma_max.
        """
        if noise_schedule_type == 'karras':
            return get_sigmas_karras(n_sampling_steps, self.sigma_min, self.sigma_max, 7,
                                     self.device)
        elif noise_schedule_type == 'exponential':  # True
            return get_sigmas_exponential(n_sampling_steps, self.sigma_min, self.sigma_max, self.device)
        elif noise_schedule_type == 'vp':
            return get_sigmas_vp(n_sampling_steps, device=self.device)
        elif noise_schedule_type == 'linear':
            return get_sigmas_linear(n_sampling_steps, self.sigma_min, self.sigma_max, device=self.device)
        elif noise_schedule_type == 'cosine_beta':
            return cosine_beta_schedule(n_sampling_steps, device=self.device)
        elif noise_schedule_type == 've':
            return get_sigmas_ve(n_sampling_steps, self.sigma_min, self.sigma_max, device=self.device)
        elif noise_schedule_type == 'iddpm':
            return get_iddpm_sigmas(n_sampling_steps, self.sigma_min, self.sigma_max, device=self.device)
        raise ValueError('Unknown noise schedule type')

    def reset(self):
        """
        Call this at the beginning of a new rollout when doing inference.
        """
        self.plan = None
        self.latent_goal = None
        self.rollout_step_counter = 0

    def forward(self,batch):  # This is used when training the model.
        return self.training_step(batch)

    def eval_forward(self, obs, goal):
        """
        Method for doing inference with the model.
        """
        if 'lang_text' in goal:
            if self.use_text_not_embedding:  # true
                latent_goal = self.language_goal(goal["lang_text"])  # torch.Size([28, 1, 512])
                latent_goal = latent_goal.to(torch.float32)
            else:
                latent_goal = self.language_goal(goal["lang"]).unsqueeze(0).to(torch.float32).to(
                    obs["rgb_obs"]['rgb_static'].device)

        rgb_static = obs["rgb_obs"]['rgb_static']  # torch.Size([28, 1, 3, 256, 256])
        rgb_gripper = obs["rgb_obs"]['rgb_gripper']  # torch.Size([28, 1, 3, 256, 256])

        language = goal["lang_text"]

        num_frames = self.Former_num_time_embeds  # Video lenth 16
        rgb_static = rgb_static.to(self.device)
        rgb_gripper = rgb_gripper.to(self.device)
        batch = rgb_static.shape[0]

        # 1. Predictive Visual Representations Learning
        with torch.no_grad():
            input_rgb = torch.cat([rgb_static, rgb_gripper], dim=0)  # torch.Size([56, 1, 3, 256, 256])
            language = [language] + [language]
            perceptual_features = self.TVP_encoder(input_rgb, language, self.timestep,  # torch.Size([56, 16, 2560, 16, 16])
                                                           self.extract_layer_idx, all_layer=self.use_all_layer,
                                                           step_time=1, max_length=self.max_length)
        
        # 这里将两个视角的图像按batch维度拼接在一起, 过TVP, 然后再按channel维度拼接在一起

        # 2. Action Learning
        perceptual_features = einops.rearrange(perceptual_features, 'b f c h w-> b f c (h w)')  # torch.Size([56, 16, 2560, 256])
        perceptual_features = einops.rearrange(perceptual_features, 'b f c l-> b f l c')  # torch.Size([56, 16, 256, 2560])
        perceptual_features = perceptual_features[:, :num_frames, :, :]  # torch.Size([56, 16, 256, 2560])

        perceptual_features, gripper_feature = torch.split(perceptual_features, [batch, batch], dim=0)
        
        #### use gripper (above) or donotuse gripper (below) ###
        if self.use_gripper:
            perceptual_features_for_videoformer = torch.cat([perceptual_features, gripper_feature], dim=2)
        else:
            perceptual_features_for_videoformer = perceptual_features

        perceptual_features_for_videoformer = perceptual_features_for_videoformer.to(torch.float32)

        perceptual_features = perceptual_features.to(torch.float32)

        # TODO 在这儿改frame, eval过程
        frame_0 = perceptual_features[:, 0]      # shape: [28, 256, 2560]
        frame_5 = perceptual_features[:, 0+5]      # shape: [28, 256, 2560]
        perceptual_features = torch.stack([frame_0, frame_5], dim=1)  # shape [28, 2, 256, 2560]
        perceptual_features = self.goal_emb(perceptual_features)  # torch.Size([28, 2, 256, 1024])
        time_pos_emb = (
            self.time_pos_emb.unsqueeze(0).expand(perceptual_features.size(0), -1, -1, -1)
        )  # torch.Size([28, 2, 1, 384])
        perceptual_features = perceptual_features + time_pos_emb  # torch.Size([28, 2, 512, 384])

        # 2.1 Video Former
        perceptual_features_for_videoformer = self.Video_Former(perceptual_features_for_videoformer)  # torch.Size([28, 224, 384])
        if self.use_Former == 'linear':  # False
            perceptual_features = rearrange(perceptual_features, 'b T q d -> b (T q) d')

        perceptual_emb = {
            'state_images': perceptual_features,
            'state_images_for_videoformer': perceptual_features_for_videoformer,
        }  # torch.Size([28, 224, 384])

        perceptual_emb['modality'] = "lang"

        # 2.2 DiT Diffusion Policy
        if self.use_univla:
            univla_out = self.lam(
                perceptual_emb['state_images'],
                goal["lang_text"],  # UniVLA stage 1 need to add, while stage 2 not need
            )
            latent_motion_tokens_up = univla_out['video_action_patches'].squeeze(1)
            perceptual_emb['state_images'] = perceptual_emb['state_images'].reshape(
                perceptual_emb['state_images'].size(0), -1, perceptual_emb['state_images'].size(-1)
            )
            act_seq = self.denoise_actions(
                torch.zeros_like(latent_goal).to(latent_goal.device),
                perceptual_emb,
                latent_goal,
                latent_motion_tokens_up,
                inference=True,
            )
        return act_seq  # torch.Size([28, 10, 7])

    def step(self, obs, goal):  # This is used when rollouting in inference.
        """
        Do one step of inference with the model. THis method handles the action chunking case.
        Our model is trained to predict a sequence of actions.
        We only compute the sequence once every self.multistep steps.

        Args:
            obs (dict): Observation from environment.
            goal (dict): Goal as visual observation or embedded language instruction.

        Returns:
            Predicted action.
        """
        # 一次 rollout 10步, 机械臂执行10步之后再 rollout 下一次
        if self.rollout_step_counter % self.multistep == 0:
            pred_action_seq = self.eval_forward(obs, goal)  # torch.Size([28, 10, 7])

            self.pred_action_seq = pred_action_seq

        current_action = self.pred_action_seq[0, self.rollout_step_counter]  # torch.Size([7])
        if len(current_action.shape) == 2:
            current_action = einops.rearrange(current_action, 'b d -> b 1 d')
        self.rollout_step_counter += 1
        if self.rollout_step_counter == self.multistep:
            self.rollout_step_counter = 0

        return current_action

    def on_train_start(self) -> None:
        if not self.use_original_diffusion_policy and self.use_univla:
            self.lam.to(dtype=self.dtype)
        self.Video_Former.to(dtype=self.dtype)
        self.language_goal.to(dtype=self.dtype)
        self.TVP_encoder.to(dtype=self.dtype)

    @rank_zero_only
    def on_train_epoch_start(self) -> None:
        logger.info(f"Start training epoch {self.current_epoch}")

    @rank_zero_only
    def on_train_epoch_end(self, unused: Optional = None) -> None:  # type: ignore
        logger.info(f"Finished training epoch {self.current_epoch}")

    @rank_zero_only
    def on_validation_epoch_end(self) -> None:
        logger.info(f"Finished validation epoch {self.current_epoch}")

    def on_validation_epoch_start(self) -> None:
        log_rank_0(f"Start validation epoch {self.current_epoch}")

    @rank_zero_only
    def on_train_epoch_start(self) -> None:
        logger.info(f"Start training epoch {self.current_epoch}")

    @rank_zero_only
    def on_train_epoch_end(self, unused: Optional = None) -> None:  # type: ignore
        logger.info(f"Finished training epoch {self.current_epoch}")

    @rank_zero_only
    def on_validation_epoch_end(self) -> None:
        logger.info(f"Finished validation epoch {self.current_epoch}")

    def on_validation_epoch_start(self) -> None:
        log_rank_0(f"Start validation epoch {self.current_epoch}")


@rank_zero_only
def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    logger.info(*args, **kwargs)
