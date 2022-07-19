from collections import OrderedDict
import torchaudio
from typing import Tuple

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from typing import Mapping


class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35  # for ?x84x84 input (default by urlb)
        # self.repr_dim = 32 * 57 * 57 # for ?x128x128 input

        self.convnet = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
        )

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)  # size will be used as self.repr_dim
        return h


class FusionEncoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert isinstance(obs_shape, Mapping)
        for k, v in obs_shape.items():
            assert len(v) == 3
        self.convnet_pixels = nn.Sequential(
            nn.Conv2d(obs_shape["pixels"][0], 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
        )
        self.convnet_spectrogram = nn.Sequential(
            nn.Conv2d(obs_shape["spectrogram"][0], 16, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, stride=1),
            nn.ReLU(),  # size is 8x11x11
            nn.MaxPool2d(3, 3),  # now size is 8x3x3
        )
        self.obs_repr_dim = 32 * 35 * 35
        self.cross_obs_repr_dim = 8 * 3 * 3
        self.repr_dim = self.obs_repr_dim + self.cross_obs_repr_dim
        # self.fc = nn.Sequential(nn.Linear(fusion_dim, self.repr_dim), nn.ReLU())
        self.apply(utils.weight_init)

    def forward(self, obs):
        if isinstance(obs, Mapping):
            pixels, spectrogram = obs["pixels"], obs["spectrogram"]
            pixels = pixels / 255.0 - 0.5
            h_pixels = self.convnet_pixels(pixels)  # shape: 1x32x35x35
            h_pixels = h_pixels.view(h_pixels.shape[0], -1)  # shape: 1x...
            h_spectrogram = self.convnet_spectrogram(spectrogram)  # shape: 1x8x3x3
            h_spectrogram = h_spectrogram.view(h_spectrogram.shape[0], -1)
            h = torch.cat((h_pixels, h_spectrogram), axis=1)  # output with fusion_dim
            return h
        pixels = obs / 255.0 - 0.5
        h_pixels = self.convnet_pixels(pixels)  # shape: 1x32x35x35
        h = h_pixels.view(h_pixels.shape[0], -1)  # shape: 1x...
        return h


class SimpleCrossEncoder(nn.Module):
    def __init__(self, obs_shape, threshold=1):
        super().__init__()
        self.threshold = threshold

        assert isinstance(obs_shape, Mapping)
        for k, v in obs_shape.items():
            assert len(v) == 3
        self.convnet_pixels = nn.Sequential(
            nn.Conv2d(obs_shape["pixels"][0], 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
        )
        self.obs_repr_dim = 32 * 35 * 35
        self.cross_obs_repr_dim = 2
        self.repr_dim = self.obs_repr_dim
        # self.fc = nn.Sequential(nn.Linear(fusion_dim, self.repr_dim), nn.ReLU())
        self.apply(utils.weight_init)

    def forward(self, obs):
        if isinstance(obs, Mapping):
            obs, obs_cross = obs["pixels"], obs["spectrogram"]
            h_cross = (obs_cross > -0.99).sum(
                dim=[1, 2, 3]
            ) > self.threshold  # -1 is min of spect
            pixels = obs / 255.0 - 0.5
            h_pixels = self.convnet_pixels(pixels)  # shape: 1x32x35x35
            h = h_pixels.view(h_pixels.shape[0], -1)  # shape: 1x...
            return h, h_cross.long()
        pixels = obs / 255.0 - 0.5
        h_pixels = self.convnet_pixels(pixels)  # shape: 1x32x35x35
        h = h_pixels.view(h_pixels.shape[0], -1)  # shape: 1x...
        return h


class CrossEncoder(FusionEncoder):
    def forward(self, obs):
        if isinstance(obs, Mapping):
            pixels, spectrogram = obs["pixels"], obs["spectrogram"]
            pixels = pixels / 255.0 - 0.5
            h_pixels = self.convnet_pixels(pixels)  # shape: 1x32x35x35
            h_pixels = h_pixels.view(h_pixels.shape[0], -1)  # shape: 1x...
            h_spectrogram = self.convnet_spectrogram(spectrogram)  # shape: 1x8x3x3
            h_spectrogram = h_spectrogram.view(h_spectrogram.shape[0], -1)
            return h_pixels, h_spectrogram
        pixels = obs / 255.0 - 0.5
        h_pixels = self.convnet_pixels(pixels)  # shape: 1x32x35x35
        h = h_pixels.view(h_pixels.shape[0], -1)  # shape: 1x...
        return h


class AddGaussianNoise:
    def __init__(self, mean=0.0, std=1.0, device="cpu"):
        self.std = std
        self.mean = mean
        self.device = device

    def __call__(self, tensor):
        return (
            tensor
            + torch.randn(tensor.size(), device=self.device) * self.std
            + self.mean
        )

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


class NoisyCrossEncoder(nn.Module):
    def __init__(self, obs_shape, noiser):
        super().__init__()

        assert isinstance(obs_shape, Mapping)
        for k, v in obs_shape.items():
            assert len(v) == 3
        self.convnet_pixels = nn.Sequential(
            nn.Conv2d(obs_shape["pixels"][0], 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
        )
        self.convnet_spectrogram = nn.Sequential(
            nn.Conv2d(obs_shape["spectrogram"][0], 16, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(8, 4, 3, stride=1),
            nn.MaxPool2d(3, stride=3),
        )
        self.obs_repr_dim = 32 * 35 * 35
        self.cross_obs_repr_dim = 4 * 3 * 3  # if use max_pool2d else 484
        self.repr_dim = self.obs_repr_dim  # + self.cross_obs_repr_dim
        self.apply(utils.weight_init)
        self.convnet_spectrogram.requires_grad_(False)
        self.noiser = noiser

    def forward(self, obs):
        if isinstance(obs, Mapping):
            obs, obs_cross = obs["pixels"], obs["spectrogram"]
            pixels = obs / 255.0 - 0.5
            h_pixels = self.convnet_pixels(pixels)  # shape: 1x32x35x35
            h = h_pixels.view(h_pixels.shape[0], -1)  # shape: 1x...

            obs_cross = self.noiser(obs_cross)
            h_cross = self.convnet_spectrogram(obs_cross)
            h_cross = h_cross.view(h_cross.shape[0], -1)
            return h, h_cross
        pixels = obs / 255.0 - 0.5
        h_pixels = self.convnet_pixels(pixels)  # shape: 1x32x35x35
        h = h_pixels.view(h_pixels.shape[0], -1)  # shape: 1x...
        return h


class Actor(nn.Module):
    def __init__(self, obs_type, obs_dim, action_dim, feature_dim, hidden_dim):
        super().__init__()

        feature_dim = feature_dim if obs_type != "states" else hidden_dim

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh()
        )

        policy_layers = []
        policy_layers += [nn.Linear(feature_dim, hidden_dim), nn.ReLU(inplace=True)]
        # add additional hidden layer for pixels
        if obs_type != "states":
            policy_layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        policy_layers += [nn.Linear(hidden_dim, action_dim)]

        self.policy = nn.Sequential(*policy_layers)

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, obs_type, obs_dim, action_dim, feature_dim, hidden_dim):
        super().__init__()

        self.obs_type = obs_type

        if obs_type == "states":
            # for states actions come in the beginning
            self.trunk = nn.Sequential(
                nn.Linear(obs_dim + action_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Tanh(),
            )
            trunk_dim = hidden_dim
        else:
            # for pixels actions will be added after trunk
            self.trunk = nn.Sequential(
                nn.Linear(obs_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh()
            )
            trunk_dim = feature_dim + action_dim

        def make_q():
            q_layers = []
            q_layers += [nn.Linear(trunk_dim, hidden_dim), nn.ReLU(inplace=True)]
            if obs_type != "states":
                q_layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
            q_layers += [nn.Linear(hidden_dim, 1)]
            return nn.Sequential(*q_layers)

        self.Q1 = make_q()
        self.Q2 = make_q()

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        inpt = torch.cat([obs, action], dim=-1) if self.obs_type == "states" else obs
        h = self.trunk(inpt)
        h = h if self.obs_type == "states" else torch.cat([h, action], dim=-1)

        q1 = self.Q1(h)
        q2 = self.Q2(h)

        return q1, q2


class DDPGAgent:
    def __init__(
        self,
        name,
        reward_free,
        obs_type,
        obs_shape,
        action_shape,
        device,
        lr,
        feature_dim,
        hidden_dim,
        critic_target_tau,
        num_expl_steps,
        update_every_steps,
        stddev_schedule,
        nstep,
        batch_size,
        stddev_clip,
        init_critic,
        use_tb,
        use_wandb,
        meta_dim=0,
        update_encoder=True,
        clip_grad_value=0,
        do_random=False,
    ):
        self.reward_free = reward_free
        self.obs_type = obs_type
        self.obs_shape = obs_shape
        self.action_dim = action_shape[0]
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.use_wandb = use_wandb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.init_critic = init_critic
        self.feature_dim = feature_dim
        self.solved_meta = None
        self.update_encoder = update_encoder
        self.clip_grad_value = clip_grad_value
        self.do_random = do_random

        # models
        if obs_type in ("pixels", "pixels_tv"):
            self.aug = utils.RandomShiftsAug(pad=4)
            self.encoder = Encoder(obs_shape).to(device)
            self.obs_dim = self.encoder.repr_dim + meta_dim
        elif obs_type == "states":
            self.aug = nn.Identity()
            self.encoder = nn.Identity()
            self.obs_dim = obs_shape[0] + meta_dim
        elif obs_type in ("spectrogram_simple", "spectrogram_simple_tv"):
            self.aug = utils.simple_crossmodal_aug
            self.encoder = SimpleCrossEncoder(obs_shape).to(device)
            self.cross_obs_dim = self.encoder.cross_obs_repr_dim
            self.obs_dim = self.encoder.repr_dim + meta_dim
        elif obs_type in ("spectrogram_fusion", "spectrogram_fusion_tv"):
            self.aug = utils.fusion_aug
            self.encoder = FusionEncoder(obs_shape).to(device)
            self.cross_obs_dim = self.encoder.cross_obs_repr_dim
            self.obs_dim = self.encoder.repr_dim + meta_dim
        elif obs_type in ("spectrogram_noisy", "spectrogram_noisy_tv"):
            self.aug = utils.noisy_crossmodal_aug
            noiser = AddGaussianNoise(0, 0.01, device=device)
            self.encoder = NoisyCrossEncoder(obs_shape, noiser).to(device)
            self.cross_obs_dim = self.encoder.cross_obs_repr_dim
            self.obs_dim = self.encoder.repr_dim + meta_dim
        else:
            raise NotImplementedError

        self.actor = Actor(
            obs_type, self.obs_dim, self.action_dim, feature_dim, hidden_dim
        ).to(device)

        self.critic = Critic(
            obs_type, self.obs_dim, self.action_dim, feature_dim, hidden_dim
        ).to(device)
        self.critic_target = Critic(
            obs_type, self.obs_dim, self.action_dim, feature_dim, hidden_dim
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers

        if obs_type == "states" or not self.update_encoder:
            self.encoder_opt = None
        else:
            self.encoder_opt = torch.optim.RAdam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.RAdam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.RAdam(self.critic.parameters(), lr=lr)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def init_from(self, other, snapshot_rerandom=None):
        utils.hard_update_params(other.encoder, self.encoder)
        utils.hard_update_params(other.actor, self.actor)
        if self.init_critic:
            utils.hard_update_params(other.critic.trunk, self.critic.trunk)
        for part in snapshot_rerandom:
            self.__dict__[part].apply(utils.weight_init)
            print(f"Re-randomlized {part}.")
        del other

    def get_meta_specs(self):
        return tuple()

    def init_meta(self):
        return OrderedDict()

    def update_meta(self, meta, global_step, time_step, finetune=False):
        return meta

    def _obs_as_tensor(self, obs):
        if isinstance(obs, dict):
            _obs = {}
            for k, v in obs.items():
                _obs[k] = torch.as_tensor(v, device=self.device).unsqueeze(0)
            return _obs
        return torch.as_tensor(obs, device=self.device).unsqueeze(0)

    def act(self, obs, meta, step, eval_mode):
        obs = self._obs_as_tensor(obs)
        h = self.encoder(obs)
        if isinstance(h, Tuple):
            h, h_cross = h
        inputs = [h]
        for value in meta.values():
            value = torch.as_tensor(value, device=self.device).unsqueeze(0)
            inputs.append(value)
        inpt = torch.cat(inputs, dim=-1)
        # assert obs.shape[-1] == self.obs_shape[-1]
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(inpt, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps or self.do_random:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb or self.use_wandb:
            metrics["critic_target_q"] = target_Q.mean().item()
            metrics["critic_q1"] = Q1.mean().item()
            metrics["critic_q2"] = Q2.mean().item()
            metrics["critic_loss"] = critic_loss.item()

        # optimize critic
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()

        # clip!
        if self.clip_grad_value > 0:
            torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.clip_grad_value
            )

        self.critic_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()
        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()

        # clip!
        if self.clip_grad_value > 0:
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.clip_grad_value
            )

        self.actor_opt.step()

        if self.use_tb or self.use_wandb:
            metrics["actor_loss"] = actor_loss.item()
            metrics["actor_logprob"] = log_prob.mean().item()
            metrics["actor_ent"] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def aug_and_encode(self, obs):
        obs = self.aug(obs)
        return self.encoder(obs)

    def update(self, replay_iter, step):
        metrics = dict()
        # import ipdb; ipdb.set_trace()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(batch, self.device)

        # augment and encode
        obs = self.aug_and_encode(obs)
        with torch.no_grad():
            next_obs = self.aug_and_encode(next_obs)

        if self.use_tb or self.use_wandb:
            metrics["batch_reward"] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step)
        )

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(
            self.critic, self.critic_target, self.critic_target_tau
        )
        return metrics
