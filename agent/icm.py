import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

import utils
from agent.ddpg import DDPGAgent

CROSSMODAL_DECODE_TYPES = [
    "spectrogram_simple",
    # "spectrogram_fusion",
    "spectrogram_noisy",
]


class ICM(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.forward_net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim),
        )

        self.backward_net = nn.Sequential(
            nn.Linear(2 * obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

        self.apply(utils.weight_init)

    def forward(self, obs, action, next_obs):
        assert obs.shape[0] == next_obs.shape[0]
        assert obs.shape[0] == action.shape[0]

        next_obs_hat = self.forward_net(torch.cat([obs, action], dim=-1))
        action_hat = self.backward_net(torch.cat([obs, next_obs], dim=-1))

        forward_error = torch.norm(next_obs - next_obs_hat, dim=-1, p=2, keepdim=True)
        backward_error = torch.norm(action - action_hat, dim=-1, p=2, keepdim=True)

        return forward_error, backward_error


class CrossModalPredictor(nn.Module):
    def __init__(self, obs_dim, cross_obs_dim, hidden_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_dim, cross_obs_dim),
        )
        self.apply(utils.weight_init)

    def forward(self, obs):
        cross_obs_hat = self.net(obs)
        return cross_obs_hat


class Acuracy:
    def __init__(self):
        self._positive = 0
        self._negative = 0

    def add(self, p, n):
        self._positive += p
        self._negative += n

    @property
    def total(self):
        return self._positive + self._negative

    def report_p(self):
        return self._positive / self.total

    def report_n(self):
        return self._negative / self.total


class ICMAgent(DDPGAgent):
    def __init__(self, icm_scale, crossmodal_scale, omit_crossmodal_loss, **kwargs):
        super().__init__(**kwargs)
        self.icm_scale = icm_scale
        self.crossmodal_scale = crossmodal_scale
        self.omit_crossmodal_loss = omit_crossmodal_loss
        self.icm = ICM(self.obs_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.cross_acc = None

        if self.obs_type in ("spectrogram_simple",):
            self.crossmodal_predictor = CrossModalPredictor(
                self.obs_dim, self.cross_obs_dim, self.hidden_dim // 8
            ).to(self.device)
            _weight = torch.Tensor([1, 100]).to(self.device)
            self.crossmodal_predictor_criterion = nn.CrossEntropyLoss(weight=_weight)
            self.crossmodal_predictor_opt = torch.optim.RAdam(
                self.crossmodal_predictor.parameters(), lr=self.lr
            )
            self.crossmodal_predictor.train()
            self.cross_acc = Acuracy()
        elif self.obs_type in ("spectrogram_noisy",):
            self.crossmodal_predictor = CrossModalPredictor(
                self.obs_dim, self.cross_obs_dim, self.hidden_dim // 8
            ).to(self.device)
            self.crossmodal_predictor_criterion = nn.MSELoss()
            self.crossmodal_predictor_opt = torch.optim.RAdam(
                self.crossmodal_predictor.parameters(), lr=self.lr
            )
            self.crossmodal_predictor.train()
        else:
            self.crossmodal_predictor = None
            self.crossmodal_predictor_opt = None

        # optimizers
        self.icm_opt = torch.optim.RAdam(self.icm.parameters(), lr=self.lr)

        self.icm.train()

    def update_crossmodal_predictor(self, obs, cross_obs, step):
        metrics = dict()
        cross_obs_hat = self.crossmodal_predictor(obs)
        self.crossmodal_predictor_opt.zero_grad(set_to_none=True)
        loss = self.crossmodal_predictor_criterion(cross_obs_hat, cross_obs)
        loss *= self.crossmodal_scale

        if self.use_tb or self.use_wandb:
            metrics["crossmodal_loss"] = loss.item()
            if self.cross_acc:
                is_silence_hat = cross_obs_hat.data[:, 0] > 0.5
                judge = (is_silence_hat == cross_obs.bool()).cpu().numpy()
                p = judge.sum()
                n = judge.shape[0] - p
                self.cross_acc.add(p, n)
                metrics["crossmodal_acc"] = self.cross_acc.report_p()

        return metrics, loss

    def update_icm(self, obs, action, next_obs, step):
        metrics = dict()

        forward_error, backward_error = self.icm(obs, action, next_obs)

        loss = forward_error.mean() + backward_error.mean()
        loss *= self.icm_scale
        loss *= 0.2

        self.icm_opt.zero_grad(set_to_none=True)

        if self.use_tb or self.use_wandb:
            metrics["icm_loss"] = loss.item()

        return metrics, loss

    def compute_intr_reward(self, obs, action, next_obs, step):
        forward_error, _ = self.icm(obs, action, next_obs)
        # error = forward_error * self.icm_scale
        reward = torch.log(forward_error + 1.0)
        reward *= self.icm_scale
        return reward

    def compute_intr_cross_reward(self, obs, cross_obs, step):
        cross_obs_hat = self.crossmodal_predictor(obs)
        error = self.crossmodal_predictor_criterion(cross_obs_hat, cross_obs)
        # error = error * self.crossmodal_scale
        reward = torch.log(error + 1.0)
        reward *= self.crossmodal_scale
        return reward

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, extr_reward, discount, next_obs = utils.to_torch(
            batch, self.device
        )

        # augment and encode
        obs = self.aug_and_encode(obs)
        with torch.no_grad():
            next_obs = self.aug_and_encode(next_obs)

        if self.obs_type in CROSSMODAL_DECODE_TYPES:
            obs, cross_obs = obs
            next_obs, next_cross_obs = next_obs

        if self.reward_free:
            if self.encoder_opt is not None:
                self.encoder_opt.zero_grad(set_to_none=True)
            if self.crossmodal_predictor is not None:
                crossmodal_metrics, crossmodal_loss = self.update_crossmodal_predictor(
                    obs, cross_obs, step
                )
                metrics.update(crossmodal_metrics)
                with torch.no_grad():
                    intr_cross_reward = self.compute_intr_cross_reward(
                        obs, cross_obs, step
                    )
                if self.use_tb or self.use_wandb:
                    metrics["intr_cross_reward"] = intr_cross_reward.mean().item()
            else:
                crossmodal_loss = 0
                intr_cross_reward = 0
            icm_metrics, icm_loss = self.update_icm(obs, action, next_obs, step)
            metrics.update(icm_metrics)
            if self.omit_crossmodal_loss:
                loss = icm_loss
            else:
                loss = icm_loss + crossmodal_loss  # already scaled
            loss.backward()
            if self.crossmodal_predictor_opt is not None:
                self.crossmodal_predictor_opt.step()
            self.icm_opt.step()
            if self.encoder_opt is not None:
                if self.clip_grad_value > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.encoder.parameters(), self.clip_grad_value
                    )
                self.encoder_opt.step()

            with torch.no_grad():
                intr_reward = self.compute_intr_reward(obs, action, next_obs, step)

            if self.use_tb or self.use_wandb:
                metrics["intr_reward"] = intr_reward.mean().item()
            reward = intr_reward + intr_cross_reward  # already scaled

        else:
            reward = extr_reward

        if self.use_tb or self.use_wandb:
            metrics["extr_reward"] = extr_reward.mean().item()
            metrics["batch_reward"] = reward.mean().item()

        if not self.update_encoder:
            obs = obs.detach()
            next_obs = next_obs.detach()

        # update critic
        metrics.update(
            self.update_critic(
                obs.detach(), action, reward, discount, next_obs.detach(), step
            )
        )

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(
            self.critic, self.critic_target, self.critic_target_tau
        )

        return metrics
