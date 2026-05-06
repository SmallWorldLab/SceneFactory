from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDict
from torch import nn
from torch.distributions import Normal

from rsl_rl.modules.actor_critic import ActorCritic, EmpiricalNormalization


def _activation_module(name: str) -> nn.Module:
    key = str(name).strip().lower()
    if key == "relu":
        return nn.ReLU()
    if key == "tanh":
        return nn.Tanh()
    if key == "elu":
        return nn.ELU()
    if key == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported late-fusion activation: {name!r}")


def _make_mlp(input_dim: int, layers: list[int], activation: str, dropout: float) -> nn.Sequential:
    modules: list[nn.Module] = []
    last_dim = int(input_dim)
    for width in layers:
        modules.append(nn.Linear(last_dim, int(width)))
        modules.append(nn.LayerNorm(int(width)))
        if float(dropout) > 0.0:
            modules.append(nn.Dropout(float(dropout)))
        modules.append(_activation_module(activation))
        last_dim = int(width)
    return nn.Sequential(*modules)


def _branch_out_dim(input_dim: int, layers: list[int]) -> int:
    if int(input_dim) <= 0:
        return 0
    return int(layers[-1]) if layers else int(input_dim)


class _StructuredLateFusionBackbone(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        *,
        ego_dim: int,
        road_point_dim: int,
        road_point_k: int,
        vehicle_dim: int,
        vehicle_k: int,
        ego_layers: list[int],
        road_layers: list[int],
        vehicle_layers: list[int],
        shared_layers: list[int],
        last_layer_dim_pi: int,
        last_layer_dim_vf: int,
        activation: str,
        dropout: float,
        pool: str,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.ego_dim = int(ego_dim)
        self.road_point_dim = max(0, int(road_point_dim))
        self.road_point_k = max(0, int(road_point_k))
        self.vehicle_dim = max(0, int(vehicle_dim))
        self.vehicle_k = max(0, int(vehicle_k))
        self.pool = str(pool).strip().lower()
        if self.pool not in {"max", "mean"}:
            raise ValueError(f"Unsupported late-fusion pool: {pool!r}")

        self.latent_dim_pi = int(last_layer_dim_pi)
        self.latent_dim_vf = int(last_layer_dim_vf)
        self.ego_out_dim = _branch_out_dim(self.ego_dim, ego_layers)
        self.road_out_dim = _branch_out_dim(self.road_point_dim, road_layers)
        self.vehicle_out_dim = _branch_out_dim(self.vehicle_dim, vehicle_layers)

        self.ego_net_actor = _make_mlp(self.ego_dim, ego_layers, activation, dropout)
        self.ego_net_critic = _make_mlp(self.ego_dim, ego_layers, activation, dropout)
        self.road_net_actor = (
            _make_mlp(self.road_point_dim, road_layers, activation, dropout) if self.road_point_dim > 0 else None
        )
        self.road_net_critic = (
            _make_mlp(self.road_point_dim, road_layers, activation, dropout) if self.road_point_dim > 0 else None
        )
        self.vehicle_net_actor = (
            _make_mlp(self.vehicle_dim, vehicle_layers, activation, dropout) if self.vehicle_dim > 0 else None
        )
        self.vehicle_net_critic = (
            _make_mlp(self.vehicle_dim, vehicle_layers, activation, dropout) if self.vehicle_dim > 0 else None
        )

        shared_in = self.ego_out_dim + self.road_out_dim + self.vehicle_out_dim
        self.actor_shared = _make_mlp(shared_in, shared_layers, activation, dropout)
        self.critic_shared = _make_mlp(shared_in, shared_layers, activation, dropout)

        actor_in = int(shared_layers[-1]) if shared_layers else int(shared_in)
        critic_in = int(shared_layers[-1]) if shared_layers else int(shared_in)
        self.actor_last = nn.Linear(actor_in, self.latent_dim_pi)
        self.critic_last = nn.Linear(critic_in, self.latent_dim_vf)

    @staticmethod
    def _slice_with_padding(features: torch.Tensor, start: int, length: int) -> torch.Tensor:
        if int(length) <= 0:
            return features.new_zeros((features.shape[0], 0))
        end = min(features.shape[1], start + length)
        chunk = features[:, start:end]
        if chunk.shape[1] == int(length):
            return chunk
        pad = features.new_zeros((features.shape[0], int(length) - chunk.shape[1]))
        return torch.cat([chunk, pad], dim=1)

    def _split_obs(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ego = self._slice_with_padding(features, 0, self.ego_dim)
        offset = self.ego_dim

        road_len = self.road_point_k * self.road_point_dim
        road_flat = self._slice_with_padding(features, offset, road_len)
        offset += road_len

        vehicle_len = self.vehicle_k * self.vehicle_dim
        vehicle_flat = self._slice_with_padding(features, offset, vehicle_len)

        if self.road_point_k > 0 and self.road_point_dim > 0:
            road_points = road_flat.reshape(-1, self.road_point_k, self.road_point_dim)
        else:
            road_points = features.new_zeros((features.shape[0], 0, max(1, self.road_point_dim)))

        if self.vehicle_k > 0 and self.vehicle_dim > 0:
            vehicles = vehicle_flat.reshape(-1, self.vehicle_k, self.vehicle_dim)
        else:
            vehicles = features.new_zeros((features.shape[0], 0, max(1, self.vehicle_dim)))

        return ego, road_points, vehicles

    def _pool_entities(self, entity_inputs: torch.Tensor, entity_emb: torch.Tensor) -> torch.Tensor:
        if entity_inputs.numel() == 0 or entity_emb.numel() == 0:
            return entity_emb.new_zeros((entity_emb.shape[0], entity_emb.shape[-1]))
        valid = torch.any(torch.abs(entity_inputs) > 1.0e-6, dim=-1)
        if self.pool == "mean":
            weights = valid.to(entity_emb.dtype).unsqueeze(-1)
            denom = torch.clamp(weights.sum(dim=1), min=1.0)
            return (entity_emb * weights).sum(dim=1) / denom
        masked = entity_emb.masked_fill(~valid.unsqueeze(-1), float("-inf"))
        pooled = masked.max(dim=1).values
        empty = ~valid.any(dim=1)
        if empty.any():
            pooled[empty] = 0.0
        return pooled

    def _encode_branch(self, inputs: torch.Tensor, net: nn.Module | None, out_dim: int) -> torch.Tensor:
        if out_dim <= 0 or net is None or inputs.numel() == 0 or inputs.shape[1] == 0:
            return inputs.new_zeros((inputs.shape[0], out_dim))
        flat = inputs.reshape(-1, inputs.shape[-1])
        emb = net(flat).reshape(inputs.shape[0], inputs.shape[1], -1)
        return self._pool_entities(inputs, emb)

    def forward_actor_latent(self, features: torch.Tensor) -> torch.Tensor:
        ego, road_points, vehicles = self._split_obs(features)
        ego_emb = self.ego_net_actor(ego)
        road_emb = self._encode_branch(road_points, self.road_net_actor, self.road_out_dim)
        vehicle_emb = self._encode_branch(vehicles, self.vehicle_net_actor, self.vehicle_out_dim)
        fused = torch.cat([ego_emb, road_emb, vehicle_emb], dim=1)
        return self.actor_last(self.actor_shared(fused))

    def forward_critic_latent(self, features: torch.Tensor) -> torch.Tensor:
        ego, road_points, vehicles = self._split_obs(features)
        ego_emb = self.ego_net_critic(ego)
        road_emb = self._encode_branch(road_points, self.road_net_critic, self.road_out_dim)
        vehicle_emb = self._encode_branch(vehicles, self.vehicle_net_critic, self.vehicle_out_dim)
        fused = torch.cat([ego_emb, road_emb, vehicle_emb], dim=1)
        return self.critic_last(self.critic_shared(fused))


class _ActorHead(nn.Module):
    def __init__(self, backbone: _StructuredLateFusionBackbone, num_actions: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(int(backbone.latent_dim_pi), int(num_actions))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone.forward_actor_latent(obs))


class _CriticHead(nn.Module):
    def __init__(self, backbone: _StructuredLateFusionBackbone) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(int(backbone.latent_dim_vf), 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone.forward_critic_latent(obs))


class SceneFactoryLateFusionActorCritic(ActorCritic):
    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        actor_hidden_dims: list[int] | tuple[int, ...] | None = None,
        critic_hidden_dims: list[int] | tuple[int, ...] | None = None,
        activation: str = "relu",
        ego_dim: int = 11,
        road_point_dim: int = 5,
        road_point_k: int = 0,
        vehicle_dim: int = 7,
        vehicle_k: int = 0,
        ego_layers: list[int] | None = None,
        road_layers: list[int] | None = None,
        vehicle_layers: list[int] | None = None,
        shared_layers: list[int] | None = None,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
        dropout: float = 0.0,
        pool: str = "max",
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "SceneFactoryLateFusionActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        if state_dependent_std:
            raise ValueError("SceneFactoryLateFusionActorCritic does not support state-dependent std.")

        nn.Module.__init__(self)
        self.obs_groups = obs_groups
        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            num_actor_obs += obs[obs_group].shape[-1]
        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            num_critic_obs += obs[obs_group].shape[-1]
        if int(num_actor_obs) != int(num_critic_obs):
            raise ValueError(
                "SceneFactoryLateFusionActorCritic expects identical actor/critic observation dimensions, "
                f"got actor={num_actor_obs}, critic={num_critic_obs}."
            )

        ego_layers = list(ego_layers or [64, 64])
        road_layers = list(road_layers or [96, 96])
        vehicle_layers = list(vehicle_layers or [96, 96])
        shared_layers = list(shared_layers or [128, 64])
        if actor_hidden_dims or critic_hidden_dims:
            print(
                "SceneFactoryLateFusionActorCritic ignores actor_hidden_dims/critic_hidden_dims and uses "
                "late-fusion branch settings instead."
            )

        self.state_dependent_std = False
        self.actor_obs_normalization = bool(actor_obs_normalization)
        self.critic_obs_normalization = bool(critic_obs_normalization)
        self.actor_obs_normalizer = (
            EmpiricalNormalization(int(num_actor_obs)) if self.actor_obs_normalization else nn.Identity()
        )
        self.critic_obs_normalizer = (
            EmpiricalNormalization(int(num_critic_obs)) if self.critic_obs_normalization else nn.Identity()
        )

        self.backbone = _StructuredLateFusionBackbone(
            int(num_actor_obs),
            ego_dim=int(ego_dim),
            road_point_dim=int(road_point_dim),
            road_point_k=int(road_point_k),
            vehicle_dim=int(vehicle_dim),
            vehicle_k=int(vehicle_k),
            ego_layers=ego_layers,
            road_layers=road_layers,
            vehicle_layers=vehicle_layers,
            shared_layers=shared_layers,
            last_layer_dim_pi=int(last_layer_dim_pi),
            last_layer_dim_vf=int(last_layer_dim_vf),
            activation=str(activation),
            dropout=float(dropout),
            pool=str(pool),
        )
        self.actor = _ActorHead(self.backbone, int(num_actions))
        self.critic = _CriticHead(self.backbone)
        print(
            "[INFO][SceneFactory] Late-fusion policy initialized "
            f"ego_dim={ego_dim} road_point_dim={road_point_dim} road_point_k={road_point_k} "
            f"vehicle_dim={vehicle_dim} vehicle_k={vehicle_k}",
            flush=True,
        )

        self.noise_std_type = str(noise_std_type)
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(float(init_noise_std) * torch.ones(int(num_actions)))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(float(init_noise_std) * torch.ones(int(num_actions))))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type!r}")

        self.distribution = None
        Normal.set_default_validate_args(False)
