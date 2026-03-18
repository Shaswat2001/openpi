"""OpenPI policy server with deterministic seeding + true GPU-batched REST endpoint.

Changes from the original:
  1. --seed CLI argument + SeededPolicy wrapper (existing)
  2. NEW: FastAPI REST server on port+1 with /act_batch endpoint
     - Preprocesses N observations individually (transforms are per-sample)
     - Stacks into one batch tensor (batch_dim = N)
     - Runs ONE GPU forward pass (model.sample_actions)
     - Splits results and postprocesses individually
     - Returns N action chunks in one HTTP response

Usage:
  python serve_policy.py --env LIBERO --seed 42 --port 8000
  # WebSocket on :8000 (unchanged)
  # REST /act_batch on :8001 (new)
"""

import dataclasses
import enum
import logging
import socket
import threading
import time

import tyro

import random
from typing import Any, Dict, List

import numpy as np

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config
from openpi.models import model as _model


# ─────────────────────────────────────────────────────────────────────
# Seeding utilities
# ─────────────────────────────────────────────────────────────────────

def seed_global_rngs(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


class SeededPolicy:
    """Deterministic wrapper around an openpi Policy with batch support."""

    def __init__(self, policy, seed: int = 0):
        self._policy = policy
        self._is_pytorch = getattr(policy, '_is_pytorch_model', False)
        self._rng_attr = None
        if not self._is_pytorch:
            for attr in ('_rng', 'rng', '_rng_key', 'rng_key'):
                if hasattr(policy, attr):
                    self._rng_attr = attr
                    logging.info("SeededPolicy: found JAX PRNG attr '%s'", attr)
                    break
        self.reset(seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._seed = seed
        self._call_count = 0
        seed_global_rngs(self._seed)
        if not self._is_pytorch:
            import jax
            self._base_key = jax.random.key(self._seed)
        logging.info("SeededPolicy.reset(seed=%d)", self._seed)

    def _set_rng_for_step(self):
        """Set deterministic RNG state for the current call_count."""
        if self._is_pytorch:
            import torch
            step_seed = self._seed + self._call_count
            torch.manual_seed(step_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(step_seed)
        else:
            if self._rng_attr is not None:
                import jax
                key = jax.random.fold_in(self._base_key, self._call_count)
                setattr(self._policy, self._rng_attr, key)
        self._call_count += 1

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Single observation inference (used by WebSocket server)."""
        if "__seed__" in obs:
            self.reset(seed=int(obs.pop("__seed__")))
        self._set_rng_for_step()
        return self._policy.infer(obs)

    def infer_batch(self, obs_list: List[dict[str, Any]]) -> List[dict[str, Any]]:
        """
        True GPU-batched inference:
          1. Input transforms per sample (these are CPU-side normalization/resize)
          2. Stack N samples into one batch tensor
          3. ONE model.sample_actions() call with batch_dim=N
          4. Split batch, output transforms per sample

        This mirrors Policy.infer() exactly but replaces [None, ...] (batch=1)
        with a proper N-way stack.
        """
        # Handle per-obs seeding
        for obs in obs_list:
            if "__seed__" in obs:
                self.reset(seed=int(obs.pop("__seed__")))
        self._set_rng_for_step()

        policy = self._policy
        N = len(obs_list)

        # ── Step 1: per-sample input transforms ─────────────────────
        import jax as _jax
        transformed = []
        for obs in obs_list:
            inputs = _jax.tree.map(lambda x: x, obs)  # shallow copy
            inputs = policy._input_transform(inputs)
            transformed.append(inputs)

        # ── Step 2: stack into batch of N ───────────────────────────
        # Use tree.map to handle nested dicts (e.g. "images": {"cam": array})
        if policy._is_pytorch_model:
            import torch
            import jax as _jax

            def _stack_torch(*samples):
                tensors = [torch.from_numpy(np.array(s)).to(policy._pytorch_device)
                           for s in samples]
                return torch.stack(tensors, dim=0)

            batched = _jax.tree.map(_stack_torch, *transformed)
            sample_rng_or_device = policy._pytorch_device
        else:
            import jax
            import jax.numpy as jnp

            def _stack_jax(*samples):
                return jnp.stack([jnp.asarray(s) for s in samples], axis=0)

            batched = jax.tree.map(_stack_jax, *transformed)
            policy._rng, sample_rng_or_device = jax.random.split(policy._rng)

        # ── Step 3: ONE GPU forward pass ────────────────────────────
        observation = _model.Observation.from_dict(batched)
        sample_kwargs = dict(policy._sample_kwargs)

        # Log shapes to verify batching is correct
        for k, v in batched.items():
            if hasattr(v, 'shape'):
                logging.info("  batched[%s].shape = %s", k, v.shape)

        start_time = time.monotonic()
        actions = policy._sample_actions(
            sample_rng_or_device, observation, **sample_kwargs,
        )
        # actions shape: [N, action_horizon, action_dim]
        model_time = time.monotonic() - start_time

        if hasattr(actions, 'shape'):
            logging.info("  actions.shape = %s", actions.shape)

        # ── Step 4: split batch → per-sample numpy ──────────────────
        import jax as _jax

        results = []
        for i in range(N):
            if policy._is_pytorch_model:
                out_state = _jax.tree.map(
                    lambda x: np.asarray(x[i].detach().cpu()), batched["state"]
                )
                out_actions = np.asarray(actions[i].detach().cpu())
            else:
                out_state = _jax.tree.map(
                    lambda x: np.asarray(x[i]), batched["state"]
                )
                out_actions = np.asarray(actions[i])

            out = {"state": out_state, "actions": out_actions}

            # ── Step 5: per-sample output transforms ────────────────
            out = policy._output_transform(out)
            out["policy_timing"] = {"infer_ms": model_time * 1000 / N}
            results.append(out)

        logging.info(
            "infer_batch: N=%d, model_time=%.0fms (%.0fms/sample)",
            N, model_time * 1000, model_time * 1000 / N,
        )
        return results

    @property
    def metadata(self) -> dict[str, Any]:
        return self._policy.metadata

    def __getattr__(self, name):
        return getattr(self._policy, name)


# ─────────────────────────────────────────────────────────────────────
# REST batch endpoint
# ─────────────────────────────────────────────────────────────────────

def start_rest_server(policy: SeededPolicy, port: int) -> None:
    """FastAPI server with /act_batch doing true GPU-batched inference."""
    import json_numpy
    json_numpy.patch()

    from fastapi import FastAPI
    from starlette.responses import Response
    import uvicorn

    app = FastAPI()

    @app.post("/act_batch")
    def act_batch(payload: Dict[str, Any]) -> Response:
        observations: List[Dict] = payload["observations"]
        results = policy.infer_batch(observations)
        return Response(
            content=json_numpy.dumps(results),
            media_type="application/json",
        )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    logging.info("Starting REST server on port %d (GPU-batched endpoint)", port)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

class EnvMode(enum.Enum):
    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    config: str
    dir: str


@dataclasses.dataclass
class Default:
    pass


@dataclasses.dataclass
class Args:
    env: EnvMode = EnvMode.ALOHA_SIM
    default_prompt: str | None = None
    port: int = 8000
    record: bool = False
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)
    seed: int = 0
    rest_port: int | None = None


DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(env, *, default_prompt=None):
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config),
            checkpoint.dir,
            default_prompt=default_prompt,
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args):
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config),
                args.policy.dir,
                default_prompt=args.default_prompt,
            )
        case Default():
            return create_default_policy(
                args.env, default_prompt=args.default_prompt,
            )


def main(args: Args) -> None:
    seed_global_rngs(args.seed)
    raw_policy = create_policy(args)
    policy_metadata = raw_policy.metadata
    policy = SeededPolicy(raw_policy, seed=args.seed)

    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    rest_port = args.rest_port or (args.port + 1)

    logging.info(
        "Server: host=%s, ip=%s, seed=%d, ws_port=%d, rest_port=%d",
        hostname, local_ip, args.seed, args.port, rest_port,
    )

    rest_thread = threading.Thread(
        target=start_rest_server,
        args=(policy, rest_port),
        daemon=True,
    )
    rest_thread.start()
    logging.info("REST batch endpoint at http://%s:%d/act_batch", local_ip, rest_port)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))