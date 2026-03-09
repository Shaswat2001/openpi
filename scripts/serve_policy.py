"""OpenPI policy server with deterministic seeding.

This is your serve_policy.py with two changes:
  1. Added --seed CLI argument
  2. Wraps the policy in SeededPolicy before handing it to the WebSocket server

The seed can be changed at runtime by the client — it sends "__seed__": N
in the observation dict on the first infer() after each episode reset.
"""

import dataclasses
import enum
import logging
import socket

import tyro

import logging
import random
from typing import Any

import numpy as np


from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config

# Import our seeded wrapper.
# Option A: if you placed seeded_policy.py in src/openpi/policies/:
#   from openpi.policies.seeded_policy import SeededPolicy, seed_global_rngs
# Option B: if you placed it next to this script:

def seed_global_rngs(seed: int) -> None:
    """Seed Python, NumPy, and (optionally) PyTorch global RNGs."""
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
    """Deterministic wrapper around an openpi Policy.

    Usage on server side (in serve_policy.py):
        raw_policy = create_policy(args)
        policy = SeededPolicy(raw_policy, seed=42)
        # Then pass `policy` to the WebSocket server as usual.

    The client triggers a reseed by including "__seed__": <int> in the
    observation dict passed to infer().  This field is detected, consumed,
    and the RNG is reset before the model runs.
    """

    def __init__(self, policy, seed: int = 0):
        self._policy = policy

        # Detect if this is a PyTorch model
        self._is_pytorch = getattr(policy, '_is_pytorch', False)

        # Auto-detect the JAX PRNG attribute name on the policy object.
        # Common names across openpi versions: _rng, rng, _rng_key
        self._rng_attr = None
        if not self._is_pytorch:
            for attr in ('_rng', 'rng', '_rng_key', 'rng_key'):
                if hasattr(policy, attr):
                    self._rng_attr = attr
                    logging.info("SeededPolicy: found JAX PRNG attr '%s'", attr)
                    break
            if self._rng_attr is None:
                logging.warning(
                    "SeededPolicy: could not find JAX PRNG attr on Policy. "
                    "Run on your machine:\n"
                    "  grep -n 'rng' src/openpi/policies/policy.py\n"
                    "to find the correct attribute name."
                )

        self.reset(seed)

    def reset(self, seed: int | None = None) -> None:
        """Reset the RNG state. Called at the start of each episode."""
        if seed is not None:
            self._seed = seed
        self._call_count = 0
        seed_global_rngs(self._seed)

        if not self._is_pytorch:
            import jax
            self._base_key = jax.random.key(self._seed)

        logging.info("SeededPolicy.reset(seed=%d)", self._seed)

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Run one inference step with deterministic RNG.

        If obs contains "__seed__", reseed first, then remove it.
        """
        # ---- Check for client-injected seed ----
        if "__seed__" in obs:
            new_seed = int(obs.pop("__seed__"))
            logging.info("SeededPolicy: received seed=%d from client", new_seed)
            self.reset(seed=new_seed)

        # ---- Set deterministic RNG for this step ----
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
        return self._policy.infer(obs)

    @property
    def metadata(self) -> dict[str, Any]:
        return self._policy.metadata

    def __getattr__(self, name):
        """Forward everything else to the wrapped policy."""
        return getattr(self._policy, name)


class EnvMode(enum.Enum):
    """Supported environments."""
    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""
    config: str
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""
    env: EnvMode = EnvMode.ALOHA_SIM
    default_prompt: str | None = None
    port: int = 8000
    record: bool = False
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)
    # ---- NEW: initial seed for deterministic inference ----
    seed: int = 0


# Default checkpoints for each environment.
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


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config),
            checkpoint.dir,
            default_prompt=default_prompt,
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config),
                args.policy.dir,
                default_prompt=args.default_prompt,
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def main(args: Args) -> None:
    # 1. Seed global RNGs before model loading
    seed_global_rngs(args.seed)

    # 2. Create the raw policy (unchanged)
    raw_policy = create_policy(args)
    policy_metadata = raw_policy.metadata

    # 3. ---- NEW: wrap in SeededPolicy ----
    policy = SeededPolicy(raw_policy, seed=args.seed)

    # 4. Optional recording
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info(
        "Creating server (host: %s, ip: %s, seed: %d)",
        hostname, local_ip, args.seed,
    )

    # 5. Start the WebSocket server (unchanged)
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