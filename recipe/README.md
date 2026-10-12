# Actor algorithms

- `gigpo/core.py`: GiGPO shared-state step advantages and trajectory returns.
- `hgpo/core.py`: HGPO history-conditioned hierarchical advantages and returns.
- GRPO: `verl/trainer/ppo/core_algos.py`.
- `dapo/`: dynamic sampling and group filtering used by the training runtime.
  Select `--method dapo` to use GRPO advantages, asymmetric clipping, token-mean
  loss, and dynamic sampling with actor KL loss disabled. The method profile is
  `configs/actor/methods/dapo.yaml`. Multi-turn sampling filters groups by the
  configured population reward standard deviation; its final refill attempt
  retains unfiltered groups, preserving the existing sampler's fallback.

All actor algorithms use `scripts/train.sh --kind actor --method METHOD`.
Method profiles are in `configs/actor/methods/`. Baseline profiles disable the
TTG meter and do not require its checkpoint. They share the task/model profiles;
the resolved configuration specifies each run's training budget.

## Source provenance

GiGPO and HGPO are based on verl-agent commit
`20bd331bdbc9026a5668e11362178e10ab7400c8`.
GiGPO uses its shared-state advantage implementation. The HGPO adapter includes
trajectory-aware discounted returns and history-conditioned grouping.
