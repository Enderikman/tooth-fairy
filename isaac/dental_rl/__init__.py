"""Reinforcement learning for the dental suction task ("dental-suction-hold").

The policy drives an SO-101 that holds the saliva ejector: the ejector tip must reach a target
point inside the lower-jaw corridor (between teeth and tongue) and hold it there, without ever
entering the forbidden zone around the teeth.

Modules
-------
corridor.py  batched torch version of jaw_sim's corridor rule (reward, zone penalty, targets)
env.py       the Isaac Lab DirectRLEnv and a thin rsl_rl VecEnv view of it
train.py     PPO training with rsl_rl, exports TorchScript policy.pt
play.py      rolls the trained policy out and writes rollout.json for the web viewer

Everything that defines the mouth itself comes from jaw_sim; this package never redefines it.
"""
