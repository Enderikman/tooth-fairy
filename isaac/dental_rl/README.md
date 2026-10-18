# dental-suction-hold (PPO in Isaac Lab)

The policy moves the five SO-101 arm joints so that the saliva ejector tip reaches a target point
inside the lower-jaw corridor (between teeth and tongue) and holds it there. The jaw is not a
physics body: `corridor.py` evaluates jaw_sim's corridor rule for all environments at once, which
supplies the targets and the forbidden-zone penalty. Observations are state only (joint positions
and velocities, tip, target, target - tip, previous action), actions are joint position deltas.

Sync, then run on the GPU VM from the repo root (`conda activate leisaac`,
`export OMNI_KIT_ACCEPT_EULA=YES LEISAAC_ASSETS_ROOT=$HOME/hackathon/leisaac/assets`):

    rsync -a --exclude .venv --exclude out --exclude .git --exclude skills --exclude isaac/.vendor . isaac:~/hackathonNebius_2026/
    python -m isaac.dental_rl.train --headless                                   # runs/<timestamp>/
    python -m isaac.dental_rl.play --checkpoint isaac/dental_rl/runs/<run>/policy.pt --headless --out rollout.json

`train.py` writes tensorboard logs, `model_<it>.pt` checkpoints and the TorchScript actor
`policy.pt`. The number that matters is not the loss but `Episode/success_rate`: the share of
episodes in which the tip stayed within 15 mm of the target for 25 control steps (0.5 s) in a row.
`play.py` prints the same success rate plus the mean final tip error for the four tooth targets and
writes `rollout.json` for the web viewer. All lengths are at demo scale: the jaw is drawn
`JAW_SCALE` (3x) larger than life, so 15 mm here is the 5 mm tolerance of a real mouth.
The rollout the web viewer reads is copied back to `runs/latest/rollout.json` on the Mac.
