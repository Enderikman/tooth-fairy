# Charter

## Vision

A robot arm that takes over the dental assistant's suction work: during drilling it holds the
suction tube in the patient's mouth and removes the drill's cooling water, so the dentist can
work without a second pair of hands at the chair.

## Mission

For this hackathon (Nebius x NVIDIA), using the Hugging Face / TheRobotStudio SO-101 arm:

1. Simulate first: an SO-101 holding a rigid suction tube moves the tip through a model mouth,
   from the lower arch to the upper arch and back, without touching teeth.
2. Transfer the same joint-target interface to the real SO-101 follower arm.
3. Demonstrate the suction tip following a target in a dental phantom (training model or printed jaw).

Out of scope for the hackathon: work on real patients, medical certification, actual suction hardware control.
