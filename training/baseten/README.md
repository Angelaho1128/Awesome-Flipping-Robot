# Self-contained Baseten training source

Physical settings: `settings.yaml`. Launch `baseten train push --config config_smoke.py`, then `config.py` only after smoke passes. Both commands start remote jobs. Read the repository root README for exact dimensions, motor-source qualifications, launch commands and remaining integration work.

Only pancake diameter (95–105 mm) and initial XY (each ±10 mm) vary. Training uses the `hardware` profile; no independently sampled motor or arm-length profiles remain. Do not resume old checkpoints.

This folder is independently uploadable: it contains the model generator, Gym environment, Warp backend, PPO training, tests, dependency pins and cloud configuration. It does not import repository-root camera or robot modules. GPU/CPU parity remains a required startup gate. Model construction and reference validation use cloud CPUs; batched toss physics and the learner use CUDA.
