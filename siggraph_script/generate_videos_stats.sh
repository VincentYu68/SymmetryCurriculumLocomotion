python baselines/test_policy.py DartWalker3d-v1 data/ppo_DartWalker3d-v10_walk/policy_params.pkl 1 0
python baselines/test_policy.py DartWalker3d-v1 data/ppo_DartWalker3d-v10_run/policy_params.pkl 1 0

python baselines/test_policy.py DartHumanWalker-v1 data/ppo_DartHumanWalker-v10_walk/policy_params.pkl 1 0
python baselines/test_policy.py DartHumanWalker-v1 data/ppo_DartHumanWalker-v10_walk_back/policy_params.pkl 1 0
python baselines/test_policy.py DartHumanWalker-v1 data/ppo_DartHumanWalker-v10_run/policy_params.pkl 1 0

python baselines/test_policy.py DartDogRobot-v1 data/ppo_DartDogRobot-v10_walk/policy_params.pkl 1 0
python baselines/test_policy.py DartDogRobot-v1 data/ppo_DartDogRobot-v10_run/policy_params.pkl 1 0

python baselines/test_policy.py DartHexapod-v1 data/ppo_DartHexapod-v10_walk/policy_params.pkl 1 0
python baselines/test_policy.py DartHexapod-v1 data/ppo_DartHexapod-v10_run/policy_params.pkl 1 0

python baselines/plot_curriculum.py data/ppo_DartWalker3d-v10_walk/
python baselines/plot_curriculum.py data/ppo_DartWalker3d-v10_run/
python baselines/plot_curriculum.py data/ppo_DartHumanWalker-v10_walk/
python baselines/plot_curriculum.py data/ppo_DartHumanWalker-v10_walk_back/
python baselines/plot_curriculum.py data/ppo_DartHumanWalker-v10_run/