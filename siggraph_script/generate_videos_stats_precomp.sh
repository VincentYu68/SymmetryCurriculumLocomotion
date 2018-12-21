python baselines/test_policy.py DartWalker3d-v1 data/precomp_data/walker3d_walk/policy_params.pkl 1 0
python baselines/test_policy.py DartWalker3d-v1 data/precomp_data/walker3d_run/policy_params.pkl 1 0

python baselines/test_policy.py DartHumanWalker-v1 data/precomp_data/human_walk/policy_params.pkl 1 0
python baselines/test_policy.py DartHumanWalker-v1 data/precomp_data/human_walk_back/policy_params.pkl 1 0
python baselines/test_policy.py DartHumanWalker-v1 data/precomp_data/human_run/policy_params.pkl 1 0

python baselines/test_policy.py DartDogRobot-v1 data/precomp_data/dog_walk/policy_params.pkl 1 0
python baselines/test_policy.py DartDogRobot-v1 data/precomp_data/dog_run1/policy_params.pkl 1 0
python baselines/test_policy.py DartDogRobot-v1 data/precomp_data/dog_run2/policy_params.pkl 1 0

python baselines/test_policy.py DartHexapod-v1 data/precomp_data/hexapod_walk/policy_params.pkl 1 0
python baselines/test_policy.py DartHexapod-v1 data/precomp_data/hexapod_run/policy_params.pkl 1 0

python baselines/plot_curriculum.py data/precomp_data/walker3d_walk/
python baselines/plot_curriculum.py data/precomp_data/walker3d_run/
python baselines/plot_curriculum.py data/precomp_data/human_walk/
python baselines/plot_curriculum.py data/precomp_data/human_walk_back/
python baselines/plot_curriculum.py data/precomp_data/human_run/