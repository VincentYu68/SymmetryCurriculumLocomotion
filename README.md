# Learning Symmetric and Low-energy Locomotion

This is code for our paper: https://arxiv.org/abs/1801.08093

## Setup

The code consists of two parts: dart-env, which is an extension of OpenAI Gym that uses Dart for rigid-body simulation, and baselines, which is adapted from OpenAI Baselines.

To install dart-env, follow the instructions at: https://github.com/DartEnv/dart-env/wiki.

To install baselines, execute the following:

```bash
cd baselines
pip install -e .
```


## How to use

To test the code on a biped walking robot, run the following command from the project directory:

```bash
mpirun -np 8 python -m baselines.ppo1.run_walker3d_staged_learning
```

The training results will be saved to data/. The final policy is saved as policy_params.pkl. You can also find the intermediate policies in the folders organized by the corresponding curriculums. To test a policy, run:

```bash
python test_policy.py ENV_NAME PATH_TO_POLICY
```

To visualize the learning curve, run:

```bash
python plot_benchmark.py PATH_TO_FOLDER
```



### Setup environment

4 example environments are included: [DartWalker3d-v1](dart-env/gym/envs/dart/walker3d.py), [DartHumanWalker-v1](dart-env/gym/envs/dart/human_walker.py), [DartDogRobot-v1](dart-env/gym/envs/dart/dog_robot.py) and [DartHexapod-v1](dart-env/gym/envs/dart/hexapod.py).

The desired velocity is controlled by three variables in the initialization of each environment: **init_tv** sets the target velocity at the beginning of the rollout, **final_tv** sets the target velocity we want the character to reach eventually, and **tv_endtime** sets the amount of time (in seconds) it takes to accelerate from **init_tv** to **final_tv**.

### Setup training script

Refer to [run_walker3d_staged_learning.py](baselines/baselines/ppo1/run_walker3d_staged_learning.py) for an example on how to setup the training script for the biped walking robot.

#### Mirror-symmetry 

The mirror-symmetry loss for a new environment is configured with the argument **observation_permutation** and **action_permutation** when initializing MlpMirrorPolicy in the training script.

For **observation_permutation** and **action_permutation**, they are two vectors used for mirror symmetric loss. Each entry in these two denotes the index of the corresponding entry in observation/action AFTER it is mirrored w.r.t the sagittal plane of the character, and the sign of the element means whether the entry should multiply -1 after mirroring. For example, if a character has its left and right elbow joint angle at index 4 and 7 of the obsevation vector, then observation_permutation[4] should be 7 and observation_permutation[7] should be 4. Further, if the behavior of the two dofs are opposite, e.g. larger value of left elbow angle means flexion while larger value of right elbow angle means extension, then a -1 should be multiplied to both entries in observation_permutation. Note that for dofs at the center of the character (like pelvis rotation), their corresponding mirrored entry are simply themselves, with -1 multiplied to some of them. Also, if the entry at index 0 need to be negated, you need to use a small negative value like -0.0001, as multiplying -1 wouldn't change 0.

## Additional notes

For a newly created dart-env environment, you can use [examine_skel.py](baselines/examine_skel.py) to test the model configurations, which I found to be helpful in debugging joint limits.
