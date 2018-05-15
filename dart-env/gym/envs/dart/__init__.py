from gym.envs.dart.dart_env import DartEnv
# ^^^^^ so that user gets the correct error
# message if Dart is not installed correctly

from gym.envs.dart.walker3d import DartWalker3dEnv
from gym.envs.dart.dog_robot import DartDogRobotEnv
from gym.envs.dart.human_walker import DartHumanWalkerEnv
from gym.envs.dart.hexapod import DartHexapodEnv