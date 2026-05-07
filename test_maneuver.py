import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mappo.envs import get_environment


warnings.filterwarnings("ignore", category=UserWarning, module="torch._utils")


def main():
    kwargs = {"hierarchy_level": 0, "training_level": 2}
    env = get_environment("aircombat", "track/L1_track", False, kwargs)
    act = {"B1": [10, 5, 5, 5, 1]}

    env.reset()
    done = False

    while not done:
        env.env.render(viewpoint=0, trajectory_view=False)
        _, _, terminated, truncated, _ = env.step(act)
        done = terminated["all"] or truncated["all"]

    env.close()


if __name__ == "__main__":
    main()
