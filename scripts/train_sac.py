"""Common SAC entry point; task and backend come from config."""
from dynamics_shift.experiments.cli import train_main
if __name__ == '__main__':
    train_main('sac')
