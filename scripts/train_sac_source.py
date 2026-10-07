"""Deprecated alias: use scripts/train_sac.py."""
from dynamics_shift.experiments.cli import train_main
def main():
    train_main('sac')
if __name__ == "__main__":
    main()
