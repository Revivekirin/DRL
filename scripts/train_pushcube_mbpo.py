"""Deprecated alias: use scripts/train_mbpo.py."""
from dynamics_shift.experiments.cli import train_main
def main():
    train_main('mbpo')
if __name__ == "__main__":
    main()
