"""Read-only local metric audit; no ML runtime."""
import argparse
import json
from dynamics_shift.utils.tracking_audit import audit
if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    print(json.dumps(audit(parser.parse_args().run_dir), indent=2))
