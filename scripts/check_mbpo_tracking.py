"""Read-only local metric audit; no ML runtime."""
import argparse
import json
from dynamics_shift.utils.tracking_audit import audit
if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--report", help="Write a new audit JSON file; refuses overwrite")
    args = parser.parse_args()
    result = audit(args.run_dir)
    rendered = json.dumps(result, indent=2)
    if args.report:
        from pathlib import Path
        with Path(args.report).open('x') as stream:
            stream.write(rendered+'\n')
    print(rendered)
