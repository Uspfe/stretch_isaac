import json
import os


def find_items(base_dir, name=None, max_depth=1, include_files=False):
    """
    Search for directories (or files if include_files=True) up to max_depth.
    - name: exact directory or file name to match
    """
    base_depth = base_dir.rstrip(os.sep).count(os.sep)
    results = []

    for root, dirs, files in os.walk(base_dir):
        current_depth = root.rstrip(os.sep).count(os.sep) - base_depth
        if current_depth > max_depth:
            dirs[:] = []  # stop recursion deeper than max_depth
            continue

        # Check directories
        if not include_files:
            for d in dirs:
                if name is None or d == name:
                    results.append(os.path.join(root, d))
        # Check files
        else:
            for f in files:
                if (name is None or f == name):
                    results.append(os.path.join(root, f))

    return results


def main():
    files = ["/home/benni/repos/stretch_isaac/experiment.json"] #, "/home/benni/repos/stretch_isaac/experiment_known.json"]

    experiments = []
    for file_path in files:
        with open(file_path, 'r') as f:
            data = json.load(f)
        experiments += data["experiments"]

    names = [exp["name"] for exp in experiments]

    result_files = ["/home/benni/datasets/sim_results/experiments_results.json"]
    results = []
    for result_file in result_files:
        with open(result_file, 'r') as f:
            data = json.load(f)
        results += data

    for res in results:
        name = res["name"].removesuffix("_perceivesemantix").removesuffix("_dynamem")
        app = res["app"]
        if name in names:
            continue

        state_traj_file = res["state_trajectory_file"]
        run_log_dir = find_items(os.path.dirname(state_traj_file) + "/" + app, name=name, max_depth=2)
        run_log_file = find_items(os.path.dirname(state_traj_file) + "/" + app, name=name + ".pkl", include_files=True, max_depth=2)

        print(f"Deleting outdated experiment result: {res['name']}")
        print(f"  State trajectory file: {state_traj_file}")
        print(f"  Run log dir: {run_log_dir}")
        print(f"  Run log file: {run_log_file}")

if __name__ == "__main__":
    main()

    