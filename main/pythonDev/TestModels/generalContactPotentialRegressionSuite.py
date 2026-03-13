#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is an EXUDYN example
#
# Details:  Structured golden-baseline regression runner for potential-contact tests
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import exudyn as exu


TEST_MODELS_DIR = Path(__file__).resolve().parent
PYTHON_DEV_DIR = TEST_MODELS_DIR.parent
REPO_ROOT = TEST_MODELS_DIR.parents[2]
BASELINE_PATH = TEST_MODELS_DIR / "generalContactPotentialGoldenBaseline.json"
REPORT_PATH = REPO_ROOT / "local" / "results" / "generalContactPotentialRegressionReport.json"

LEGACY_SIGNATURE_SCRIPTS = [
    "generalContactPotentialSettings.py",
    "generalContactRigidBodySurfaceMesh.py",
    "generalContactPotentialVF.py",
    "generalContactPotentialVFJacobian.py",
    "generalContactPotentialGCPExactHessian.py",
    "generalContactPotentialEE.py",
    "generalContactPotentialEEJacobian.py",
]


def ensure_pythonpath():
    if str(PYTHON_DEV_DIR) not in sys.path:
        sys.path.insert(0, str(PYTHON_DEV_DIR))


def build_env():
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(PYTHON_DEV_DIR) if not existing_pythonpath else str(PYTHON_DEV_DIR) + os.pathsep + existing_pythonpath
    return env


def run_legacy_signature(script_name, env):
    script_path = TEST_MODELS_DIR / script_name
    completed = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(script_path.parent),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    if completed.returncode != 0:
        raise RuntimeError(f"{script_name} failed with code {completed.returncode}\n{output}")

    matches = re.findall(r"([A-Za-z0-9_]+)=\s*([-+0-9.eE]+)", output)
    if not matches:
        raise RuntimeError(f"{script_name} did not emit a scalar signature\n{output}")

    return float(matches[-1][1])


def to_builtin(value):
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if hasattr(value, "tolist"):
        return to_builtin(value.tolist())
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def flatten_results(value, prefix="", flattened=None):
    if flattened is None:
        flattened = {}

    if isinstance(value, dict):
        for key, item in value.items():
            next_prefix = key if prefix == "" else prefix + "." + key
            flatten_results(item, next_prefix, flattened)
        return flattened

    if isinstance(value, list):
        for index, item in enumerate(value):
            next_prefix = f"{prefix}[{index}]"
            flatten_results(item, next_prefix, flattened)
        return flattened

    flattened[prefix] = value
    return flattened


def infer_tolerance(flat_key, value):
    if isinstance(value, bool):
        return {"kind": "bool", "value": value}
    if isinstance(value, int):
        return {"kind": "int", "value": value}
    if value is None:
        return {"kind": "none", "value": None}
    if isinstance(value, str):
        return {"kind": "str", "value": value}

    abs_tol = 1e-10
    rel_tol = 1e-6
    if "solveTimeMilliseconds" in flat_key or "solveTimeSeconds" in flat_key:
        return {"kind": "float", "value": float(value), "abs_tol": 10.0, "rel_tol": 0.75}
    tolerance_tokens = [
        "force", "energy", "velocity", "minimumDistance", "ratio", "alpha",
        "radius", "coordinate", "normalized", "interference", "signature",
    ]
    if any(token in flat_key for token in tolerance_tokens):
        abs_tol = 5e-8
        rel_tol = 5e-3
    if "legacy_signatures" in flat_key:
        abs_tol = 5e-8
        rel_tol = 5e-4
    return {"kind": "float", "value": float(value), "abs_tol": abs_tol, "rel_tol": rel_tol}


def make_baseline(flattened_results):
    return {
        "formatVersion": 1,
        "entries": {key: infer_tolerance(key, value) for key, value in flattened_results.items()},
    }


def distill_step_controller_results(results):
    return {
        "gcp": {
            "controllerType": str(results["gcp"]["pyData"]["lastPotentialStepControllerType"]),
            "clippedSteps": int(results["gcp"]["pyData"]["totalPotentialCCDClippedSteps"]),
            "minimumDistance": float(results["gcp"]["pyData"]["lastPotentialCCDMinimumDistance"]),
            "coordinateZ": float(results["gcp"]["coordinates"][2]),
        },
        "ogc": {
            "controllerType": str(results["ogc"]["pyData"]["lastPotentialStepControllerType"]),
            "trustRegionRadius": float(results["ogc"]["pyData"]["lastPotentialTrustRegionRadius"]),
            "minimumDistance": float(results["ogc"]["pyData"]["lastPotentialCCDMinimumDistance"]),
            "coordinateZ": float(results["ogc"]["coordinates"][2]),
        },
    }


def distill_ccd_step_filter_results(results):
    return {
        "on": {
            "clippedSteps": int(results["step_filter_on"]["pyData"]["totalPotentialCCDClippedSteps"]),
            "minimumDistance": float(results["step_filter_on"]["pyData"]["lastPotentialCCDMinimumDistance"]),
            "contactMinimumDistance": float(results["step_filter_on"]["pyData"]["lastPotentialContactMinimumDistance"]),
            "coordinateZ": float(results["step_filter_on"]["coordinates"][2]),
        },
        "off": {
            "clippedSteps": int(results["step_filter_off"]["pyData"]["totalPotentialCCDClippedSteps"]),
            "coordinateZ": float(results["step_filter_off"]["coordinates"][2]),
        },
    }


def distill_benchmark_results(results):
    distilled = {}
    for item in results:
        benchmark_key = item["benchmark"]
        formulation_key = item["formulation"]
        distilled.setdefault(benchmark_key, {})[formulation_key] = {
            "controllerType": item["lastPotentialStepControllerType"],
            "contactCandidates": item["lastPotentialContactCandidates"],
            "vfCandidates": item["lastPotentialContactVertexFaceCandidates"],
            "eeCandidates": item["lastPotentialContactEdgeEdgeCandidates"],
            "tangentialCandidates": item["lastPotentialTangentialCandidates"],
            "normalizedMinimumDistance": item["normalizedMinimumDistance"],
            "normalizedProxyInterference": item["normalizedProxyInterference"],
            "velocityRetention": item["velocityRetention"],
            "tangentialCandidateRatio": item["tangentialCandidateRatio"],
            "frictionEnergyPerTangentialCandidate": item["frictionEnergyPerTangentialCandidate"],
            "newtonStepsCount": item["newtonStepsCount"],
        }
    return distilled


def distill_auxiliary_results(results, tag_field):
    distilled = {}
    for item in results:
        tag = item[tag_field]
        distilled[tag] = {
            "benchmark": item["benchmark"],
            "formulation": item["formulation"],
            "experimentSet": item["experimentSet"],
            "experimentTag": item["experimentTag"],
            "normalizedMinimumDistance": item["normalizedMinimumDistance"],
            "normalizedProxyInterference": item["normalizedProxyInterference"],
            "velocityRetention": item["velocityRetention"],
            "tangentialCandidateRatio": item["tangentialCandidateRatio"],
            "contactCandidates": item["lastPotentialContactCandidates"],
            "tangentialCandidates": item["lastPotentialTangentialCandidates"],
            "newtonStepsCount": item["newtonStepsCount"],
            "solveTimeMilliseconds": item["solveTimeMilliseconds"],
            "useNonlinearCCDStepFilter": item["useNonlinearCCDStepFilter"],
            "enablePotentialFriction": item["enablePotentialFriction"],
            "barrierActivationDistance": item["barrierActivationDistance"],
            "barrierStiffness": item["barrierStiffness"],
            "barrierMinimumDistance": item["barrierMinimumDistance"],
            "frictionProportionalZone": item["frictionProportionalZone"],
            "sweepParameter": item.get("sweepParameter"),
            "sweepValue": item.get("sweepValue"),
        }
    return distilled


def collect_results():
    ensure_pythonpath()
    env = build_env()

    from generalContactPotentialModelSplit import run_regression as run_model_split
    from generalContactPotentialCollisionSetSplit import run_regression as run_collision_set_split
    from generalContactPotentialStepControllerSplit import run_regression as run_step_controller_split
    from generalContactPotentialMixedDynamic import run_regression as run_mixed_dynamic
    from generalContactPotentialFriction import run_case as run_friction_case
    from generalContactPotentialTangentialSplit import run_regression as run_tangential_split
    from generalContactPotentialCCDStepFilter import run_regression as run_ccd_step_filter
    from generalContactPotentialBenchmarkMechanisms import (
        run_all_benchmarks,
        run_ablation_benchmarks,
        run_sensitivity_benchmarks,
    )

    legacy_signatures = {}
    for script_name in LEGACY_SIGNATURE_SCRIPTS:
        legacy_signatures[script_name.replace(".py", "")] = run_legacy_signature(script_name, env)

    friction_results = {
        "baseline": run_friction_case(exu.ContactFormulation.GCPBarrier, enable_friction=False),
        "ipc": run_friction_case(exu.ContactFormulation.IPCBarrier, enable_friction=True),
        "gcp": run_friction_case(exu.ContactFormulation.GCPBarrier, enable_friction=True),
        "ogc": run_friction_case(exu.ContactFormulation.OGCBarrier, enable_friction=True),
    }

    results = {
        "legacy_signatures": legacy_signatures,
        "model_split": run_model_split(),
        "collision_set_split": run_collision_set_split(),
        "step_controller_split": distill_step_controller_results(run_step_controller_split()),
        "mixed_dynamic": run_mixed_dynamic(),
        "friction": friction_results,
        "tangential_split": run_tangential_split(),
        "ccd_step_filter": distill_ccd_step_filter_results(run_ccd_step_filter()),
        "mechanism_benchmarks": distill_benchmark_results(run_all_benchmarks()),
        "mechanism_ablations": distill_auxiliary_results(run_ablation_benchmarks(), "experimentTag"),
        "mechanism_sensitivity": distill_auxiliary_results(run_sensitivity_benchmarks(), "experimentTag"),
    }
    return to_builtin(results)


def compare_baseline(flattened_results, baseline):
    entries = baseline["entries"]
    failures = []
    comparisons = []

    for key, specification in entries.items():
        if key not in flattened_results:
            failures.append({"key": key, "reason": "missing_current_value"})
            continue

        current_value = flattened_results[key]
        expected_value = specification["value"]
        kind = specification["kind"]
        passed = True
        details = {"key": key, "kind": kind, "expected": expected_value, "current": current_value}

        if kind == "float":
            abs_tol = float(specification["abs_tol"])
            rel_tol = float(specification["rel_tol"])
            difference = abs(float(current_value) - float(expected_value))
            tolerance = max(abs_tol, rel_tol * max(abs(float(expected_value)), abs(float(current_value)), 1.0))
            passed = difference <= tolerance
            details["difference"] = difference
            details["tolerance"] = tolerance
        else:
            passed = current_value == expected_value

        details["passed"] = passed
        comparisons.append(details)
        if not passed:
            failures.append(details)

    unexpected_keys = sorted(set(flattened_results.keys()) - set(entries.keys()))
    return comparisons, failures, unexpected_keys


def write_report(current_results, flattened_results, baseline, comparisons, failures, unexpected_keys):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "baselinePath": str(BASELINE_PATH),
        "checkedFields": len(comparisons),
        "failedFields": len(failures),
        "unexpectedFields": unexpected_keys,
        "failures": failures,
        "comparisons": comparisons,
        "currentResults": current_results,
        "baseline": baseline,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--update-baseline", action="store_true")
    args = parser.parse_args()

    current_results = collect_results()
    flattened_results = flatten_results(current_results)

    if args.update_baseline:
        baseline = make_baseline(flattened_results)
        BASELINE_PATH.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
        write_report(current_results, flattened_results, baseline, [], [], [])
        exu.Print("generalContactPotentialRegressionCheckedFields=", float(len(flattened_results)))
        exu.Print("generalContactPotentialRegressionSuite=", 1.0)
        return

    if not BASELINE_PATH.exists():
        raise FileNotFoundError(f"missing golden baseline: {BASELINE_PATH}")

    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    comparisons, failures, unexpected_keys = compare_baseline(flattened_results, baseline)
    write_report(current_results, flattened_results, baseline, comparisons, failures, unexpected_keys)

    if failures or unexpected_keys:
        first_issue = failures[0]["key"] if failures else unexpected_keys[0]
        raise RuntimeError(
            f"golden baseline regression failed; checked={len(comparisons)}, failed={len(failures)}, "
            f"unexpected={len(unexpected_keys)}, firstIssue={first_issue}, report={REPORT_PATH}"
        )

    exu.Print("generalContactPotentialRegressionCheckedFields=", float(len(comparisons)))
    exu.Print("generalContactPotentialRegressionSuite=", 1.0)


if __name__ == "__main__":
    main()
