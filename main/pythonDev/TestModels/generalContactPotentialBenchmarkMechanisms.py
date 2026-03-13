#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is an EXUDYN example
#
# Details:  Mechanism-focused benchmark suite for IPC / GCP / OGC potential contact
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

import exudyn as exu
from exudyn.itemInterface import *

from generalContactBenchmarkCommon import (
    add_coordinate_constraints,
    add_coordinate_ground,
    add_rigid_body,
    box_mesh,
    extract_contact_force_norm,
    shift_points,
    solve_dynamic_system,
)


FORMULATIONS = [
    ("IPCBarrier", exu.ContactFormulation.IPCBarrier),
    ("GCPBarrier", exu.ContactFormulation.GCPBarrier),
    ("OGCBarrier", exu.ContactFormulation.OGCBarrier),
]

INITIAL_SLIDER_VELOCITY_X = 0.05

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "local" / "results"
SUMMARY_PATH = REPO_ROOT / "local" / "design" / "potential-mechanism-benchmark-summary.md"
RESULTS_JSON_PATH = RESULTS_DIR / "potential-mechanism-benchmarks.json"
RESULTS_CSV_PATH = RESULTS_DIR / "potential-mechanism-benchmarks.csv"
PACKAGE_JSON_PATH = RESULTS_DIR / "potential-mechanism-benchmark-package.json"
ABLATION_JSON_PATH = RESULTS_DIR / "potential-mechanism-ablations.json"
ABLATION_CSV_PATH = RESULTS_DIR / "potential-mechanism-ablations.csv"
SENSITIVITY_JSON_PATH = RESULTS_DIR / "potential-mechanism-sensitivity.json"
SENSITIVITY_CSV_PATH = RESULTS_DIR / "potential-mechanism-sensitivity.csv"

FIGURE_OUTPUTS = {
    "stop_min_distance": RESULTS_DIR / "potential-mechanism-stop-min-distance.png",
    "stop_interference": RESULTS_DIR / "potential-mechanism-stop-interference.png",
    "solver_cost": RESULTS_DIR / "potential-mechanism-solver-cost.png",
    "solve_time": RESULTS_DIR / "potential-mechanism-solve-time-ms.png",
    "friction_response": RESULTS_DIR / "potential-mechanism-friction-response.png",
    "candidate_split": RESULTS_DIR / "potential-mechanism-candidate-split.png",
    "activation_sensitivity": RESULTS_DIR / "potential-mechanism-activation-sensitivity.png",
    "friction_zone_sensitivity": RESULTS_DIR / "potential-mechanism-friction-zone-sensitivity.png",
    "stiffness_sensitivity": RESULTS_DIR / "potential-mechanism-stiffness-sensitivity.png",
}

BENCHMARK_METADATA = {
    "guided_slider_stop": {
        "sceneCategory": "mechanism",
        "contactMode": "normal-stop",
        "description": "Prismatic slider impacting a rigid stop along x.",
        "characteristicLength": 0.05,
        "numberOfSteps": 10,
        "endTimeSeconds": 0.01,
        "primaryResponseMetric": "normalizedProxyInterference",
    },
    "articulated_arm_stop": {
        "sceneCategory": "mechanism",
        "contactMode": "normal-stop",
        "description": "Single-link articulated arm rotating into a planar stop.",
        "characteristicLength": 0.05,
        "numberOfSteps": 20,
        "endTimeSeconds": 0.02,
        "primaryResponseMetric": "normalizedProxyInterference",
    },
    "guided_slider_friction": {
        "sceneCategory": "mechanism",
        "contactMode": "normal-plus-friction",
        "description": "Prismatic slider supported by a plane with tangential sliding friction.",
        "characteristicLength": 0.05,
        "numberOfSteps": 2,
        "endTimeSeconds": 0.002,
        "primaryResponseMetric": "velocityRetention",
        "initialVelocityX": INITIAL_SLIDER_VELOCITY_X,
    },
}

METRIC_DEFINITIONS = {
    "lastPotentialContactMinimumDistance": "Final-step minimum geometric clearance reported by the potential-contact pipeline.",
    "normalizedMinimumDistance": "Minimum geometric clearance divided by the benchmark characteristic length.",
    "proxyInterference": "Scene-specific kinematic overlap proxy derived from the final rigid-body reference position; this is not identical to the geometric minimum distance.",
    "normalizedProxyInterference": "Kinematic overlap proxy divided by the benchmark characteristic length.",
    "newtonStepsPerTimeStep": "Total Newton iterations divided by the number of time steps in the benchmark.",
    "solveTimeMilliseconds": "Wall-clock solve time in milliseconds for the benchmark run.",
    "tangentialCandidateRatio": "Tangential candidate count divided by total normal candidates at the final step.",
    "normalEnergyPerCandidate": "Accumulated normal potential energy divided by the final normal candidate count.",
    "frictionEnergyPerTangentialCandidate": "Accumulated tangential/friction energy divided by the final tangential candidate count.",
    "velocityRetention": "Final slider x-velocity divided by the prescribed initial slider x-velocity.",
    "relativeSolveTimeVsIPC": "Per-benchmark solve-time ratio relative to the IPCBarrier run.",
    "relativeNewtonStepsVsIPC": "Per-benchmark Newton-step ratio relative to the IPCBarrier run.",
    "relativeVelocityRetentionVsIPC": "Per-benchmark velocity-retention ratio relative to the IPCBarrier run when defined.",
}


def safe_ratio(numerator, denominator, default_value=0.0):
    numerator = float(numerator)
    denominator = float(denominator)
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) < 1e-16:
        return float(default_value)
    return float(numerator / denominator)


def format_optional_scientific(value):
    if value is None:
        return ""
    return f"{float(value):.6e}"


def default_friction_barrier_stiffness(formulation):
    if formulation == exu.ContactFormulation.GCPBarrier:
        return 300.0
    return 250.0


def configure_potential_contact(mbs, formulation, enable_friction=False, friction_coefficient=0.6, contact_overrides=None):
    overrides = {} if contact_overrides is None else dict(contact_overrides)
    configuration = {
        "barrierActivationDistance": float(overrides.get("barrierActivationDistance", 0.08)),
        "barrierStiffness": float(overrides.get(
            "barrierStiffness",
            default_friction_barrier_stiffness(formulation) if enable_friction else 1200.0,
        )),
        "barrierMinimumDistance": float(overrides.get("barrierMinimumDistance", 1e-6)),
        "enablePotentialFriction": bool(overrides.get("enablePotentialFriction", enable_friction)),
        "frictionProportionalZone": float(overrides.get("frictionProportionalZone", 1.0)),
        "useGaussNewtonHessian": bool(overrides.get("useGaussNewtonHessian", True)),
        "useNonlinearCCDStepFilter": bool(overrides.get("useNonlinearCCDStepFilter", True)),
        "ccdTolerance": float(overrides.get("ccdTolerance", 1e-6)),
        "frictionCoefficient": float(overrides.get("frictionCoefficient", friction_coefficient if enable_friction else 0.0)),
    }
    g_contact = mbs.AddGeneralContact()
    g_contact.contactFormulation = formulation
    g_contact.computeContactForces = True
    g_contact.barrierActivationDistance = configuration["barrierActivationDistance"]
    g_contact.barrierStiffness = configuration["barrierStiffness"]
    g_contact.barrierMinimumDistance = configuration["barrierMinimumDistance"]
    g_contact.enablePotentialFriction = configuration["enablePotentialFriction"]
    g_contact.frictionProportionalZone = configuration["frictionProportionalZone"]
    g_contact.useGaussNewtonHessian = configuration["useGaussNewtonHessian"]
    g_contact.useNonlinearCCDStepFilter = configuration["useNonlinearCCDStepFilter"]
    g_contact.ccdTolerance = configuration["ccdTolerance"]
    g_contact.SetSearchTreeBox([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
    g_contact.SetFrictionPairings([[configuration["frictionCoefficient"]]])
    return g_contact, configuration


def add_barrier_plane(g_contact, ground_marker, axis):
    if axis == "z":
        plane_points, plane_triangles = box_mesh(1.2, 1.2, 0.05)
        plane_points = shift_points(plane_points, [0.0, 0.0, -0.025])
    elif axis == "x":
        plane_points, plane_triangles = box_mesh(0.05, 1.2, 1.2)
        plane_points = shift_points(plane_points, [-0.025, 0.0, 0.0])
    else:
        raise ValueError("unsupported barrier plane axis")
    g_contact.AddRigidBodySurfaceMesh(
        ground_marker,
        plane_points,
        plane_triangles,
        frictionMaterialIndex=0,
        staticMesh=True,
    )


def add_barrier_cube(g_contact, body_marker, characteristic_size):
    cube_points, cube_triangles = box_mesh(2.0 * characteristic_size, 2.0 * characteristic_size, 2.0 * characteristic_size)
    g_contact.AddRigidBodySurfaceMesh(
        body_marker,
        cube_points,
        cube_triangles,
        frictionMaterialIndex=0,
    )


def base_result(benchmark_name, formulation_name, solver, elapsed, g_contact, contact_configuration,
                experiment_set="main", experiment_tag="default"):
    metadata = BENCHMARK_METADATA[benchmark_name]
    py_data = g_contact.GetPythonObject()
    number_of_steps = int(metadata["numberOfSteps"])
    end_time_seconds = float(metadata["endTimeSeconds"])
    characteristic_length = float(metadata["characteristicLength"])
    result = {
        "benchmark": benchmark_name,
        "sceneCategory": metadata["sceneCategory"],
        "contactMode": metadata["contactMode"],
        "sceneDescription": metadata["description"],
        "primaryResponseMetric": metadata["primaryResponseMetric"],
        "formulation": formulation_name,
        "experimentSet": experiment_set,
        "experimentTag": experiment_tag,
        "solverName": solver.GetSolverName(),
        "numberOfSteps": number_of_steps,
        "endTimeSeconds": end_time_seconds,
        "timeStepSizeSeconds": safe_ratio(end_time_seconds, number_of_steps),
        "characteristicLength": characteristic_length,
        "solveTimeSeconds": float(elapsed),
        "solveTimeMilliseconds": 1e3 * float(elapsed),
        "newtonStepsCount": int(solver.it.newtonStepsCount),
        "newtonStepsPerTimeStep": safe_ratio(int(solver.it.newtonStepsCount), number_of_steps),
        "contactForceNorm": float(extract_contact_force_norm(g_contact)),
        "lastPotentialContactCoarseBroadPhasePairs": int(py_data["lastPotentialContactCoarseBroadPhasePairs"]),
        "lastPotentialContactBroadPhasePairs": int(py_data["lastPotentialContactBroadPhasePairs"]),
        "lastPotentialBroadPhaseRejectedPairs": int(py_data["lastPotentialBroadPhaseRejectedPairs"]),
        "lastPotentialMeshPairBuilderType": str(py_data["lastPotentialMeshPairBuilderType"]),
        "lastPotentialContactVertexFaceSeeds": int(py_data["lastPotentialContactVertexFaceSeeds"]),
        "lastPotentialContactEdgeEdgeSeeds": int(py_data["lastPotentialContactEdgeEdgeSeeds"]),
        "lastPotentialContactSeeds": int(py_data["lastPotentialContactSeeds"]),
        "lastPotentialSeedRejectedCandidates": int(py_data["lastPotentialSeedRejectedCandidates"]),
        "lastPotentialSeedBuilderType": str(py_data["lastPotentialSeedBuilderType"]),
        "lastPotentialContactVertexFaceCandidates": int(py_data["lastPotentialContactVertexFaceCandidates"]),
        "lastPotentialContactEdgeEdgeCandidates": int(py_data["lastPotentialContactEdgeEdgeCandidates"]),
        "lastPotentialTangentialCandidates": int(py_data["lastPotentialTangentialCandidates"]),
        "lastPotentialContactCandidates": int(py_data["lastPotentialContactCandidates"]),
        "lastPotentialCollisionSetRejectedCandidates": int(py_data["lastPotentialCollisionSetRejectedCandidates"]),
        "lastPotentialCollisionSetBuilderType": str(py_data["lastPotentialCollisionSetBuilderType"]),
        "lastPotentialContactMinimumDistance": float(py_data["lastPotentialContactMinimumDistance"]),
        "lastPotentialAccumulatedNormalEnergy": float(py_data["lastPotentialAccumulatedNormalEnergy"]),
        "lastPotentialAccumulatedFrictionEnergy": float(py_data["lastPotentialAccumulatedFrictionEnergy"]),
        "lastPotentialStepControllerType": str(py_data["lastPotentialStepControllerType"]),
        "lastPotentialCCDAlpha": float(py_data["lastPotentialCCDAlpha"]),
        "lastPotentialCCDMinimumDistance": float(py_data["lastPotentialCCDMinimumDistance"]),
        "lastPotentialCCDNumberOfEvaluations": int(py_data["lastPotentialCCDNumberOfEvaluations"]),
        "lastPotentialCCDStepReductions": int(py_data["lastPotentialCCDStepReductions"]),
        "lastPotentialTrustRegionRejects": int(py_data["lastPotentialTrustRegionRejects"]),
        "lastPotentialTrustRegionRadius": float(py_data["lastPotentialTrustRegionRadius"]),
        "totalPotentialCCDClippedSteps": int(py_data["totalPotentialCCDClippedSteps"]),
        "totalPotentialCCDStepFailures": int(py_data["totalPotentialCCDStepFailures"]),
        "barrierActivationDistance": contact_configuration["barrierActivationDistance"],
        "barrierStiffness": contact_configuration["barrierStiffness"],
        "barrierMinimumDistance": contact_configuration["barrierMinimumDistance"],
        "enablePotentialFriction": contact_configuration["enablePotentialFriction"],
        "frictionProportionalZone": contact_configuration["frictionProportionalZone"],
        "frictionCoefficient": contact_configuration["frictionCoefficient"],
        "useNonlinearCCDStepFilter": contact_configuration["useNonlinearCCDStepFilter"],
    }
    if "initialVelocityX" in metadata:
        result["initialVelocityX"] = float(metadata["initialVelocityX"])
    return result


def enrich_result(result):
    result["normalizedMinimumDistance"] = safe_ratio(
        result["lastPotentialContactMinimumDistance"],
        result["characteristicLength"],
    )
    result["tangentialCandidateRatio"] = safe_ratio(
        result["lastPotentialTangentialCandidates"],
        result["lastPotentialContactCandidates"],
    )
    result["normalEnergyPerCandidate"] = safe_ratio(
        result["lastPotentialAccumulatedNormalEnergy"],
        result["lastPotentialContactCandidates"],
    )
    result["frictionEnergyPerTangentialCandidate"] = safe_ratio(
        result["lastPotentialAccumulatedFrictionEnergy"],
        result["lastPotentialTangentialCandidates"],
    )
    result["normalizedProxyInterference"] = safe_ratio(
        result.get("proxyInterference", 0.0),
        result["characteristicLength"],
    )
    if "initialVelocityX" in result and "velocityX" in result:
        result["velocityRetention"] = safe_ratio(result["velocityX"], result["initialVelocityX"])
        result["velocityLoss"] = float(result["initialVelocityX"] - result["velocityX"])
        result["velocityLossRatio"] = safe_ratio(result["velocityLoss"], result["initialVelocityX"])
    else:
        result["velocityRetention"] = None
        result["velocityLoss"] = None
        result["velocityLossRatio"] = None


def add_relative_metrics(results):
    grouped = {}
    for item in results:
        grouped.setdefault(item["benchmark"], {})[item["formulation"]] = item

    for benchmark_name, cases in grouped.items():
        ipc = cases["IPCBarrier"]
        for formulation_name, item in cases.items():
            item["relativeSolveTimeVsIPC"] = safe_ratio(item["solveTimeSeconds"], ipc["solveTimeSeconds"], default_value=1.0)
            item["relativeNewtonStepsVsIPC"] = safe_ratio(item["newtonStepsCount"], ipc["newtonStepsCount"], default_value=1.0)
            item["relativeMinimumDistanceVsIPC"] = safe_ratio(
                item["lastPotentialContactMinimumDistance"],
                ipc["lastPotentialContactMinimumDistance"],
                default_value=1.0,
            )
            item["relativeProxyInterferenceVsIPC"] = safe_ratio(
                item.get("proxyInterference", 0.0),
                ipc.get("proxyInterference", 0.0),
                default_value=1.0 if formulation_name == "IPCBarrier" else 0.0,
            )
            if item.get("velocityRetention") is not None:
                item["relativeVelocityRetentionVsIPC"] = safe_ratio(
                    item["velocityRetention"],
                    ipc.get("velocityRetention", 0.0),
                    default_value=1.0 if formulation_name == "IPCBarrier" else 0.0,
                )
            else:
                item["relativeVelocityRetentionVsIPC"] = None


def run_guided_slider_stop(formulation_name, formulation, experiment_set="main", experiment_tag="default",
                           contact_overrides=None):
    benchmark_name = "guided_slider_stop"
    characteristic_size = BENCHMARK_METADATA[benchmark_name]["characteristicLength"]
    SC = exu.SystemContainer()
    mbs = SC.AddSystem()

    g_contact, contact_configuration = configure_potential_contact(
        mbs, formulation, enable_friction=False, contact_overrides=contact_overrides
    )
    o_ground = mbs.AddObject(ObjectGround())
    m_ground_body = mbs.AddMarker(MarkerBodyRigid(bodyNumber=o_ground, localPosition=[0.0, 0.0, 0.0]))
    _, m_coordinate_ground = add_coordinate_ground(mbs)

    n_slider, _, m_slider = add_rigid_body(mbs, [0.09, 0.0, 0.0, 0.0, 0.0, 0.0], mass=1.5, inertia_value=0.02)
    add_coordinate_constraints(mbs, n_slider, m_coordinate_ground, [1, 2, 3, 4, 5])

    m_slider_x = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=n_slider, coordinate=0))
    mbs.AddObject(CoordinateSpringDamper(
        markerNumbers=[m_coordinate_ground, m_slider_x],
        stiffness=500.0,
        damping=5.0,
        offset=0.09,
        visualization=VCoordinateSpringDamper(show=False),
    ))
    mbs.AddLoad(Force(markerNumber=m_slider, loadVector=[-2500.0, 0.0, 0.0]))

    add_barrier_plane(g_contact, m_ground_body, "x")
    add_barrier_cube(g_contact, m_slider, characteristic_size)

    mbs.Assemble()
    solver, elapsed = solve_dynamic_system(
        mbs,
        number_of_steps=BENCHMARK_METADATA[benchmark_name]["numberOfSteps"],
        end_time=BENCHMARK_METADATA[benchmark_name]["endTimeSeconds"],
    )

    position = np.array(mbs.GetNodeOutput(n_slider, exu.OutputVariableType.Position), dtype=float)
    result = base_result(
        benchmark_name, formulation_name, solver, elapsed, g_contact, contact_configuration,
        experiment_set=experiment_set, experiment_tag=experiment_tag
    )
    result["bodyPositionX"] = float(position[0])
    result["proxyInterference"] = max(0.0, characteristic_size - float(position[0]))
    result["maxPenetration"] = result["proxyInterference"]
    enrich_result(result)
    return result


def run_articulated_arm_stop(formulation_name, formulation, experiment_set="main", experiment_tag="default",
                             contact_overrides=None):
    benchmark_name = "articulated_arm_stop"
    characteristic_size = BENCHMARK_METADATA[benchmark_name]["characteristicLength"]
    arm_length = 0.1
    SC = exu.SystemContainer()
    mbs = SC.AddSystem()

    g_contact, contact_configuration = configure_potential_contact(
        mbs, formulation, enable_friction=False, contact_overrides=contact_overrides
    )
    o_ground = mbs.AddObject(ObjectGround())
    m_ground_body = mbs.AddMarker(MarkerBodyRigid(bodyNumber=o_ground, localPosition=[0.0, 0.0, 0.0]))
    _, m_coordinate_ground = add_coordinate_ground(mbs)

    n_rigid, body_number, m_rigid_com = add_rigid_body(mbs, [0.0, 0.0, 0.12, 0.0, 0.0, 0.0], mass=1.0, inertia_value=0.02)
    m_tip = mbs.AddMarker(MarkerBodyRigid(bodyNumber=body_number, localPosition=[arm_length, 0.0, 0.0]))
    add_coordinate_constraints(mbs, n_rigid, m_coordinate_ground, [0, 1, 2, 3, 5])

    m_phi_y = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=n_rigid, coordinate=4))
    mbs.AddObject(CoordinateSpringDamper(
        markerNumbers=[m_coordinate_ground, m_phi_y],
        stiffness=20.0,
        damping=1.0,
        offset=0.0,
        visualization=VCoordinateSpringDamper(show=False),
    ))
    mbs.AddLoad(Torque(markerNumber=m_rigid_com, loadVector=[0.0, 300.0, 0.0]))

    add_barrier_plane(g_contact, m_ground_body, "z")
    add_barrier_cube(g_contact, m_tip, characteristic_size)

    mbs.Assemble()
    solver, elapsed = solve_dynamic_system(
        mbs,
        number_of_steps=BENCHMARK_METADATA[benchmark_name]["numberOfSteps"],
        end_time=BENCHMARK_METADATA[benchmark_name]["endTimeSeconds"],
    )

    tip_position = np.array(mbs.GetMarkerOutput(m_tip, exu.OutputVariableType.Position), dtype=float)
    result = base_result(
        benchmark_name, formulation_name, solver, elapsed, g_contact, contact_configuration,
        experiment_set=experiment_set, experiment_tag=experiment_tag
    )
    result["tipPositionZ"] = float(tip_position[2])
    result["proxyInterference"] = max(0.0, characteristic_size - float(tip_position[2]))
    result["maxPenetration"] = result["proxyInterference"]
    enrich_result(result)
    return result


def run_guided_slider_friction(formulation_name, formulation, experiment_set="main", experiment_tag="default",
                               contact_overrides=None):
    benchmark_name = "guided_slider_friction"
    characteristic_size = BENCHMARK_METADATA[benchmark_name]["characteristicLength"]
    SC = exu.SystemContainer()
    mbs = SC.AddSystem()

    g_contact, contact_configuration = configure_potential_contact(
        mbs, formulation, enable_friction=True, contact_overrides=contact_overrides
    )
    o_ground = mbs.AddObject(ObjectGround())
    m_ground_body = mbs.AddMarker(MarkerBodyRigid(bodyNumber=o_ground, localPosition=[0.0, 0.0, 0.0]))
    _, m_coordinate_ground = add_coordinate_ground(mbs)

    n_slider = mbs.AddNode(NodeRigidBodyRxyz(
        referenceCoordinates=[0.0, 0.0, 0.085, 0.0, 0.0, 0.0],
        initialVelocities=[INITIAL_SLIDER_VELOCITY_X, 0.0, 0.0, 0.0, 0.0, 0.0],
    ))
    o_slider = mbs.AddObject(ObjectRigidBody(
        physicsMass=1.0,
        physicsInertia=[0.02, 0.02, 0.02, 0.0, 0.0, 0.0],
        nodeNumber=n_slider,
        visualization=VObjectRigidBody(graphicsData=[]),
    ))
    m_slider = mbs.AddMarker(MarkerBodyRigid(bodyNumber=o_slider, localPosition=[0.0, 0.0, 0.0]))
    add_coordinate_constraints(mbs, n_slider, m_coordinate_ground, [1, 3, 4, 5])

    m_slider_z = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=n_slider, coordinate=2))
    mbs.AddObject(CoordinateSpringDamper(
        markerNumbers=[m_coordinate_ground, m_slider_z],
        stiffness=1200.0,
        damping=40.0,
        offset=0.085,
        visualization=VCoordinateSpringDamper(show=False),
    ))
    mbs.AddLoad(Force(markerNumber=m_slider, loadVector=[0.0, 0.0, -80.0]))

    add_barrier_plane(g_contact, m_ground_body, "z")
    add_barrier_cube(g_contact, m_slider, characteristic_size)

    mbs.Assemble()
    solver, elapsed = solve_dynamic_system(
        mbs,
        number_of_steps=BENCHMARK_METADATA[benchmark_name]["numberOfSteps"],
        end_time=BENCHMARK_METADATA[benchmark_name]["endTimeSeconds"],
    )

    velocity = np.array(mbs.GetNodeOutput(n_slider, exu.OutputVariableType.Velocity), dtype=float)
    position = np.array(mbs.GetNodeOutput(n_slider, exu.OutputVariableType.Position), dtype=float)
    result = base_result(
        benchmark_name, formulation_name, solver, elapsed, g_contact, contact_configuration,
        experiment_set=experiment_set, experiment_tag=experiment_tag
    )
    result["velocityX"] = float(velocity[0])
    result["bodyPositionZ"] = float(position[2])
    result["proxyInterference"] = max(0.0, characteristic_size - float(position[2]))
    result["maxPenetration"] = result["proxyInterference"]
    enrich_result(result)
    return result


BENCHMARKS = [
    run_guided_slider_stop,
    run_articulated_arm_stop,
    run_guided_slider_friction,
]


def run_all_benchmarks():
    results = []
    for formulation_name, formulation in FORMULATIONS:
        for benchmark in BENCHMARKS:
            results.append(benchmark(formulation_name, formulation))
    add_relative_metrics(results)
    return results


def run_ablation_benchmarks():
    results = []

    for use_step_filter in [False, True]:
        item = run_guided_slider_stop(
            "GCPBarrier", exu.ContactFormulation.GCPBarrier,
            experiment_set="ablation",
            experiment_tag=f"gcp_step_filter_{'on' if use_step_filter else 'off'}",
            contact_overrides={"useNonlinearCCDStepFilter": use_step_filter},
        )
        item["ablationName"] = "step_filter"
        item["ablationLevel"] = "on" if use_step_filter else "off"
        results.append(item)

    for formulation_name, formulation in [("GCPBarrier", exu.ContactFormulation.GCPBarrier),
                                          ("OGCBarrier", exu.ContactFormulation.OGCBarrier)]:
        for enable_friction in [False, True]:
            item = run_guided_slider_friction(
                formulation_name, formulation,
                experiment_set="ablation",
                experiment_tag=f"{formulation_name.lower()}_friction_{'on' if enable_friction else 'off'}",
                contact_overrides={"enablePotentialFriction": enable_friction,
                                   "frictionCoefficient": 0.6 if enable_friction else 0.0},
            )
            item["ablationName"] = "potential_friction"
            item["ablationLevel"] = "on" if enable_friction else "off"
            results.append(item)

    return results


def run_sensitivity_benchmarks():
    results = []
    for activation_distance in [0.04, 0.07, 0.08]:
        for formulation_name, formulation in [("GCPBarrier", exu.ContactFormulation.GCPBarrier),
                                              ("OGCBarrier", exu.ContactFormulation.OGCBarrier)]:
            item = run_guided_slider_stop(
                formulation_name, formulation,
                experiment_set="sensitivity",
                experiment_tag=f"{formulation_name.lower()}_activation_{activation_distance:.2f}",
                contact_overrides={"barrierActivationDistance": activation_distance},
            )
            item["sweepParameter"] = "barrierActivationDistance"
            item["sweepValue"] = activation_distance
            results.append(item)

    for regularization_velocity in [0.25, 0.5, 1.0, 1.5]:
        for formulation_name, formulation in [("GCPBarrier", exu.ContactFormulation.GCPBarrier),
                                              ("OGCBarrier", exu.ContactFormulation.OGCBarrier)]:
            item = run_guided_slider_friction(
                formulation_name, formulation,
                experiment_set="sensitivity",
                experiment_tag=f"{formulation_name.lower()}_friction_zone_{regularization_velocity:.2f}",
                contact_overrides={"frictionProportionalZone": regularization_velocity},
            )
            item["sweepParameter"] = "frictionProportionalZone"
            item["sweepValue"] = regularization_velocity
            results.append(item)

    for barrier_stiffness in [200.0, 250.0, 300.0]:
        item = run_guided_slider_friction(
            "GCPBarrier", exu.ContactFormulation.GCPBarrier,
            experiment_set="sensitivity",
            experiment_tag=f"gcpbarrier_friction_stiffness_{barrier_stiffness:.0f}",
            contact_overrides={"barrierStiffness": barrier_stiffness},
        )
        item["sweepParameter"] = "barrierStiffness"
        item["sweepValue"] = barrier_stiffness
        results.append(item)

    return results


def csv_header():
    return [
        "benchmark",
        "sceneCategory",
        "contactMode",
        "formulation",
        "experimentSet",
        "experimentTag",
        "primaryResponseMetric",
        "lastPotentialStepControllerType",
        "lastPotentialMeshPairBuilderType",
        "lastPotentialSeedBuilderType",
        "lastPotentialCollisionSetBuilderType",
        "numberOfSteps",
        "timeStepSizeSeconds",
        "solveTimeSeconds",
        "solveTimeMilliseconds",
        "newtonStepsCount",
        "newtonStepsPerTimeStep",
        "contactForceNorm",
        "barrierActivationDistance",
        "barrierStiffness",
        "barrierMinimumDistance",
        "enablePotentialFriction",
        "frictionProportionalZone",
        "frictionCoefficient",
        "useNonlinearCCDStepFilter",
        "lastPotentialContactMinimumDistance",
        "normalizedMinimumDistance",
        "proxyInterference",
        "normalizedProxyInterference",
        "velocityX",
        "initialVelocityX",
        "velocityRetention",
        "velocityLossRatio",
        "lastPotentialContactCandidates",
        "lastPotentialContactVertexFaceCandidates",
        "lastPotentialContactEdgeEdgeCandidates",
        "lastPotentialTangentialCandidates",
        "tangentialCandidateRatio",
        "lastPotentialContactSeeds",
        "lastPotentialContactVertexFaceSeeds",
        "lastPotentialContactEdgeEdgeSeeds",
        "lastPotentialSeedRejectedCandidates",
        "lastPotentialCollisionSetRejectedCandidates",
        "lastPotentialAccumulatedNormalEnergy",
        "normalEnergyPerCandidate",
        "lastPotentialAccumulatedFrictionEnergy",
        "frictionEnergyPerTangentialCandidate",
        "lastPotentialCCDAlpha",
        "lastPotentialCCDMinimumDistance",
        "lastPotentialCCDNumberOfEvaluations",
        "lastPotentialCCDStepReductions",
        "lastPotentialTrustRegionRejects",
        "lastPotentialTrustRegionRadius",
        "totalPotentialCCDClippedSteps",
        "totalPotentialCCDStepFailures",
        "relativeSolveTimeVsIPC",
        "relativeNewtonStepsVsIPC",
        "relativeMinimumDistanceVsIPC",
        "relativeProxyInterferenceVsIPC",
        "relativeVelocityRetentionVsIPC",
        "ablationName",
        "ablationLevel",
        "sweepParameter",
        "sweepValue",
    ]


def build_package(results, ablation_results=None, sensitivity_results=None):
    benchmark_scenes = []
    for benchmark_name, metadata in BENCHMARK_METADATA.items():
        benchmark_scenes.append(
            {
                "benchmark": benchmark_name,
                "sceneCategory": metadata["sceneCategory"],
                "contactMode": metadata["contactMode"],
                "description": metadata["description"],
                "characteristicLength": metadata["characteristicLength"],
                "numberOfSteps": metadata["numberOfSteps"],
                "endTimeSeconds": metadata["endTimeSeconds"],
                "primaryResponseMetric": metadata["primaryResponseMetric"],
            }
        )

    return {
        "packageName": "potential-mechanism-benchmark-v2",
        "generatedAt": datetime.now().isoformat(timespec="seconds"),
        "resultsJson": str(RESULTS_JSON_PATH),
        "resultsCsv": str(RESULTS_CSV_PATH),
        "ablationResultsJson": str(ABLATION_JSON_PATH),
        "ablationResultsCsv": str(ABLATION_CSV_PATH),
        "sensitivityResultsJson": str(SENSITIVITY_JSON_PATH),
        "sensitivityResultsCsv": str(SENSITIVITY_CSV_PATH),
        "summaryMarkdown": str(SUMMARY_PATH),
        "figures": {name: str(path) for name, path in FIGURE_OUTPUTS.items()},
        "metricDefinitions": METRIC_DEFINITIONS,
        "benchmarkScenes": benchmark_scenes,
        "results": results,
        "ablationResults": [] if ablation_results is None else ablation_results,
        "sensitivityResults": [] if sensitivity_results is None else sensitivity_results,
    }


def write_result_set(results, json_path, csv_path):
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=csv_header())
        writer.writeheader()
        for item in results:
            writer.writerow({key: item.get(key, "") for key in csv_header()})


def write_results(results, ablation_results=None, sensitivity_results=None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_result_set(results, RESULTS_JSON_PATH, RESULTS_CSV_PATH)
    if ablation_results is not None:
        write_result_set(ablation_results, ABLATION_JSON_PATH, ABLATION_CSV_PATH)
    if sensitivity_results is not None:
        write_result_set(sensitivity_results, SENSITIVITY_JSON_PATH, SENSITIVITY_CSV_PATH)
    with PACKAGE_JSON_PATH.open("w", encoding="utf-8") as file:
        json.dump(build_package(results, ablation_results, sensitivity_results), file, indent=2)


def build_summary(results, ablation_results=None, sensitivity_results=None):
    grouped = {}
    for item in results:
        grouped.setdefault(item["benchmark"], {})[item["formulation"]] = item

    lines = [
        "# Potential Mechanism Benchmark Summary",
        "",
        "## Protocol",
        "",
        "- Scope: mechanism-style rigid-contact scenes only; no free-fall toy cases are included in this package.",
        "- Formulations: `IPCBarrier`, `GCPBarrier`, and `OGCBarrier` are compared under the same scene geometry and time-step settings.",
        "- Controller split: IPC/GCP remain on `CCDLineSearch`; OGC remains on `TrustRegion`.",
        "- Interpretation note: `proxyInterference` is a kinematic response proxy derived from the final reference position. It is not the same quantity as `lastPotentialContactMinimumDistance`.",
        "",
        "## Benchmark Scenes",
        "",
        "| Benchmark | Contact mode | Steps | dt [s] | Characteristic length | Primary response | Description |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]

    for benchmark_name, metadata in BENCHMARK_METADATA.items():
        lines.append(
            f"| {benchmark_name} | {metadata['contactMode']} | {metadata['numberOfSteps']} | "
            f"{safe_ratio(metadata['endTimeSeconds'], metadata['numberOfSteps']):.3e} | "
            f"{metadata['characteristicLength']:.3e} | {metadata['primaryResponseMetric']} | {metadata['description']} |"
        )

    lines += [
        "",
        "## Metric Definitions",
        "",
    ]
    for metric_name, description in METRIC_DEFINITIONS.items():
        lines.append(f"- `{metric_name}`: {description}")

    lines += [
        "",
        "## Stop-Contact Results",
        "",
        "| Benchmark | Formulation | Controller | Min distance / L | Proxy interference / L | Newton / step | Solve time [ms] | Candidates (VF/EE) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for benchmark_name in ["guided_slider_stop", "articulated_arm_stop"]:
        for formulation_name in ["IPCBarrier", "GCPBarrier", "OGCBarrier"]:
            item = grouped[benchmark_name][formulation_name]
            lines.append(
                f"| {benchmark_name} | {formulation_name} | {item['lastPotentialStepControllerType']} | "
                f"{item['normalizedMinimumDistance']:.6e} | {item['normalizedProxyInterference']:.6e} | "
                f"{item['newtonStepsPerTimeStep']:.3f} | {item['solveTimeMilliseconds']:.3f} | "
                f"{item['lastPotentialContactCandidates']} ({item['lastPotentialContactVertexFaceCandidates']}/{item['lastPotentialContactEdgeEdgeCandidates']}) |"
            )

    lines += [
        "",
        "## Friction Results",
        "",
        "| Benchmark | Formulation | Controller | Min distance / L | Velocity retention | Tangential ratio | Friction energy / tangential cand. | Solve time [ms] |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for formulation_name in ["IPCBarrier", "GCPBarrier", "OGCBarrier"]:
        item = grouped["guided_slider_friction"][formulation_name]
        lines.append(
            f"| guided_slider_friction | {formulation_name} | {item['lastPotentialStepControllerType']} | "
            f"{item['normalizedMinimumDistance']:.6e} | {item['velocityRetention']:.6e} | "
            f"{item['tangentialCandidateRatio']:.6e} | {item['frictionEnergyPerTangentialCandidate']:.6e} | "
            f"{item['solveTimeMilliseconds']:.3f} |"
        )

    slider_friction_cases = grouped["guided_slider_friction"]
    ablation_results = [] if ablation_results is None else ablation_results
    sensitivity_results = [] if sensitivity_results is None else sensitivity_results
    lines += [
        "",
        "## Key Observations",
        "",
        "- The tangential-contact path is no longer a direct clone of the normal candidate set. In the friction benchmark, all three formulations keep fewer tangential candidates than normal candidates.",
        f"- OGC remains the strictest frictional selector in the current mechanism suite: tangential candidates = "
        f"{slider_friction_cases['OGCBarrier']['lastPotentialTangentialCandidates']} versus "
        f"{slider_friction_cases['IPCBarrier']['lastPotentialTangentialCandidates']} for IPC and "
        f"{slider_friction_cases['GCPBarrier']['lastPotentialTangentialCandidates']} for GCP.",
        "- Stop-contact scenes should be compared primarily via `normalizedMinimumDistance` and solver effort. `normalizedProxyInterference` is retained as a scene-level response proxy, not as a replacement for the geometric clearance metric.",
    ]

    if ablation_results:
        lines += [
            "",
            "## Ablation Results",
            "",
            "| Benchmark | Formulation | Ablation | Level | Controller | Min distance / L | Velocity retention | Tangential ratio |",
            "| --- | --- | --- | --- | --- | ---: | ---: | ---: |",
        ]
        for item in ablation_results:
            lines.append(
                f"| {item['benchmark']} | {item['formulation']} | {item.get('ablationName', '')} | "
                f"{item.get('ablationLevel', '')} | {item['lastPotentialStepControllerType']} | "
                f"{item['normalizedMinimumDistance']:.6e} | "
                f"{format_optional_scientific(item.get('velocityRetention'))} | "
                f"{item['tangentialCandidateRatio']:.6e} |"
            )

    if sensitivity_results:
        lines += [
            "",
            "## Sensitivity Results",
            "",
            "| Benchmark | Formulation | Sweep parameter | Sweep value | Min distance / L | Proxy interference / L | Velocity retention | Tangential ratio |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for item in sensitivity_results:
            lines.append(
                f"| {item['benchmark']} | {item['formulation']} | {item.get('sweepParameter', '')} | "
                f"{item.get('sweepValue', 0.):.3e} | {item['normalizedMinimumDistance']:.6e} | "
                f"{item['normalizedProxyInterference']:.6e} | "
                f"{format_optional_scientific(item.get('velocityRetention'))} | "
                f"{item['tangentialCandidateRatio']:.6e} |"
            )

    lines += [
        "",
        "## Output Files",
        "",
        f"- Raw JSON results: `{RESULTS_JSON_PATH}`",
        f"- CSV export: `{RESULTS_CSV_PATH}`",
        f"- Ablation JSON results: `{ABLATION_JSON_PATH}`",
        f"- Ablation CSV export: `{ABLATION_CSV_PATH}`",
        f"- Sensitivity JSON results: `{SENSITIVITY_JSON_PATH}`",
        f"- Sensitivity CSV export: `{SENSITIVITY_CSV_PATH}`",
        f"- Package manifest: `{PACKAGE_JSON_PATH}`",
        f"- Summary: `{SUMMARY_PATH}`",
    ]

    for figure_name, figure_path in FIGURE_OUTPUTS.items():
        lines.append(f"- Figure `{figure_name}`: `{figure_path}`")

    return "\n".join(lines) + "\n"


def write_summary(results, ablation_results=None, sensitivity_results=None):
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(build_summary(results, ablation_results, sensitivity_results), encoding="utf-8")


def main():
    results = run_all_benchmarks()
    ablation_results = run_ablation_benchmarks()
    sensitivity_results = run_sensitivity_benchmarks()
    write_results(results, ablation_results, sensitivity_results)
    write_summary(results, ablation_results, sensitivity_results)

    signature = 0.0
    for item in results:
        signature += item["normalizedMinimumDistance"]
        signature += 0.01 * item["newtonStepsCount"]
        signature += item["lastPotentialAccumulatedFrictionEnergy"]
        signature += item["tangentialCandidateRatio"]
    for item in ablation_results:
        signature += item["tangentialCandidateRatio"]
    for item in sensitivity_results:
        signature += 0.1 * item["normalizedMinimumDistance"]

    exu.Print("generalContactPotentialBenchmarkMechanisms=", float(signature))


if __name__ == "__main__":
    main()
