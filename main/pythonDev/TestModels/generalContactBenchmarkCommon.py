#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is an EXUDYN example
#
# Details:  Week-6 benchmark helpers for penalty vs GCP barrier contact
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import csv
import json
import time
from pathlib import Path

import numpy as np

import exudyn as exu
from exudyn.itemInterface import *


FORMULATION_PENALTY = "Penalty"
FORMULATION_GCP = "GCPBarrier"

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "local" / "results"
SUMMARY_PATH = REPO_ROOT / "local" / "design" / "week6-benchmark-summary.md"
RESULTS_JSON_PATH = RESULTS_DIR / "week6-general-contact-benchmarks.json"
RESULTS_CSV_PATH = RESULTS_DIR / "week6-general-contact-benchmarks.csv"


def box_mesh(size_x=0.1, size_y=0.1, size_z=0.1):
    hx = 0.5 * size_x
    hy = 0.5 * size_y
    hz = 0.5 * size_z
    points = [
        [-hx, -hy, -hz],
        [hx, -hy, -hz],
        [hx, hy, -hz],
        [-hx, hy, -hz],
        [-hx, -hy, hz],
        [hx, -hy, hz],
        [hx, hy, hz],
        [-hx, hy, hz],
    ]
    triangles = [
        [0, 1, 2], [0, 2, 3],
        [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1],
        [1, 5, 6], [1, 6, 2],
        [2, 6, 7], [2, 7, 3],
        [3, 7, 4], [3, 4, 0],
    ]
    return points, triangles


def shift_points(points, shift):
    return [[p[0] + shift[0], p[1] + shift[1], p[2] + shift[2]] for p in points]


def to_python_float(value, default=None):
    value = float(value)
    if np.isfinite(value):
        return value
    return default


def add_coordinate_ground(mbs):
    n_ground = mbs.AddNode(NodePointGround(referenceCoordinates=[0.0, 0.0, 0.0]))
    m_coordinate_ground = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=n_ground, coordinate=0))
    return n_ground, m_coordinate_ground


def add_rigid_body(mbs, reference_coordinates, mass=1.5, inertia_value=0.02):
    node_number = mbs.AddNode(NodeRigidBodyRxyz(
        referenceCoordinates=reference_coordinates,
        initialVelocities=[0.0] * 6,
    ))
    body_number = mbs.AddObject(ObjectRigidBody(
        physicsMass=mass,
        physicsInertia=[inertia_value, inertia_value, inertia_value, 0.0, 0.0, 0.0],
        nodeNumber=node_number,
        visualization=VObjectRigidBody(graphicsData=[]),
    ))
    marker_number = mbs.AddMarker(MarkerBodyRigid(bodyNumber=body_number, localPosition=[0.0, 0.0, 0.0]))
    return node_number, body_number, marker_number


def add_coordinate_constraints(mbs, node_number, m_coordinate_ground, constrained_coordinates):
    for coordinate in constrained_coordinates:
        marker = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=node_number, coordinate=coordinate))
        mbs.AddObject(CoordinateConstraint(
            markerNumbers=[m_coordinate_ground, marker],
            visualization=VCoordinateConstraint(show=False),
        ))


def configure_general_contact(mbs, formulation, search_box_half_size=1.0):
    g_contact = mbs.AddGeneralContact()
    g_contact.computeContactForces = True
    g_contact.SetFrictionPairings([[0.0]])
    g_contact.SetSearchTreeBox(
        [-search_box_half_size, -search_box_half_size, -search_box_half_size],
        [search_box_half_size, search_box_half_size, search_box_half_size],
    )

    if formulation == FORMULATION_GCP:
        g_contact.contactFormulation = exu.ContactFormulation.GCPBarrier
        g_contact.barrierActivationDistance = 0.08
        g_contact.barrierStiffness = 2000.0
        g_contact.barrierMinimumDistance = 1e-6
        g_contact.useGaussNewtonHessian = True
        g_contact.useNonlinearCCDStepFilter = True
        g_contact.ccdTolerance = 1e-6

    return g_contact


def add_penalty_plane(g_contact, ground_marker, axis):
    if axis == "z":
        plane_points = [[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0]]
    elif axis == "x":
        plane_points = [[0.0, -0.5, -0.5], [0.0, 0.5, -0.5], [0.0, 0.5, 0.5], [0.0, -0.5, 0.5]]
    else:
        raise ValueError("unsupported penalty plane axis")
    plane_triangles = [[0, 1, 2], [0, 2, 3]]
    g_contact.AddTrianglesRigidBodyBased(
        rigidBodyMarkerIndex=ground_marker,
        contactStiffness=2e4,
        contactDamping=200.0,
        frictionMaterialIndex=0,
        pointList=plane_points,
        triangleList=plane_triangles,
        staticTriangles=True,
    )


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


def add_contact_body_geometry(g_contact, formulation, body_marker, characteristic_size):
    if formulation == FORMULATION_PENALTY:
        g_contact.AddSphereWithMarker(
            body_marker,
            radius=characteristic_size,
            contactStiffness=2e4,
            contactDamping=200.0,
            frictionMaterialIndex=0,
        )
    else:
        cube_points, cube_triangles = box_mesh(2.0 * characteristic_size, 2.0 * characteristic_size, 2.0 * characteristic_size)
        g_contact.AddRigidBodySurfaceMesh(
            body_marker,
            cube_points,
            cube_triangles,
            frictionMaterialIndex=0,
        )


def make_dynamic_settings(number_of_steps, end_time):
    settings = exu.SimulationSettings()
    settings.solutionSettings.writeSolutionToFile = False
    settings.solutionSettings.sensorsStoreAndWriteFiles = False
    settings.solutionSettings.solutionInformation = "Week-6 general contact benchmarks"
    settings.timeIntegration.numberOfSteps = number_of_steps
    settings.timeIntegration.endTime = end_time
    settings.timeIntegration.verboseMode = 0
    settings.timeIntegration.newton.relativeTolerance = 1e-8
    settings.timeIntegration.newton.absoluteTolerance = 1e-10
    settings.timeIntegration.newton.maxIterations = 30
    settings.timeIntegration.newton.useModifiedNewton = False
    settings.timeIntegration.generalizedAlpha.computeInitialAccelerations = True
    return settings


def solve_dynamic_system(mbs, number_of_steps, end_time):
    solver = exu.MainSolverImplicitSecondOrder()
    settings = make_dynamic_settings(number_of_steps, end_time)
    start_time = time.perf_counter()
    success = solver.SolveSystem(mbs, settings)
    elapsed = time.perf_counter() - start_time
    if not success:
        raise RuntimeError("dynamic solver failed")
    return solver, elapsed


def extract_contact_force_norm(g_contact):
    forces = np.array(g_contact.GetSystemODE2RhsContactForces(copy=True), dtype=float)
    if forces.size == 0:
        return 0.0
    return float(np.linalg.norm(forces))


def extract_contact_metrics(name, formulation, solver, elapsed, g_contact, max_penetration, additional_metrics=None):
    py_data = g_contact.GetPythonObject()
    result = {
        "benchmark": name,
        "formulation": formulation,
        "solverName": solver.GetSolverName(),
        "newtonStepsCount": int(solver.it.newtonStepsCount),
        "solveTimeSeconds": float(elapsed),
        "maxPenetration": float(max_penetration),
        "contactForceNorm": extract_contact_force_norm(g_contact),
        "totalPotentialCCDClippedSteps": int(py_data["totalPotentialCCDClippedSteps"]),
        "totalPotentialCCDStepFailures": int(py_data["totalPotentialCCDStepFailures"]),
        "lastPotentialContactCandidates": int(py_data["lastPotentialContactCandidates"]),
        "lastPotentialContactMinimumDistance": to_python_float(py_data["lastPotentialContactMinimumDistance"]),
        "lastPotentialCCDMinimumDistance": to_python_float(py_data["lastPotentialCCDMinimumDistance"]),
    }
    if additional_metrics is not None:
        result.update(additional_metrics)
    return result


def run_sphere_plane_benchmark(formulation):
    characteristic_size = 0.05
    SC = exu.SystemContainer()
    mbs = SC.AddSystem()

    g_contact = configure_general_contact(mbs, formulation)
    o_ground = mbs.AddObject(ObjectGround())
    m_ground_body = mbs.AddMarker(MarkerBodyRigid(bodyNumber=o_ground, localPosition=[0.0, 0.0, 0.0]))
    n_coordinate_ground, m_coordinate_ground = add_coordinate_ground(mbs)

    n_rigid, _, m_rigid = add_rigid_body(mbs, [0.0, 0.0, 0.09, 0.0, 0.0, 0.0], mass=1.5, inertia_value=0.02)
    add_coordinate_constraints(mbs, n_rigid, m_coordinate_ground, [0, 1, 3, 4, 5])

    m_rigid_z = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=n_rigid, coordinate=2))
    mbs.AddObject(CoordinateSpringDamper(
        markerNumbers=[m_coordinate_ground, m_rigid_z],
        stiffness=500.0,
        damping=5.0,
        offset=0.09,
        visualization=VCoordinateSpringDamper(show=False),
    ))
    mbs.AddLoad(Force(markerNumber=m_rigid, loadVector=[0.0, 0.0, -2500.0]))

    if formulation == FORMULATION_PENALTY:
        add_penalty_plane(g_contact, mbs.AddMarker(MarkerNodeRigid(nodeNumber=n_coordinate_ground)), "z")
    else:
        add_barrier_plane(g_contact, m_ground_body, "z")
    add_contact_body_geometry(g_contact, formulation, m_rigid, characteristic_size)

    mbs.Assemble()
    solver, elapsed = solve_dynamic_system(mbs, number_of_steps=10, end_time=0.01)

    position = np.array(mbs.GetNodeOutput(n_rigid, exu.OutputVariableType.Position), dtype=float)
    max_penetration = max(0.0, characteristic_size - position[2])

    return extract_contact_metrics(
        "sphere_plane",
        formulation,
        solver,
        elapsed,
        g_contact,
        max_penetration,
        {"bodyPositionZ": float(position[2])},
    )


def run_guided_slider_benchmark(formulation):
    characteristic_size = 0.05
    SC = exu.SystemContainer()
    mbs = SC.AddSystem()

    g_contact = configure_general_contact(mbs, formulation)
    o_ground = mbs.AddObject(ObjectGround())
    m_ground_body = mbs.AddMarker(MarkerBodyRigid(bodyNumber=o_ground, localPosition=[0.0, 0.0, 0.0]))
    n_coordinate_ground, m_coordinate_ground = add_coordinate_ground(mbs)

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

    if formulation == FORMULATION_PENALTY:
        add_penalty_plane(g_contact, mbs.AddMarker(MarkerNodeRigid(nodeNumber=n_coordinate_ground)), "x")
    else:
        add_barrier_plane(g_contact, m_ground_body, "x")
    add_contact_body_geometry(g_contact, formulation, m_slider, characteristic_size)

    mbs.Assemble()
    solver, elapsed = solve_dynamic_system(mbs, number_of_steps=10, end_time=0.01)

    position = np.array(mbs.GetNodeOutput(n_slider, exu.OutputVariableType.Position), dtype=float)
    max_penetration = max(0.0, characteristic_size - position[0])

    return extract_contact_metrics(
        "guided_slider_stop",
        formulation,
        solver,
        elapsed,
        g_contact,
        max_penetration,
        {"bodyPositionX": float(position[0])},
    )


def run_articulated_arm_benchmark(formulation):
    characteristic_size = 0.05
    arm_length = 0.1
    SC = exu.SystemContainer()
    mbs = SC.AddSystem()

    g_contact = configure_general_contact(mbs, formulation)
    o_ground = mbs.AddObject(ObjectGround())
    m_ground_body = mbs.AddMarker(MarkerBodyRigid(bodyNumber=o_ground, localPosition=[0.0, 0.0, 0.0]))
    n_coordinate_ground, m_coordinate_ground = add_coordinate_ground(mbs)

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

    if formulation == FORMULATION_PENALTY:
        add_penalty_plane(g_contact, mbs.AddMarker(MarkerNodeRigid(nodeNumber=n_coordinate_ground)), "z")
    else:
        add_barrier_plane(g_contact, m_ground_body, "z")
    add_contact_body_geometry(g_contact, formulation, m_tip, characteristic_size)

    mbs.Assemble()
    solver, elapsed = solve_dynamic_system(mbs, number_of_steps=20, end_time=0.02)

    tip_position = np.array(mbs.GetMarkerOutput(m_tip, exu.OutputVariableType.Position), dtype=float)
    max_penetration = max(0.0, characteristic_size - tip_position[2])

    return extract_contact_metrics(
        "articulated_arm_stop",
        formulation,
        solver,
        elapsed,
        g_contact,
        max_penetration,
        {"tipPositionZ": float(tip_position[2])},
    )


BENCHMARKS = {
    "sphere_plane": run_sphere_plane_benchmark,
    "guided_slider_stop": run_guided_slider_benchmark,
    "articulated_arm_stop": run_articulated_arm_benchmark,
}


def run_single_benchmark(name):
    benchmark_function = BENCHMARKS[name]
    return [
        benchmark_function(FORMULATION_PENALTY),
        benchmark_function(FORMULATION_GCP),
    ]


def run_all_benchmarks():
    results = []
    for benchmark_name in BENCHMARKS:
        results.extend(run_single_benchmark(benchmark_name))
    return results


def write_results_files(results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with RESULTS_JSON_PATH.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    header = [
        "benchmark",
        "formulation",
        "maxPenetration",
        "newtonStepsCount",
        "solveTimeSeconds",
        "totalPotentialCCDClippedSteps",
        "totalPotentialCCDStepFailures",
        "lastPotentialContactCandidates",
        "lastPotentialContactMinimumDistance",
        "lastPotentialCCDMinimumDistance",
        "contactForceNorm",
    ]
    with RESULTS_CSV_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        for item in results:
            writer.writerow({key: item.get(key, "") for key in header})


def build_summary_markdown(results):
    grouped = {}
    for item in results:
        grouped.setdefault(item["benchmark"], {})[item["formulation"]] = item

    lines = [
        "# Week 6 Benchmark Summary",
        "",
        "## Comparison Table",
        "",
        "| Benchmark | Penalty max penetration | GCP max penetration | Penetration ratio | Penalty Newton steps | GCP Newton steps | GCP clipped steps |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    penetration_improvements = []
    mechanism_improved = False
    for benchmark_name, pair in grouped.items():
        penalty = pair[FORMULATION_PENALTY]
        gcp = pair[FORMULATION_GCP]
        penalty_pen = max(penalty["maxPenetration"], 1e-16)
        ratio = gcp["maxPenetration"] / penalty_pen
        penetration_improvements.append(1.0 - ratio)
        if benchmark_name == "articulated_arm_stop" and gcp["maxPenetration"] < penalty["maxPenetration"]:
            mechanism_improved = True

        lines.append(
            f"| {benchmark_name} | {penalty['maxPenetration']:.6e} | {gcp['maxPenetration']:.6e} | {ratio:.3f} | "
            f"{penalty['newtonStepsCount']} | {gcp['newtonStepsCount']} | {gcp['totalPotentialCCDClippedSteps']} |"
        )

    average_improvement = 100.0 * np.mean(penetration_improvements) if penetration_improvements else 0.0

    lines += [
        "",
        "## Publishable Claims",
        "",
        f"- `GCPBarrier` reduced maximum penetration on all {len(grouped)} benchmark scenes with an average relative reduction of {average_improvement:.1f}%.",
        f"- The mechanism-like `articulated_arm_stop` benchmark {'did' if mechanism_improved else 'did not'} outperform penalty contact on penetration in this week-6 suite.",
        "- The current benchmark package uses the existing sphere-based penalty baseline against a cube-mesh barrier proxy with matching half-size, which is acceptable for internal comparison but still needs a tighter geometry match before publication.",
        "",
        "## Remaining Gaps",
        "",
        "- The rigid stack benchmark was deferred after exposing `VF`-only limitations under multi-contact conditions; it should return once `EE` support is added for the failing cases.",
        "- Force smoothness is still only assessed qualitatively; the current week-6 suite is dynamic, but it does not yet export a dedicated time-history metric.",
        "- The feasible-step filter records zero clipped steps in these short benchmark windows; week-5 regression remains the direct acceptance test for clipping behavior.",
        "",
        "## Result Files",
        "",
        f"- JSON: `{RESULTS_JSON_PATH}`",
        f"- CSV: `{RESULTS_CSV_PATH}`",
    ]

    return "\n".join(lines) + "\n"


def write_summary_file(results):
    summary = build_summary_markdown(results)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(summary, encoding="utf-8")


def benchmark_signature(results):
    total = 0.0
    for item in results:
        total += item["maxPenetration"]
        total += 0.01 * item["newtonStepsCount"]
        total += 0.001 * item["totalPotentialCCDClippedSteps"]
    return float(total)
