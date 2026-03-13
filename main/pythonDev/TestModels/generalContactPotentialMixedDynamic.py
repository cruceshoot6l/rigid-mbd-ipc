#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is an EXUDYN example
#
# Details:  Dynamic mixed-contact regression covering VF + EE and step-controller split
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np

import exudyn as exu
from exudyn.itemInterface import *


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


def run_case(formulation):
    SC = exu.SystemContainer()
    mbs = SC.AddSystem()

    o_ground = mbs.AddObject(ObjectGround())
    m_ground = mbs.AddMarker(MarkerBodyRigid(bodyNumber=o_ground, localPosition=[0.0, 0.0, 0.0]))

    n_coordinate_ground = mbs.AddNode(NodePointGround(referenceCoordinates=[0.0, 0.0, 0.0]))
    m_coordinate_ground = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=n_coordinate_ground, coordinate=0))

    n_rigid = mbs.AddNode(NodeRigidBodyRxyz(
        referenceCoordinates=[0.042, 0.0, 0.092, 0.0, 0.0, 0.0],
        initialVelocities=[0.0] * 6,
    ))
    o_rigid = mbs.AddObject(ObjectRigidBody(
        physicsMass=1.5,
        physicsInertia=[0.02, 0.02, 0.02, 0.0, 0.0, 0.0],
        nodeNumber=n_rigid,
        visualization=VObjectRigidBody(graphicsData=[]),
    ))
    m_rigid = mbs.AddMarker(MarkerBodyRigid(bodyNumber=o_rigid, localPosition=[0.0, 0.0, 0.0]))

    for coordinate in [1, 3, 4, 5]:
        marker = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=n_rigid, coordinate=coordinate))
        mbs.AddObject(CoordinateConstraint(
            markerNumbers=[m_coordinate_ground, marker],
            visualization=VCoordinateConstraint(show=False),
        ))

    for coordinate, offset in [(0, 0.042), (2, 0.092)]:
        marker = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=n_rigid, coordinate=coordinate))
        mbs.AddObject(CoordinateSpringDamper(
            markerNumbers=[m_coordinate_ground, marker],
            stiffness=2000.0,
            damping=30.0,
            offset=offset,
            visualization=VCoordinateSpringDamper(show=False),
        ))

    mbs.AddLoad(Force(markerNumber=m_rigid, loadVector=[-180.0, 0.0, -650.0]))

    g_contact = mbs.AddGeneralContact()
    g_contact.contactFormulation = formulation
    g_contact.computeContactForces = True
    g_contact.barrierActivationDistance = 0.07
    g_contact.barrierStiffness = 1500.0
    g_contact.barrierMinimumDistance = 1e-6
    g_contact.useGaussNewtonHessian = True
    g_contact.useNonlinearCCDStepFilter = True
    g_contact.ccdTolerance = 1e-6
    g_contact.SetSearchTreeBox([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
    g_contact.SetFrictionPairings([[0.0]])

    floor_points, floor_triangles = box_mesh(size_x=1.0, size_y=1.0, size_z=0.05)
    floor_points = shift_points(floor_points, [0.0, 0.0, -0.025])
    wall_points, wall_triangles = box_mesh(size_x=0.05, size_y=0.4, size_z=0.4)
    wall_points = shift_points(wall_points, [0.105, 0.0, 0.11])
    cube_points, cube_triangles = box_mesh(size_x=0.1, size_y=0.1, size_z=0.1)

    g_contact.AddRigidBodySurfaceMesh(m_ground, floor_points, floor_triangles, frictionMaterialIndex=0, staticMesh=True)
    g_contact.AddRigidBodySurfaceMesh(m_ground, wall_points, wall_triangles, frictionMaterialIndex=0, staticMesh=True)
    g_contact.AddRigidBodySurfaceMesh(m_rigid, cube_points, cube_triangles, frictionMaterialIndex=0)

    mbs.Assemble()

    simulation_settings = exu.SimulationSettings()
    simulation_settings.solutionSettings.writeSolutionToFile = False
    simulation_settings.solutionSettings.sensorsStoreAndWriteFiles = False
    simulation_settings.timeIntegration.numberOfSteps = 20
    simulation_settings.timeIntegration.endTime = 0.01
    simulation_settings.timeIntegration.verboseMode = 0
    simulation_settings.timeIntegration.newton.relativeTolerance = 1e-8
    simulation_settings.timeIntegration.newton.absoluteTolerance = 1e-10
    simulation_settings.timeIntegration.newton.maxIterations = 30
    simulation_settings.timeIntegration.generalizedAlpha.computeInitialAccelerations = True

    solver = exu.MainSolverImplicitSecondOrder()
    if not solver.SolveSystem(mbs, simulation_settings):
        raise RuntimeError("mixed-contact dynamic regression failed to solve")

    py_data = g_contact.GetPythonObject()
    force_norm = float(np.linalg.norm(np.array(g_contact.GetSystemODE2RhsContactForces(copy=True), dtype=float)))
    position = np.array(mbs.GetNodeOutput(n_rigid, exu.OutputVariableType.Position), dtype=float)

    return {
        "position": position,
        "forceNorm": force_norm,
        "vf": int(py_data["lastPotentialContactVertexFaceCandidates"]),
        "ee": int(py_data["lastPotentialContactEdgeEdgeCandidates"]),
        "tangential": int(py_data["lastPotentialTangentialCandidates"]),
        "candidates": int(py_data["lastPotentialContactCandidates"]),
        "minimumDistance": float(py_data["lastPotentialContactMinimumDistance"]),
        "ccdMinimumDistance": float(py_data["lastPotentialCCDMinimumDistance"]),
        "controllerType": str(py_data["lastPotentialStepControllerType"]),
        "hadFailure": bool(py_data["lastPotentialCCDHadFailure"]),
        "alpha": float(py_data["lastPotentialCCDAlpha"]),
    }


def run_regression():
    ipc = run_case(exu.ContactFormulation.IPCBarrier)
    gcp = run_case(exu.ContactFormulation.GCPBarrier)
    ogc = run_case(exu.ContactFormulation.OGCBarrier)

    for name, case in [("IPC", ipc), ("GCP", gcp), ("OGC", ogc)]:
        if case["vf"] <= 0 or case["ee"] <= 0:
            raise ValueError(f"{name} mixed-contact regression must retain both VF and EE candidates")
        if case["candidates"] < case["vf"] + case["ee"]:
            raise ValueError(f"{name} mixed-contact regression returned inconsistent candidate counts")
        if not np.isfinite(case["minimumDistance"]) or case["minimumDistance"] <= 0.0:
            raise ValueError(f"{name} mixed-contact regression returned invalid minimum distance")
        if not np.isfinite(case["ccdMinimumDistance"]) or case["ccdMinimumDistance"] <= 0.0:
            raise ValueError(f"{name} mixed-contact regression returned invalid step-filter minimum distance")
        if case["hadFailure"]:
            raise ValueError(f"{name} mixed-contact regression reported a feasible-step failure")
        if case["alpha"] <= 0.0 or case["alpha"] > 1.0:
            raise ValueError(f"{name} mixed-contact regression returned invalid feasible-step alpha")
        if case["forceNorm"] <= 0.0:
            raise ValueError(f"{name} mixed-contact regression produced zero contact-force norm")
        if case["position"][2] <= 0.05:
            raise ValueError(f"{name} mixed-contact regression dropped the body below the floor reference height")

    if ipc["controllerType"] != "CCDLineSearch" or gcp["controllerType"] != "CCDLineSearch":
        raise ValueError("IPC and GCP must stay on CCD line search in mixed-contact regression")
    if ogc["controllerType"] != "TrustRegion":
        raise ValueError("OGC must stay on trust region in mixed-contact regression")

    return {"ipc": ipc, "gcp": gcp, "ogc": ogc}


def main():
    results = run_regression()
    ipc = results["ipc"]
    gcp = results["gcp"]
    ogc = results["ogc"]
    test_value = (
        float(ipc["vf"] + ipc["ee"] + ipc["candidates"])
        + float(gcp["vf"] + gcp["ee"] + gcp["candidates"])
        + float(ogc["vf"] + ogc["ee"] + ogc["candidates"])
        + float(ipc["forceNorm"] + gcp["forceNorm"] + ogc["forceNorm"])
    )

    exu.Print("generalContactPotentialMixedDynamic=", test_value)


if __name__ == "__main__":
    main()
