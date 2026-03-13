#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is an EXUDYN example
#
# Details:  Regression for tangential/frictional potential contact on a sliding rigid body
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np

import exudyn as exu
from exudyn.itemInterface import *


INITIAL_VELOCITY_X = 0.05


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


def run_case(formulation, enable_friction):
    SC = exu.SystemContainer()
    mbs = SC.AddSystem()

    o_ground = mbs.AddObject(ObjectGround())
    m_ground = mbs.AddMarker(MarkerBodyRigid(bodyNumber=o_ground, localPosition=[0.0, 0.0, 0.0]))

    n_coordinate_ground = mbs.AddNode(NodePointGround(referenceCoordinates=[0.0, 0.0, 0.0]))
    m_coordinate_ground = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=n_coordinate_ground, coordinate=0))

    n_rigid = mbs.AddNode(NodeRigidBodyRxyz(
        referenceCoordinates=[0.0, 0.0, 0.085, 0.0, 0.0, 0.0],
        initialVelocities=[INITIAL_VELOCITY_X, 0.0, 0.0, 0.0, 0.0, 0.0],
    ))
    o_rigid = mbs.AddObject(ObjectRigidBody(
        physicsMass=1.0,
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

    m_rigid_z = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=n_rigid, coordinate=2))
    mbs.AddObject(CoordinateSpringDamper(
        markerNumbers=[m_coordinate_ground, m_rigid_z],
        stiffness=1200.0,
        damping=40.0,
        offset=0.085,
        visualization=VCoordinateSpringDamper(show=False),
    ))
    mbs.AddLoad(Force(markerNumber=m_rigid, loadVector=[0.0, 0.0, -80.0]))

    g_contact = mbs.AddGeneralContact()
    g_contact.contactFormulation = formulation
    g_contact.computeContactForces = True
    g_contact.barrierActivationDistance = 0.08
    g_contact.barrierStiffness = 300.0 if (enable_friction and formulation == exu.ContactFormulation.GCPBarrier) else 250.0
    g_contact.barrierMinimumDistance = 1e-6
    g_contact.enablePotentialFriction = enable_friction
    g_contact.frictionProportionalZone = 1.0
    g_contact.useGaussNewtonHessian = True
    g_contact.useNonlinearCCDStepFilter = True
    g_contact.ccdTolerance = 1e-6
    g_contact.SetSearchTreeBox([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
    g_contact.SetFrictionPairings([[0.6]])

    floor_points, floor_triangles = box_mesh(size_x=1.2, size_y=1.2, size_z=0.05)
    floor_points = shift_points(floor_points, [0.0, 0.0, -0.025])
    cube_points, cube_triangles = box_mesh(size_x=0.1, size_y=0.1, size_z=0.1)

    g_contact.AddRigidBodySurfaceMesh(m_ground, floor_points, floor_triangles, frictionMaterialIndex=0, staticMesh=True)
    g_contact.AddRigidBodySurfaceMesh(m_rigid, cube_points, cube_triangles, frictionMaterialIndex=0)

    mbs.Assemble()

    simulation_settings = exu.SimulationSettings()
    simulation_settings.solutionSettings.writeSolutionToFile = False
    simulation_settings.solutionSettings.sensorsStoreAndWriteFiles = False
    simulation_settings.timeIntegration.numberOfSteps = 2
    simulation_settings.timeIntegration.endTime = 0.002
    simulation_settings.timeIntegration.verboseMode = 0
    simulation_settings.timeIntegration.newton.relativeTolerance = 1e-8
    simulation_settings.timeIntegration.newton.absoluteTolerance = 1e-10
    simulation_settings.timeIntegration.newton.maxIterations = 40
    simulation_settings.timeIntegration.generalizedAlpha.computeInitialAccelerations = True

    solver = exu.MainSolverImplicitSecondOrder()
    if not solver.SolveSystem(mbs, simulation_settings):
        raise RuntimeError("potential friction regression failed to solve")

    py_data = g_contact.GetPythonObject()
    velocity = np.array(mbs.GetNodeOutput(n_rigid, exu.OutputVariableType.Velocity), dtype=float)
    position = np.array(mbs.GetNodeOutput(n_rigid, exu.OutputVariableType.Position), dtype=float)
    contact_forces = np.array(g_contact.GetSystemODE2RhsContactForces(copy=True), dtype=float)

    return {
        "velocityX": float(velocity[0]),
        "positionZ": float(position[2]),
        "forceX": float(contact_forces[0]),
        "forceZ": float(contact_forces[2]),
        "vfCandidates": int(py_data["lastPotentialContactVertexFaceCandidates"]),
        "eeCandidates": int(py_data["lastPotentialContactEdgeEdgeCandidates"]),
        "contactCandidates": int(py_data["lastPotentialContactCandidates"]),
        "tangentialCandidates": int(py_data["lastPotentialTangentialCandidates"]),
        "frictionEnergy": float(py_data["lastPotentialAccumulatedFrictionEnergy"]),
        "controllerType": str(py_data["lastPotentialStepControllerType"]),
        "hadFailure": bool(py_data["lastPotentialCCDHadFailure"]),
    }


def main():
    baseline = run_case(exu.ContactFormulation.GCPBarrier, enable_friction=False)
    ipc = run_case(exu.ContactFormulation.IPCBarrier, enable_friction=True)
    gcp = run_case(exu.ContactFormulation.GCPBarrier, enable_friction=True)
    ogc = run_case(exu.ContactFormulation.OGCBarrier, enable_friction=True)

    if baseline["tangentialCandidates"] != 0:
        raise ValueError("frictionless baseline should not report tangential potential candidates")
    if baseline["frictionEnergy"] != 0.0:
        raise ValueError("frictionless baseline should not accumulate tangential friction energy")

    for name, case in [("IPC", ipc), ("GCP", gcp), ("OGC", ogc)]:
        if case["tangentialCandidates"] <= 0:
            raise ValueError(f"{name} friction regression expected active tangential candidates")
        if case["frictionEnergy"] <= 0.0:
            raise ValueError(f"{name} friction regression expected positive friction energy")
        if case["forceX"] >= 0.0:
            raise ValueError(f"{name} friction regression expected a tangential force opposing positive sliding velocity")
        if case["forceZ"] <= 0.0:
            raise ValueError(f"{name} friction regression expected positive normal support force")
        if case["velocityX"] >= INITIAL_VELOCITY_X:
            raise ValueError(f"{name} friction regression did not slow down the tangential motion")
        if case["positionZ"] <= 0.05:
            raise ValueError(f"{name} friction regression dropped below the floor reference height")
        if case["hadFailure"]:
            raise ValueError(f"{name} friction regression reported a feasible-step failure")

    if gcp["velocityX"] >= baseline["velocityX"]:
        raise ValueError("GCP friction case must reduce tangential velocity relative to the frictionless baseline")
    if ipc["controllerType"] != "CCDLineSearch" or gcp["controllerType"] != "CCDLineSearch":
        raise ValueError("IPC and GCP friction cases must stay on CCD line search")
    if ogc["controllerType"] != "TrustRegion":
        raise ValueError("OGC friction case must stay on trust region")

    test_value = (
        float(baseline["velocityX"])
        + float(ipc["tangentialCandidates"] + gcp["tangentialCandidates"] + ogc["tangentialCandidates"])
        + float(ipc["frictionEnergy"] + gcp["frictionEnergy"] + ogc["frictionEnergy"])
        + float(-ipc["forceX"] - gcp["forceX"] - ogc["forceX"])
    )

    exu.Print("generalContactPotentialFriction=", test_value)


if __name__ == "__main__":
    main()
