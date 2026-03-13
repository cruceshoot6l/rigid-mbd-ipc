#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is an EXUDYN example
#
# Details:  Regression for IPC / GCP / OGC normal-potential model split
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


def run_case(formulation):
    SC = exu.SystemContainer()
    mbs = SC.AddSystem()

    ground = mbs.AddObject(ObjectGround())
    mGround = mbs.AddMarker(MarkerBodyRigid(bodyNumber=ground, localPosition=[0.0, 0.0, 0.0]))

    nRigid = mbs.AddNode(NodeRigidBodyRxyz(
        referenceCoordinates=[0.0, 0.0, 0.09, 0.0, 0.0, 0.0],
        initialVelocities=[0.0] * 6,
    ))
    oRigid = mbs.AddObject(ObjectRigidBody(
        physicsMass=1.5,
        physicsInertia=[0.02, 0.02, 0.02, 0.0, 0.0, 0.0],
        nodeNumber=nRigid,
        visualization=VObjectRigidBody(graphicsData=[]),
    ))
    mRigid = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid, localPosition=[0.0, 0.0, 0.0]))

    gContact = mbs.AddGeneralContact()
    gContact.contactFormulation = formulation
    gContact.computeContactForces = True
    gContact.barrierActivationDistance = 0.06
    gContact.barrierStiffness = 250.0
    gContact.barrierMinimumDistance = 1e-6
    gContact.useGaussNewtonHessian = True
    gContact.SetSearchTreeBox([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
    gContact.SetFrictionPairings([[0.0]])

    ground_points, ground_triangles = box_mesh(size_x=0.8, size_y=0.8, size_z=0.05)
    ground_points = [[p[0], p[1], p[2] - 0.025] for p in ground_points]
    cube_points, cube_triangles = box_mesh(size_x=0.1, size_y=0.1, size_z=0.1)

    gContact.AddRigidBodySurfaceMesh(mGround, ground_points, ground_triangles, frictionMaterialIndex=0, staticMesh=True)
    gContact.AddRigidBodySurfaceMesh(mRigid, cube_points, cube_triangles, frictionMaterialIndex=0)

    mbs.Assemble()

    simulationSettings = exu.SimulationSettings()
    simulationSettings.timeIntegration.numberOfSteps = 2
    simulationSettings.timeIntegration.endTime = 2e-4
    simulationSettings.timeIntegration.verboseMode = 0
    simulationSettings.solutionSettings.writeSolutionToFile = False

    mbs.SolveDynamic(simulationSettings=simulationSettings, solverType=exu.DynamicSolverType.ExplicitEuler)

    pyData = gContact.GetPythonObject()
    contactForces = np.array(gContact.GetSystemODE2RhsContactForces(copy=True), dtype=float)
    ode2_t = np.array(mbs.systemData.GetODE2Coordinates_t(), dtype=float)

    return {
        "builderType": str(pyData["lastPotentialCollisionSetBuilderType"]),
        "rejected": int(pyData["lastPotentialCollisionSetRejectedCandidates"]),
        "candidates": int(pyData["lastPotentialContactCandidates"]),
        "vf": int(pyData["lastPotentialContactVertexFaceCandidates"]),
        "ee": int(pyData["lastPotentialContactEdgeEdgeCandidates"]),
        "minimumDistance": float(pyData["lastPotentialContactMinimumDistance"]),
        "forceZ": float(contactForces[2]),
        "velocityZ": float(ode2_t[2]),
    }


def run_regression():
    ipc = run_case(exu.ContactFormulation.IPCBarrier)
    gcp = run_case(exu.ContactFormulation.GCPBarrier)
    ogc = run_case(exu.ContactFormulation.OGCBarrier)

    for case in [ipc, gcp, ogc]:
        if case["candidates"] <= 0 or case["vf"] <= 0:
            raise ValueError("expected active VF candidates in potential model split regression")
        if case["ee"] != 0:
            raise ValueError("model split regression should stay on the VF-only path")
        if case["minimumDistance"] >= 0.06:
            raise ValueError("model split regression did not enter the activation zone")
        if case["forceZ"] <= 0.0 or case["velocityZ"] <= 0.0:
            raise ValueError("potential model split regression produced non-repulsive response")

    if ipc["builderType"] != "IPCCompatible" or gcp["builderType"] != "IPCCompatible":
        raise ValueError("IPC and GCP should stay on the shared IPC-compatible collision-set builder")
    if ipc["candidates"] != gcp["candidates"]:
        raise ValueError("IPC and GCP should still share the same collision set in this regression")
    if ogc["builderType"] != "OGCFeasibleRegion":
        raise ValueError("OGC must use the dedicated feasible-region collision-set builder")
    if abs(ipc["forceZ"] - ogc["forceZ"]) > 1e-6 * max(abs(ipc["forceZ"]), 1.0):
        raise ValueError("IPC and OGC should share the same barrier normal model in the split regression")
    if abs(ipc["forceZ"] - gcp["forceZ"]) <= 1e-3 * max(abs(ipc["forceZ"]), 1.0):
        raise ValueError("GCP response is too close to IPC; normal-model split not effective")
    if abs(gcp["velocityZ"] - ipc["velocityZ"]) <= 1e-4 * max(abs(ipc["velocityZ"]), 1.0):
        raise ValueError("GCP should generate a measurably different dynamic response than IPC/OGC")

    return {"ipc": ipc, "gcp": gcp, "ogc": ogc}


def main():
    results = run_regression()
    testValue = (
        results["ipc"]["forceZ"]
        + results["ogc"]["forceZ"]
        + results["gcp"]["forceZ"]
        + results["ipc"]["candidates"]
    )
    exu.Print("generalContactPotentialModelSplit=", testValue)


if __name__ == "__main__":
    main()
