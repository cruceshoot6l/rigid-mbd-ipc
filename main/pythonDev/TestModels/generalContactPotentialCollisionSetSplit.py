#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is an EXUDYN example
#
# Details:  Regression for IPC-compatible vs OGC feasible-region collision-set split
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
        referenceCoordinates=[0.0, 0.0, 0.09, 0.1, 0.0, 0.0],
        initialVelocities=[0.0] * 6,
    ))
    oRigid = mbs.AddObject(ObjectRigidBody(
        physicsMass=1.5,
        physicsInertia=[0.02, 0.02, 0.02, 0.0, 0.0, 0.0],
        nodeNumber=nRigid,
        visualization=VObjectRigidBody(graphicsData=[]),
    ))
    mRigid = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid, localPosition=[0.0, 0.0, 0.0]))

    nRigidFar = mbs.AddNode(NodeRigidBodyRxyz(
        referenceCoordinates=[0.35, 0.0, 0.15, 0.0, 0.0, 0.0],
        initialVelocities=[0.0] * 6,
    ))
    oRigidFar = mbs.AddObject(ObjectRigidBody(
        physicsMass=1.0,
        physicsInertia=[0.02, 0.02, 0.02, 0.0, 0.0, 0.0],
        nodeNumber=nRigidFar,
        visualization=VObjectRigidBody(graphicsData=[]),
    ))
    mRigidFar = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigidFar, localPosition=[0.0, 0.0, 0.0]))

    gContact = mbs.AddGeneralContact()
    gContact.contactFormulation = formulation
    gContact.computeContactForces = True
    gContact.barrierActivationDistance = 0.08
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
    gContact.AddRigidBodySurfaceMesh(mRigidFar, cube_points, cube_triangles, frictionMaterialIndex=0)

    mbs.Assemble()

    simulationSettings = exu.SimulationSettings()
    simulationSettings.timeIntegration.numberOfSteps = 1
    simulationSettings.timeIntegration.endTime = 1e-4
    simulationSettings.timeIntegration.verboseMode = 0
    simulationSettings.solutionSettings.writeSolutionToFile = False

    mbs.SolveDynamic(simulationSettings=simulationSettings, solverType=exu.DynamicSolverType.ExplicitEuler)

    pyData = gContact.GetPythonObject()
    contactForces = np.array(gContact.GetSystemODE2RhsContactForces(copy=True), dtype=float)
    return {
        "meshPairBuilderType": str(pyData["lastPotentialMeshPairBuilderType"]),
        "coarseBroadPhasePairs": int(pyData["lastPotentialContactCoarseBroadPhasePairs"]),
        "seedBuilderType": str(pyData["lastPotentialSeedBuilderType"]),
        "vfSeeds": int(pyData["lastPotentialContactVertexFaceSeeds"]),
        "eeSeeds": int(pyData["lastPotentialContactEdgeEdgeSeeds"]),
        "seedRejected": int(pyData["lastPotentialSeedRejectedCandidates"]),
        "broadPhaseRejected": int(pyData["lastPotentialBroadPhaseRejectedPairs"]),
        "builderType": str(pyData["lastPotentialCollisionSetBuilderType"]),
        "rejected": int(pyData["lastPotentialCollisionSetRejectedCandidates"]),
        "broadPhasePairs": int(pyData["lastPotentialContactBroadPhasePairs"]),
        "candidates": int(pyData["lastPotentialContactCandidates"]),
        "vf": int(pyData["lastPotentialContactVertexFaceCandidates"]),
        "ee": int(pyData["lastPotentialContactEdgeEdgeCandidates"]),
        "minimumDistance": float(pyData["lastPotentialContactMinimumDistance"]),
        "forceZ": float(contactForces[2]),
    }


def run_regression():
    gcp = run_case(exu.ContactFormulation.GCPBarrier)
    ogc = run_case(exu.ContactFormulation.OGCBarrier)

    if gcp["meshPairBuilderType"] != "IPCCompatible":
        raise ValueError("GCP must use the IPC-compatible mesh-pair broad phase")
    if ogc["meshPairBuilderType"] != "OGCFeasibleRegion":
        raise ValueError("OGC must use the dedicated feasible-region mesh-pair broad phase")
    if gcp["seedBuilderType"] != "IPCCompatible":
        raise ValueError("GCP must use the IPC-compatible seed builder")
    if ogc["seedBuilderType"] != "OGCFeasibleRegion":
        raise ValueError("OGC must use the dedicated feasible-region seed builder")
    if gcp["builderType"] != "IPCCompatible":
        raise ValueError("GCP must use the IPC-compatible collision-set builder")
    if ogc["builderType"] != "OGCFeasibleRegion":
        raise ValueError("OGC must use the dedicated feasible-region collision-set builder")
    if gcp["coarseBroadPhasePairs"] != ogc["coarseBroadPhasePairs"]:
        raise ValueError("IPC/GCP and OGC should start from the same coarse broad-phase pair count in this regression")
    if gcp["broadPhasePairs"] <= 0 or ogc["broadPhasePairs"] <= 0:
        raise ValueError("collision-set split regression expected overlapping broad-phase mesh pairs")
    if ogc["broadPhaseRejected"] <= 0:
        raise ValueError("OGC broad phase should reject at least one coarse mesh pair")
    if ogc["broadPhasePairs"] >= gcp["broadPhasePairs"]:
        raise ValueError("OGC mesh-pair broad phase should be more selective than the shared broad phase")
    if gcp["vfSeeds"] <= ogc["vfSeeds"] or gcp["eeSeeds"] <= ogc["eeSeeds"]:
        raise ValueError("OGC seed generation should reduce both VF and EE seeds in this regression")
    if ogc["seedRejected"] <= 0:
        raise ValueError("OGC seed generation should reject raw AABB seeds in this regression")
    if gcp["candidates"] <= 0 or ogc["candidates"] <= 0:
        raise ValueError("collision-set split regression expected active potential candidates")
    if gcp["vf"] <= 0 or gcp["ee"] <= 0:
        raise ValueError("shared collision-set builder should retain both VF and EE candidates in this regression")
    if ogc["vf"] <= 0 or ogc["ee"] <= 0:
        raise ValueError("OGC collision-set builder should keep both VF and EE candidates after filtering")
    if ogc["rejected"] <= 0:
        raise ValueError("OGC feasible-region builder should reject at least one shared candidate")
    if ogc["candidates"] >= gcp["candidates"]:
        raise ValueError("OGC collision-set builder should be more selective than the IPC-compatible builder")
    if ogc["minimumDistance"] >= 0.08 or gcp["minimumDistance"] >= 0.08:
        raise ValueError("collision-set split regression did not enter the activation zone")
    if gcp["forceZ"] <= 0.0 or ogc["forceZ"] <= 0.0:
        raise ValueError("collision-set split regression produced a non-repulsive response")

    return {"gcp": gcp, "ogc": ogc}


def main():
    results = run_regression()
    gcp = results["gcp"]
    ogc = results["ogc"]
    testValue = (
        float(gcp["coarseBroadPhasePairs"])
        + float(ogc["broadPhaseRejected"])
        + float(gcp["vfSeeds"])
        + float(gcp["eeSeeds"])
        + float(ogc["vfSeeds"])
        + float(ogc["eeSeeds"])
        + float(ogc["seedRejected"])
        + float(gcp["candidates"])
        + float(ogc["candidates"])
        + float(ogc["rejected"])
        + float(gcp["forceZ"])
        + float(ogc["forceZ"])
    )

    exu.Print("generalContactPotentialCollisionSetSplit=", testValue)


if __name__ == "__main__":
    main()
