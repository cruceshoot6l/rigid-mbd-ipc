#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is an EXUDYN example
#
# Details:  Regression for CCD-line-search vs trust-region step-controller split
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

    oGround = mbs.AddObject(ObjectGround())
    mGround = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oGround, localPosition=[0.0, 0.0, 0.0]))

    nCoordinateGround = mbs.AddNode(NodePointGround(referenceCoordinates=[0.0, 0.0, 0.0]))
    mCoordinateGround = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=nCoordinateGround, coordinate=0))

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

    for coordinate in [0, 1, 3, 4, 5]:
        mCoordinate = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=nRigid, coordinate=coordinate))
        mbs.AddObject(CoordinateConstraint(
            markerNumbers=[mCoordinateGround, mCoordinate],
            visualization=VCoordinateConstraint(show=False),
        ))

    mRigidZ = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=nRigid, coordinate=2))
    mbs.AddObject(CoordinateSpringDamper(
        markerNumbers=[mCoordinateGround, mRigidZ],
        stiffness=500.0,
        damping=0.0,
        offset=0.09,
        visualization=VCoordinateSpringDamper(show=False),
    ))
    mbs.AddLoad(Force(markerNumber=mRigid, loadVector=[0.0, 0.0, -2500.0]))

    gContact = mbs.AddGeneralContact()
    gContact.contactFormulation = formulation
    gContact.computeContactForces = True
    gContact.barrierActivationDistance = 0.08
    gContact.barrierStiffness = 500.0
    gContact.barrierMinimumDistance = 1e-6
    gContact.useGaussNewtonHessian = True
    gContact.useNonlinearCCDStepFilter = True
    gContact.ccdTolerance = 1e-6
    gContact.SetSearchTreeBox([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
    gContact.SetFrictionPairings([[0.0]])

    ground_points, ground_triangles = box_mesh(size_x=1.0, size_y=1.0, size_z=0.05)
    ground_points = [[p[0], p[1], p[2] - 0.025] for p in ground_points]
    cube_points, cube_triangles = box_mesh(size_x=0.1, size_y=0.1, size_z=0.1)

    gContact.AddRigidBodySurfaceMesh(mGround, ground_points, ground_triangles, frictionMaterialIndex=0, staticMesh=True)
    gContact.AddRigidBodySurfaceMesh(mRigid, cube_points, cube_triangles, frictionMaterialIndex=0)

    mbs.Assemble()

    simulationSettings = exu.SimulationSettings()
    simulationSettings.solutionSettings.writeSolutionToFile = False
    simulationSettings.staticSolver.verboseMode = 0
    simulationSettings.staticSolver.numberOfLoadSteps = 1
    simulationSettings.staticSolver.newton.relativeTolerance = 1e-10
    simulationSettings.staticSolver.newton.absoluteTolerance = 1e-8
    simulationSettings.staticSolver.newton.maxIterations = 20

    mbs.SolveStatic(simulationSettings)
    return gContact.GetPythonObject(), np.array(mbs.systemData.GetODE2Coordinates(), dtype=float)


def run_regression():
    pyDataGCP, coordinatesGCP = run_case(exu.ContactFormulation.GCPBarrier)
    pyDataOGC, coordinatesOGC = run_case(exu.ContactFormulation.OGCBarrier)

    if pyDataGCP["potentialStepControllerModuleVersion"] != "PotentialStepController-Week8":
        raise ValueError("unexpected potential step-controller module version")
    if pyDataGCP["lastPotentialStepControllerType"] != "CCDLineSearch":
        raise ValueError("GCP must use CCD line search in step-controller split regression")
    if pyDataOGC["lastPotentialStepControllerType"] != "TrustRegion":
        raise ValueError("OGC must use trust region in step-controller split regression")
    if pyDataGCP["lastPotentialCCDNumberOfEvaluations"] <= 0:
        raise ValueError("expected GCP CCD line search to evaluate the feasible-step filter")
    if pyDataGCP["lastPotentialCCDAlpha"] <= 0.0 or pyDataGCP["lastPotentialCCDAlpha"] > 1.0:
        raise ValueError("GCP CCD line search returned an invalid step length")
    if pyDataOGC["totalPotentialCCDClippedSteps"] != 0:
        raise ValueError("OGC trust-region path should not report legacy CCD clipping in this regression")
    if pyDataOGC["lastPotentialTrustRegionRadius"] <= 0.0:
        raise ValueError("expected a positive trust-region radius for OGC")
    if not np.isfinite(pyDataGCP["lastPotentialCCDMinimumDistance"]) or not np.isfinite(pyDataOGC["lastPotentialCCDMinimumDistance"]):
        raise ValueError("step-controller split regression returned invalid minimum-distance statistics")
    if pyDataOGC["lastPotentialCCDMinimumDistance"] <= 0.0:
        raise ValueError("expected positive minimum distance after OGC trust-region filtering")
    if not np.isfinite(coordinatesGCP[2]) or not np.isfinite(coordinatesOGC[2]):
        raise ValueError("step-controller split regression returned invalid ODE2 coordinates")

    return {
        "gcp": {"pyData": pyDataGCP, "coordinates": coordinatesGCP},
        "ogc": {"pyData": pyDataOGC, "coordinates": coordinatesOGC},
    }


def main():
    results = run_regression()
    pyDataGCP = results["gcp"]["pyData"]
    pyDataOGC = results["ogc"]["pyData"]
    coordinatesGCP = results["gcp"]["coordinates"]
    coordinatesOGC = results["ogc"]["coordinates"]

    testValue = (
        float(pyDataGCP["totalPotentialCCDClippedSteps"])
        + float(pyDataOGC["lastPotentialTrustRegionRadius"])
        + float(pyDataOGC["lastPotentialCCDMinimumDistance"])
        + float(abs(coordinatesGCP[2]))
        + float(abs(coordinatesOGC[2]))
    )

    exu.Print("generalContactPotentialStepControllerSplit=", testValue)


if __name__ == "__main__":
    main()
