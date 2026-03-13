#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is an EXUDYN example
#
# Details:  Week-5 regression for nonlinear CCD style feasible-step filtering
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


def run_case(use_step_filter):
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
    gContact.contactFormulation = exu.ContactFormulation.GCPBarrier
    gContact.computeContactForces = True
    gContact.barrierActivationDistance = 0.08
    gContact.barrierStiffness = 500.0
    gContact.barrierMinimumDistance = 1e-6
    gContact.useGaussNewtonHessian = True
    gContact.useNonlinearCCDStepFilter = use_step_filter
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
    simulationSettings.solutionSettings.solutionInformation = "generalContact potential CCD step filter regression"
    simulationSettings.staticSolver.verboseMode = 0
    simulationSettings.staticSolver.numberOfLoadSteps = 1
    simulationSettings.staticSolver.newton.relativeTolerance = 1e-10
    simulationSettings.staticSolver.newton.absoluteTolerance = 1e-8
    simulationSettings.staticSolver.newton.maxIterations = 20

    mbs.SolveStatic(simulationSettings)

    return gContact.GetPythonObject(), np.array(mbs.systemData.GetODE2Coordinates(), dtype=float)


def run_regression():
    pyDataOn, coordinatesOn = run_case(True)
    pyDataOff, coordinatesOff = run_case(False)

    if pyDataOn["potentialCCDModuleVersion"] != "PotentialCCD-Week8":
        raise ValueError("unexpected potential CCD module version")
    if pyDataOn["lastPotentialContactCandidates"] <= 0:
        raise ValueError("expected active potential vertex-face candidates in CCD regression")
    if pyDataOn["lastPotentialCCDNumberOfEvaluations"] <= 0:
        raise ValueError("expected nonlinear CCD step filter to evaluate at least one feasible step")
    if pyDataOn["lastPotentialCCDAlpha"] <= 0.0 or pyDataOn["lastPotentialCCDAlpha"] > 1.0:
        raise ValueError("nonlinear CCD step filter returned an invalid step length")
    if pyDataOn["totalPotentialCCDStepFailures"] != 0:
        raise ValueError("unexpected nonlinear CCD step-filter failure")
    if pyDataOn["lastPotentialCCDMinimumDistance"] <= 0.0:
        raise ValueError("expected positive minimum distance after feasible-step filtering")
    if pyDataOff["totalPotentialCCDClippedSteps"] != 0:
        raise ValueError("disabled nonlinear CCD step filter must not report clipped steps")
    if not np.isfinite(coordinatesOn[2]) or not np.isfinite(coordinatesOff[2]):
        raise ValueError("static solve returned invalid ODE2 coordinates")

    return {
        "step_filter_on": {"pyData": pyDataOn, "coordinates": coordinatesOn},
        "step_filter_off": {"pyData": pyDataOff, "coordinates": coordinatesOff},
    }


def main():
    results = run_regression()
    pyDataOn = results["step_filter_on"]["pyData"]
    coordinatesOn = results["step_filter_on"]["coordinates"]
    coordinatesOff = results["step_filter_off"]["coordinates"]

    testValue = (
        float(pyDataOn["totalPotentialCCDClippedSteps"])
        + float(pyDataOn["lastPotentialCCDMinimumDistance"])
        + float(pyDataOn["lastPotentialContactMinimumDistance"])
        + float(abs(coordinatesOn[2]))
        + float(abs(coordinatesOff[2]))
    )

    exu.Print("generalContactPotentialCCDStepFilter=", testValue)


if __name__ == "__main__":
    main()
