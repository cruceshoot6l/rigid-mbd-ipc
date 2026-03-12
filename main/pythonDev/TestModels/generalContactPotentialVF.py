#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is an EXUDYN example
#
# Details:  Week-3 regression for minimal vertex-face potential contact residual
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
gContact.contactFormulation = exu.ContactFormulation.GCPBarrier
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
simulationSettings.solutionSettings.solutionInformation = "generalContact potential VF regression"

mbs.SolveDynamic(simulationSettings=simulationSettings, solverType=exu.DynamicSolverType.ExplicitEuler)

pyData = gContact.GetPythonObject()
contactForces = np.array(gContact.GetSystemODE2RhsContactForces(copy=True), dtype=float)
ode2_t = np.array(mbs.systemData.GetODE2Coordinates_t(), dtype=float)

if pyData["potentialContactModuleVersion"] != "PotentialContact-Week3":
    raise ValueError("unexpected potential contact module version")
if pyData["lastPotentialContactCandidates"] <= 0:
    raise ValueError("no potential vertex-face candidates detected")
if not np.isfinite(pyData["lastPotentialContactMinimumDistance"]):
    raise ValueError("potential contact minimum distance not updated")
if pyData["lastPotentialContactMinimumDistance"] >= gContact.barrierActivationDistance:
    raise ValueError("potential contact minimum distance is outside activation zone")
if contactForces.shape[0] < 3 or contactForces[2] <= 0.0:
    raise ValueError("expected positive z contact force from barrier residual")
if ode2_t[2] <= 0.0:
    raise ValueError("expected positive z velocity after explicit barrier step")

testValue = (
    float(pyData["lastPotentialContactCandidates"])
    + float(pyData["lastPotentialContactMinimumDistance"])
    + float(contactForces[2])
    + float(ode2_t[2])
)

exu.Print("generalContactPotentialVF=", testValue)
