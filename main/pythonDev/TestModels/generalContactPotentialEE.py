#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is an EXUDYN example
#
# Details:  Shared-foundation regression for minimal edge-edge potential contact
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np

import exudyn as exu
from exudyn.itemInterface import *


SC = exu.SystemContainer()
mbs = SC.AddSystem()

ground = mbs.AddObject(ObjectGround())
mGround = mbs.AddMarker(MarkerBodyRigid(bodyNumber=ground, localPosition=[0.0, 0.0, 0.0]))

nRigid = mbs.AddNode(NodeRigidBodyRxyz(
    referenceCoordinates=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    initialVelocities=[0.0] * 6,
))
oRigid = mbs.AddObject(ObjectRigidBody(
    physicsMass=1.0,
    physicsInertia=[0.1, 0.1, 0.1, 0.0, 0.0, 0.0],
    nodeNumber=nRigid,
    visualization=VObjectRigidBody(graphicsData=[]),
))
mRigid = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid, localPosition=[0.0, 0.0, 0.0]))

gContact = mbs.AddGeneralContact()
gContact.contactFormulation = exu.ContactFormulation.GCPBarrier
gContact.computeContactForces = True
gContact.barrierActivationDistance = 0.08
gContact.barrierStiffness = 100.0
gContact.barrierMinimumDistance = 1e-6
gContact.useGaussNewtonHessian = True
gContact.SetSearchTreeBox([-2.0, -2.0, -2.0], [2.0, 2.0, 2.0])
gContact.SetFrictionPairings([[0.0]])

# The long base edge of triangle A and the long side edge of triangle B form a
# stable interior-interior EE candidate while avoiding VF candidates.
triangleA = [
    [-0.2, -0.02, 0.0],
    [0.2, -0.02, 0.0],
    [0.0, 0.25, 0.0],
]
triangleB = [
    [0.0, -0.05, 0.04],
    [0.0, 0.4, 0.04],
    [0.25, 0.1, 0.24],
]

gContact.AddRigidBodySurfaceMesh(mGround, triangleA, [[0, 1, 2]], frictionMaterialIndex=0, staticMesh=True)
gContact.AddRigidBodySurfaceMesh(mRigid, triangleB, [[0, 1, 2]], frictionMaterialIndex=0)

mbs.Assemble()

simulationSettings = exu.SimulationSettings()
simulationSettings.timeIntegration.numberOfSteps = 2
simulationSettings.timeIntegration.endTime = 2e-4
simulationSettings.timeIntegration.verboseMode = 0
simulationSettings.solutionSettings.writeSolutionToFile = False
simulationSettings.solutionSettings.solutionInformation = "generalContact potential EE regression"

mbs.SolveDynamic(simulationSettings=simulationSettings, solverType=exu.DynamicSolverType.ExplicitEuler)

pyData = gContact.GetPythonObject()
contactForces = np.array(gContact.GetSystemODE2RhsContactForces(copy=True), dtype=float)
ode2_t = np.array(mbs.systemData.GetODE2Coordinates_t(), dtype=float)

if pyData["potentialContactModuleVersion"] != "PotentialContact-Week4":
    raise ValueError("unexpected potential contact module version")
if pyData["lastPotentialContactBroadPhasePairs"] <= 0:
    raise ValueError("expected at least one potential broad-phase pair")
if pyData["lastPotentialContactVertexFaceCandidates"] != 0:
    raise ValueError("expected pure edge-edge regression without VF candidates")
if pyData["lastPotentialContactEdgeEdgeCandidates"] <= 0:
    raise ValueError("expected active potential edge-edge candidates")
if pyData["lastPotentialContactCandidates"] != pyData["lastPotentialContactEdgeEdgeCandidates"]:
    raise ValueError("expected all potential candidates to come from EE contact")
if not np.isfinite(pyData["lastPotentialContactMinimumDistance"]):
    raise ValueError("potential contact minimum distance not updated")
if pyData["lastPotentialContactMinimumDistance"] >= gContact.barrierActivationDistance:
    raise ValueError("edge-edge minimum distance is outside activation zone")
if contactForces.shape[0] < 3 or contactForces[2] <= 0.0:
    raise ValueError("expected positive z contact force from edge-edge barrier residual")
if ode2_t[2] <= 0.0:
    raise ValueError("expected positive z velocity after explicit edge-edge barrier step")

testValue = (
    float(pyData["lastPotentialContactEdgeEdgeCandidates"])
    + float(pyData["lastPotentialContactMinimumDistance"])
    + float(contactForces[2])
    + float(ode2_t[2])
)

exu.Print("generalContactPotentialEE=", testValue)
