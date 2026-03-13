#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is an EXUDYN example
#
# Details:  Shared-foundation regression for minimal edge-edge potential contact tangent
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

solver = exu.MainSolverStatic()
simulationSettings = exu.SimulationSettings()
simulationSettings.solutionSettings.writeSolutionToFile = False
solver.InitializeSolver(mbs, simulationSettings)

solver.ComputeODE2RHS(mbs)
residual0 = np.array(solver.GetSystemResidual(), dtype=float)
pyData = gContact.GetPythonObject()

solver.ComputeJacobianODE2RHS(mbs, scalarFactor_ODE2=1.0, scalarFactor_ODE2_t=0.0, scalarFactor_ODE1=0.0)
jacobian = np.array(solver.GetSystemJacobian(), dtype=float)

coordinates0 = np.array(mbs.systemData.GetODE2Coordinates(), dtype=float)
coordinates_t0 = np.array(mbs.systemData.GetODE2Coordinates_t(), dtype=float)
eps = 1e-7

coordinates1 = coordinates0.copy()
coordinates1[2] += eps
mbs.systemData.SetODE2Coordinates(coordinates=coordinates1)
mbs.systemData.SetODE2Coordinates_t(coordinates=coordinates_t0)

solver.ComputeODE2RHS(mbs)
residual1 = np.array(solver.GetSystemResidual(), dtype=float)

mbs.systemData.SetODE2Coordinates(coordinates=coordinates0)
mbs.systemData.SetODE2Coordinates_t(coordinates=coordinates_t0)
solver.FinalizeSolver(mbs, simulationSettings)

finite_difference = (residual1[2] - residual0[2]) / eps
analytic = jacobian[2, 2]
relative_error = abs(analytic - finite_difference) / max(1.0, abs(finite_difference))

if pyData["potentialContactModuleVersion"] != "PotentialContact-Week4":
    raise ValueError("unexpected potential contact module version")
if pyData["lastPotentialContactBroadPhasePairs"] <= 0:
    raise ValueError("expected at least one potential broad-phase pair before tangent evaluation")
if pyData["lastPotentialContactVertexFaceCandidates"] != 0:
    raise ValueError("expected pure edge-edge tangent regression without VF candidates")
if pyData["lastPotentialContactEdgeEdgeCandidates"] <= 0:
    raise ValueError("expected active potential edge-edge candidates before tangent evaluation")
if analytic >= 0.0:
    raise ValueError("expected negative z-z tangent entry for repulsive edge-edge barrier contact")
if relative_error > 5e-2:
    raise ValueError(f"edge-edge potential contact tangent mismatch: analytic={analytic}, fd={finite_difference}")

testValue = float(abs(analytic)) + float(relative_error) + float(pyData["lastPotentialContactEdgeEdgeCandidates"])

exu.Print("generalContactPotentialEEJacobian=", testValue)
