#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is an EXUDYN example
#
# Details:  Week-2 regression for rigid surface mesh registration and kinematics
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np

import exudyn as exu
from exudyn.itemInterface import *
from exudyn.utilities import AngularVelocity2EulerParameters_t, eulerParameters0


def cube_mesh(edge_length=0.4):
    h = 0.5 * edge_length
    points = [
        [-h, -h, -h],
        [h, -h, -h],
        [h, h, -h],
        [-h, h, -h],
        [-h, -h, h],
        [h, -h, h],
        [h, h, h],
        [-h, h, h],
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

reference_position = [0.2, -0.1, 0.3]
initial_velocity = [0.4, -0.25, 0.15]
initial_angular_velocity = [0.35, -0.2, 0.3]
ep0 = list(eulerParameters0)
ep_t0 = list(AngularVelocity2EulerParameters_t(initial_angular_velocity, ep0))

nRB = mbs.AddNode(NodeRigidBodyEP(
    referenceCoordinates=reference_position + ep0,
    initialVelocities=initial_velocity + ep_t0,
))
oRB = mbs.AddObject(ObjectRigidBody(
    physicsMass=2.0,
    physicsInertia=[1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    nodeNumber=nRB,
    visualization=VObjectRigidBody(graphicsData=[]),
))
mRigid = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRB, localPosition=[0.0, 0.0, 0.0]))

gContact = mbs.AddGeneralContact()
gContact.contactFormulation = exu.ContactFormulation.GCPBarrier
gContact.SetSearchTreeBox([-2.0, -2.0, -2.0], [2.0, 2.0, 2.0])
gContact.SetFrictionPairings([[0.0]])

point_list, triangle_list = cube_mesh()
mesh_index = gContact.AddRigidBodySurfaceMesh(mRigid, point_list, triangle_list, frictionMaterialIndex=0)

mbs.Assemble()
gContact.UpdateRigidBodySurfaceMeshes(mbs, True)

mesh0 = gContact.GetRigidBodySurfaceMesh(mesh_index, includeJacobians=True)

vertices0 = np.array(mesh0["verticesWorld"], dtype=float)
velocities0 = np.array(mesh0["vertexVelocities"], dtype=float)
vertex_jacobian0 = np.array(mesh0["vertexJacobians"][0], dtype=float)
ode2_t0 = np.array(mbs.systemData.GetODE2Coordinates_t(), dtype=float)

if mesh0["numberOfVertices"] != 8 or mesh0["numberOfTriangles"] != 12:
    raise ValueError("unexpected rigid surface mesh size")
if vertex_jacobian0.shape != (3, len(ode2_t0)):
    raise ValueError("unexpected vertex jacobian size")
if np.linalg.norm(vertex_jacobian0 @ ode2_t0 - velocities0[0]) > 1e-12:
    raise ValueError("vertex jacobian does not reproduce surface velocity")

simulationSettings = exu.SimulationSettings()
simulationSettings.timeIntegration.numberOfSteps = 1
simulationSettings.timeIntegration.endTime = 1e-3
simulationSettings.timeIntegration.verboseMode = 0
simulationSettings.solutionSettings.writeSolutionToFile = False
simulationSettings.solutionSettings.solutionInformation = "generalContact rigid surface mesh regression"

mbs.SolveDynamic(simulationSettings=simulationSettings)

gContact.UpdateRigidBodySurfaceMeshes(mbs, True)
mesh1 = gContact.GetRigidBodySurfaceMesh(mesh_index, includeJacobians=True)
vertices1 = np.array(mesh1["verticesWorld"], dtype=float)
velocities1 = np.array(mesh1["vertexVelocities"], dtype=float)
vertex_jacobian1 = np.array(mesh1["vertexJacobians"][0], dtype=float)
ode2_t1 = np.array(mbs.systemData.GetODE2Coordinates_t(), dtype=float)

if np.linalg.norm(vertex_jacobian1 @ ode2_t1 - velocities1[0]) > 1e-10:
    raise ValueError("updated vertex jacobian does not reproduce surface velocity")
if np.linalg.norm(vertices1[0] - vertices0[0]) <= 1e-8:
    raise ValueError("surface mesh vertex did not move during simulation")

dt = simulationSettings.timeIntegration.endTime
prediction_error = np.linalg.norm(vertices1[0] - (vertices0[0] + velocities0[0] * dt))
if prediction_error > 5e-4:
    raise ValueError("surface mesh update deviates from first-order prediction")

test_value = (
    float(np.linalg.norm(vertices1[0]))
    + float(np.linalg.norm(velocities1[0]))
    + float(np.linalg.norm(vertex_jacobian1))
    + float(prediction_error)
)

exu.Print("generalContactRigidBodySurfaceMesh=", test_value)
