/** ***********************************************************************************************
* @brief        Surface-mesh data and kinematics utilities for potential contact
*
************************************************************************************************ */

#include "System/PotentialSurfaceMesh.h"

#include <set>
#include <utility>

#include "Utilities/RigidBodyMath.h"

namespace PotentialContact
{
    Vector3D ComputeVertexPosition(const PotentialRigidMeshState& state, const Vector3D& localPosition)
    {
        return state.markerPosition + state.markerOrientation * localPosition;
    }

    Vector3D ComputeVertexVelocity(const PotentialRigidMeshState& state, const Vector3D& localPosition)
    {
        return state.markerVelocity + state.markerOrientation * state.markerAngularVelocityLocal.CrossProduct(localPosition);
    }

    void ComputeVertexPositionJacobian(const PotentialRigidMeshState& state, const Vector3D& localPosition, ResizableMatrix& positionJacobian)
    {
        positionJacobian.CopyFrom(state.markerPositionJacobian);
        if (state.markerRotationJacobian.NumberOfColumns() == 0)
        {
            return;
        }

        ResizableMatrix temp;
        Vector3D globalOffset = state.markerOrientation * localPosition;
        EXUmath::MultMatrixMatrixTemplate<Matrix3D, ResizableMatrix, ResizableMatrix>(
            RigidBodyMath::Vector2SkewMatrix(-globalOffset), state.markerRotationJacobian, temp);

        CHECKandTHROW(temp.NumberOfRows() == positionJacobian.NumberOfRows() &&
            temp.NumberOfColumns() == positionJacobian.NumberOfColumns(),
            "PotentialContact::ComputeVertexPositionJacobian: inconsistent jacobian dimensions");

        for (Index i = 0; i < positionJacobian.NumberOfRows(); i++)
        {
            for (Index j = 0; j < positionJacobian.NumberOfColumns(); j++)
            {
                positionJacobian(i, j) += temp(i, j);
            }
        }
    }

    void UpdateTriangleAABBs(PotentialRigidMesh& mesh)
    {
        mesh.triangleAABBs.SetNumberOfItems(mesh.triangles.NumberOfItems());
        for (Index i = 0; i < mesh.triangles.NumberOfItems(); i++)
        {
            const PotentialMeshTriangle& triangle = mesh.triangles[i];
            Box3D box;
            for (Index j = 0; j < triangle.vertices.NumberOfItems(); j++)
            {
                if (!mesh.vertexKinematics.IsValidIndex(triangle.vertices[j]))
                {
                    CHECKandTHROWstring("PotentialContact::UpdateTriangleAABBs: invalid triangle vertex index " +
                        EXUstd::ToString(triangle.vertices[j]) + " for triangle " + EXUstd::ToString(i) +
                        " with vertexKinematics size " + EXUstd::ToString(mesh.vertexKinematics.NumberOfItems()));
                }
                box.Add(mesh.vertexKinematics[triangle.vertices[j]].position);
            }
            mesh.triangleAABBs[i] = box;
        }
    }

    Box3D ExpandedBox(const Box3D& box, Real margin)
    {
        Box3D expandedBox(box);
        expandedBox.Increase(margin);
        return expandedBox;
    }

    void BuildUniqueEdges(PotentialRigidMesh& mesh)
    {
        std::set<std::pair<Index, Index>> uniqueEdges;
        for (Index triangleIndex = 0; triangleIndex < mesh.triangles.NumberOfItems(); triangleIndex++)
        {
            const auto& triangle = mesh.triangles[triangleIndex];
            for (Index localEdge = 0; localEdge < 3; localEdge++)
            {
                Index v0 = triangle.vertices[localEdge];
                Index v1 = triangle.vertices[(localEdge + 1) % 3];
                if (v1 < v0)
                {
                    std::swap(v0, v1);
                }
                uniqueEdges.insert(std::make_pair(v0, v1));
            }
        }

        mesh.edges.SetNumberOfItems((Index)uniqueEdges.size());
        Index edgeIndex = 0;
        for (const auto& edge : uniqueEdges)
        {
            mesh.edges[edgeIndex].vertices = Index2({ edge.first, edge.second });
            edgeIndex++;
        }

        mesh.edgeKinematics.SetNumberOfItems(mesh.edges.NumberOfItems());
        mesh.edgeAABBs.SetNumberOfItems(mesh.edges.NumberOfItems());
    }

    void UpdateEdgeKinematics(PotentialRigidMesh& mesh, bool computeJacobians)
    {
        mesh.edgeKinematics.SetNumberOfItems(mesh.edges.NumberOfItems());
        for (Index edgeIndex = 0; edgeIndex < mesh.edges.NumberOfItems(); edgeIndex++)
        {
            const auto& edge = mesh.edges[edgeIndex];
            auto& edgeKinematics = mesh.edgeKinematics[edgeIndex];

            CHECKandTHROW(mesh.vertexKinematics.IsValidIndex(edge.vertices[0]) &&
                mesh.vertexKinematics.IsValidIndex(edge.vertices[1]),
                "PotentialContact::UpdateEdgeKinematics: invalid edge vertex index");

            const auto& point0 = mesh.vertexKinematics[edge.vertices[0]];
            const auto& point1 = mesh.vertexKinematics[edge.vertices[1]];
            edgeKinematics.point0 = point0.position;
            edgeKinematics.point1 = point1.position;
            edgeKinematics.velocity0 = point0.velocity;
            edgeKinematics.velocity1 = point1.velocity;
            if (computeJacobians)
            {
                edgeKinematics.point0Jacobian.CopyFrom(point0.positionJacobian);
                edgeKinematics.point1Jacobian.CopyFrom(point1.positionJacobian);
            }
            else
            {
                edgeKinematics.point0Jacobian.SetNumberOfRowsAndColumns(0, 0);
                edgeKinematics.point1Jacobian.SetNumberOfRowsAndColumns(0, 0);
            }
        }
    }

    void UpdateEdgeAABBs(PotentialRigidMesh& mesh)
    {
        mesh.edgeAABBs.SetNumberOfItems(mesh.edges.NumberOfItems());
        for (Index edgeIndex = 0; edgeIndex < mesh.edges.NumberOfItems(); edgeIndex++)
        {
            Box3D box;
            box.Add(mesh.edgeKinematics[edgeIndex].point0);
            box.Add(mesh.edgeKinematics[edgeIndex].point1);
            mesh.edgeAABBs[edgeIndex] = box;
        }
    }

    void PotentialSurfaceMeshRegistry::FinalizeTopology(PotentialRigidMesh& mesh)
    {
        BuildUniqueEdges(mesh);
    }

    void PotentialSurfaceMeshRegistry::UpdateDerivedKinematics(PotentialRigidMesh& mesh, bool computeJacobians)
    {
        UpdateEdgeKinematics(mesh, computeJacobians);
        UpdateEdgeAABBs(mesh);
        UpdateTriangleAABBs(mesh);
    }
}
