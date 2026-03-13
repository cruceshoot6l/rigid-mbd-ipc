/** ***********************************************************************************************
* @brief        Surface-mesh data and kinematics utilities for potential contact
* @details      Stage-A split-out of rigid surface mesh handling, including edge topology
*
************************************************************************************************ */
#ifndef POTENTIALSURFACEMESH__H
#define POTENTIALSURFACEMESH__H

#include "System/PotentialContactCommon.h"

namespace PotentialContact
{
    struct PotentialMeshVertexLocal
    {
        Vector3D xLocal;

        PotentialMeshVertexLocal()
        {
            xLocal.SetAll(0.);
        }
    };

    struct PotentialMeshEdge
    {
        Index2 vertices;

        PotentialMeshEdge()
        {
            vertices = Index2({ 0, 0 });
        }
    };

    struct PotentialMeshTriangle
    {
        Index3 vertices;

        PotentialMeshTriangle()
        {
            vertices = Index3({ 0,0,0 });
        }
    };

    struct PotentialSurfacePointKinematics
    {
        Vector3D position;
        Vector3D velocity;
        ResizableMatrix positionJacobian;

        PotentialSurfacePointKinematics()
        {
            Reset(false);
        }

        void Reset(bool freeMemory)
        {
            position.SetAll(0.);
            velocity.SetAll(0.);
            if (freeMemory) { positionJacobian.Flush(); }
            else { positionJacobian.SetNumberOfRowsAndColumns(0, 0); }
        }
    };

    struct PotentialSurfaceEdgeKinematics
    {
        Vector3D point0;
        Vector3D point1;
        Vector3D velocity0;
        Vector3D velocity1;
        ResizableMatrix point0Jacobian;
        ResizableMatrix point1Jacobian;

        PotentialSurfaceEdgeKinematics()
        {
            Reset(false);
        }

        void Reset(bool freeMemory)
        {
            point0.SetAll(0.);
            point1.SetAll(0.);
            velocity0.SetAll(0.);
            velocity1.SetAll(0.);
            if (freeMemory)
            {
                point0Jacobian.Flush();
                point1Jacobian.Flush();
            }
            else
            {
                point0Jacobian.SetNumberOfRowsAndColumns(0, 0);
                point1Jacobian.SetNumberOfRowsAndColumns(0, 0);
            }
        }
    };

    struct PotentialRigidMeshState
    {
        Vector3D markerPosition;
        Matrix3D markerOrientation;
        Vector3D markerVelocity;
        Vector3D markerAngularVelocityLocal;
        ResizableMatrix markerPositionJacobian;
        ResizableMatrix markerRotationJacobian;
        ArrayIndex markerLTG;

        PotentialRigidMeshState()
        {
            Reset(false);
        }

        void Reset(bool freeMemory)
        {
            markerPosition.SetAll(0.);
            markerOrientation = EXUmath::unitMatrix3D;
            markerVelocity.SetAll(0.);
            markerAngularVelocityLocal.SetAll(0.);
            if (freeMemory)
            {
                markerPositionJacobian.Flush();
                markerRotationJacobian.Flush();
                markerLTG.Flush();
            }
            else
            {
                markerPositionJacobian.SetNumberOfRowsAndColumns(0, 0);
                markerRotationJacobian.SetNumberOfRowsAndColumns(0, 0);
                markerLTG.SetNumberOfItems(0);
            }
        }
    };

    struct PotentialRigidMesh
    {
        Index markerIndex;
        Index frictionMaterialIndex;
        bool staticMesh;
        ResizableArray<PotentialMeshVertexLocal> verticesLocal;
        ResizableArray<PotentialMeshEdge> edges;
        ResizableArray<PotentialMeshTriangle> triangles;
        ResizableArray<PotentialSurfacePointKinematics> vertexKinematics;
        ResizableArray<PotentialSurfaceEdgeKinematics> edgeKinematics;
        ResizableArray<Box3D> edgeAABBs;
        ResizableArray<Box3D> triangleAABBs;
        PotentialRigidMeshState state;

        PotentialRigidMesh()
        {
            Reset(false);
        }

        void Reset(bool freeMemory)
        {
            markerIndex = EXUstd::InvalidIndex;
            frictionMaterialIndex = 0;
            staticMesh = false;
            if (freeMemory)
            {
                verticesLocal.Flush();
                edges.Flush();
                triangles.Flush();
                vertexKinematics.Flush();
                edgeKinematics.Flush();
                edgeAABBs.Flush();
                triangleAABBs.Flush();
            }
            else
            {
                verticesLocal.SetNumberOfItems(0);
                edges.SetNumberOfItems(0);
                triangles.SetNumberOfItems(0);
                vertexKinematics.SetNumberOfItems(0);
                edgeKinematics.SetNumberOfItems(0);
                edgeAABBs.SetNumberOfItems(0);
                triangleAABBs.SetNumberOfItems(0);
            }
            state.Reset(freeMemory);
        }

        Index NumberOfVertices() const { return verticesLocal.NumberOfItems(); }
        Index NumberOfEdges() const { return edges.NumberOfItems(); }
        Index NumberOfTriangles() const { return triangles.NumberOfItems(); }
    };

    Vector3D ComputeVertexPosition(const PotentialRigidMeshState& state, const Vector3D& localPosition);
    Vector3D ComputeVertexVelocity(const PotentialRigidMeshState& state, const Vector3D& localPosition);
    void ComputeVertexPositionJacobian(const PotentialRigidMeshState& state, const Vector3D& localPosition, ResizableMatrix& positionJacobian);
    void UpdateTriangleAABBs(PotentialRigidMesh& mesh);
    Box3D ExpandedBox(const Box3D& box, Real margin);
    void BuildUniqueEdges(PotentialRigidMesh& mesh);
    void UpdateEdgeKinematics(PotentialRigidMesh& mesh, bool computeJacobians);
    void UpdateEdgeAABBs(PotentialRigidMesh& mesh);

    class PotentialSurfaceMeshRegistry
    {
    public:
        static void FinalizeTopology(PotentialRigidMesh& mesh);
        static void UpdateDerivedKinematics(PotentialRigidMesh& mesh, bool computeJacobians);
    };
}

#endif
