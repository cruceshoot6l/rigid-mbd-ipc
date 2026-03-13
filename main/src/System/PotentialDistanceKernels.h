/** ***********************************************************************************************
* @brief        Distance kernels and barrier evaluation for potential contact
* @details      Stage-A split-out of VF and EE candidate generation
*
************************************************************************************************ */
#ifndef POTENTIALDISTANCEKERNELS__H
#define POTENTIALDISTANCEKERNELS__H

#include "System/PotentialSurfaceMesh.h"

namespace PotentialContact
{
    struct PotentialLocalVertexReference
    {
        Index meshIndex;
        Index vertexIndex;

        PotentialLocalVertexReference()
        {
            Reset();
        }

        void Reset()
        {
            meshIndex = EXUstd::InvalidIndex;
            vertexIndex = EXUstd::InvalidIndex;
        }
    };

    struct PotentialContactCandidate
    {
        PotentialCandidateType type;
        Index sourceMeshIndex;
        Index targetMeshIndex;
        Index vertexIndex;
        Index triangleIndex;
        Index edgeIndexA;
        Index edgeIndexB;
        Vector3D closestPoint;
        Vector3D closestPointA;
        Vector3D closestPointB;
        Vector3D normal;
        Vector3D barycentricCoordinates;
        Real edgeCoordinateA;
        Real edgeCoordinateB;
        Real distance;

        PotentialContactCandidate()
        {
            Reset();
        }

        void Reset()
        {
            type = PotentialCandidateType::VertexFace;
            sourceMeshIndex = EXUstd::InvalidIndex;
            targetMeshIndex = EXUstd::InvalidIndex;
            vertexIndex = EXUstd::InvalidIndex;
            triangleIndex = EXUstd::InvalidIndex;
            edgeIndexA = EXUstd::InvalidIndex;
            edgeIndexB = EXUstd::InvalidIndex;
            closestPoint.SetAll(0.);
            closestPointA.SetAll(0.);
            closestPointB.SetAll(0.);
            normal.SetAll(0.);
            barycentricCoordinates.SetAll(0.);
            edgeCoordinateA = 0.;
            edgeCoordinateB = 0.;
            distance = EXUstd::MAXREAL;
        }
    };

    struct PotentialContactEvaluation
    {
        PotentialModelType modelType;
        Real energy;
        Real derivativeWRTDistance;
        Real secondDerivativeWRTDistance;
        Real distanceClamped;
        Real effectiveActivationDistance;
        bool hasLocalDerivatives;
        bool hasLocalHessian;
        ResizableArray<PotentialLocalVertexReference> localVertexReferences;
        ResizableVector localGradient;
        ResizableMatrix localHessian;
        bool active;

        PotentialContactEvaluation()
        {
            Reset();
        }

        void Reset()
        {
            modelType = PotentialModelType::GCP;
            energy = 0.;
            derivativeWRTDistance = 0.;
            secondDerivativeWRTDistance = 0.;
            distanceClamped = 0.;
            effectiveActivationDistance = 0.;
            hasLocalDerivatives = false;
            hasLocalHessian = false;
            localVertexReferences.SetNumberOfItems(0);
            localGradient.SetNumberOfItems(0);
            localHessian.SetNumberOfRowsAndColumns(0, 0);
            active = false;
        }
    };

    class VertexFaceDistanceKernel
    {
    public:
        static bool BuildCandidate(const PotentialRigidMesh& sourceMesh, Index sourceMeshIndex, Index vertexIndex,
            const PotentialRigidMesh& targetMesh, Index targetMeshIndex, Index triangleIndex, Real activationDistance,
            PotentialContactCandidate& candidate);
    };

    class EdgeEdgeDistanceKernel
    {
    public:
        static bool BuildCandidate(const PotentialRigidMesh& sourceMesh, Index sourceMeshIndex, Index edgeIndexA,
            const PotentialRigidMesh& targetMesh, Index targetMeshIndex, Index edgeIndexB, Real activationDistance,
            PotentialContactCandidate& candidate);
    };

    bool ComputeVertexFaceCandidate(const PotentialRigidMesh& sourceMesh, Index sourceMeshIndex, Index vertexIndex,
        const PotentialRigidMesh& targetMesh, Index targetMeshIndex, Index triangleIndex, Real activationDistance,
        PotentialContactCandidate& candidate);

    bool ComputeEdgeEdgeCandidate(const PotentialRigidMesh& sourceMesh, Index sourceMeshIndex, Index edgeIndexA,
        const PotentialRigidMesh& targetMesh, Index targetMeshIndex, Index edgeIndexB, Real activationDistance,
        PotentialContactCandidate& candidate);
}

#endif
