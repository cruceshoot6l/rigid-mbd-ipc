/** ***********************************************************************************************
* @brief        Collision-set assembly helpers for potential contact
*
************************************************************************************************ */

#include "System/PotentialCollisionSet.h"

#include <cmath>

namespace PotentialContact
{
    namespace
    {
        Vector3D ComputeTriangleNormal(const PotentialRigidMesh& mesh, Index triangleIndex)
        {
            const auto& triangle = mesh.triangles[triangleIndex];
            const Vector3D& x0 = mesh.vertexKinematics[triangle.vertices[0]].position;
            const Vector3D& x1 = mesh.vertexKinematics[triangle.vertices[1]].position;
            const Vector3D& x2 = mesh.vertexKinematics[triangle.vertices[2]].position;
            Vector3D normal = (x1 - x0).CrossProduct(x2 - x0);
            Real norm = normal.GetL2Norm();
            if (norm == 0.)
            {
                normal.SetAll(0.);
                return normal;
            }
            normal *= 1. / norm;
            return normal;
        }

        Vector3D ComputeEdgeDirection(const PotentialRigidMesh& mesh, Index edgeIndex)
        {
            const auto& edge = mesh.edgeKinematics[edgeIndex];
            Vector3D direction = edge.point1 - edge.point0;
            Real norm = direction.GetL2Norm();
            if (norm == 0.)
            {
                direction.SetAll(0.);
                return direction;
            }
            direction *= 1. / norm;
            return direction;
        }

        bool IsVertexFaceFeasibleForOGC(const PotentialRigidMesh& sourceMesh, const PotentialRigidMesh& targetMesh,
            const PotentialContactCandidate& candidate, const PotentialModelSettings& modelSettings)
        {
            const Real baryTolerance = modelSettings.ogcFaceInteriorTolerance;
            for (Index i = 0; i < 3; i++)
            {
                if (candidate.barycentricCoordinates[i] <= baryTolerance)
                {
                    return false;
                }
            }

            Vector3D faceNormal = ComputeTriangleNormal(targetMesh, candidate.triangleIndex);
            if (faceNormal.GetL2NormSquared() == 0.)
            {
                return false;
            }

            Real normalAlignment = fabs(candidate.normal * faceNormal);
            if (normalAlignment < modelSettings.ogcNormalAlignmentTolerance)
            {
                return false;
            }

            const auto& sourceVertex = sourceMesh.vertexKinematics[candidate.vertexIndex];
            const auto& triangle = targetMesh.triangles[candidate.triangleIndex];
            const Vector3D& x0 = targetMesh.vertexKinematics[triangle.vertices[0]].position;
            Real signedDistance = (sourceVertex.position - x0) * faceNormal;
            Real geometricDistance = fabs(signedDistance);
            return geometricDistance <= modelSettings.activationDistance;
        }

        bool IsEdgeEdgeFeasibleForOGC(const PotentialRigidMesh& sourceMesh, const PotentialRigidMesh& targetMesh,
            const PotentialContactCandidate& candidate, const PotentialModelSettings& modelSettings)
        {
            const Real edgeTolerance = modelSettings.ogcEdgeInteriorTolerance;
            if (candidate.edgeCoordinateA <= edgeTolerance || candidate.edgeCoordinateA >= 1. - edgeTolerance ||
                candidate.edgeCoordinateB <= edgeTolerance || candidate.edgeCoordinateB >= 1. - edgeTolerance)
            {
                return false;
            }

            Vector3D edgeDirectionA = ComputeEdgeDirection(sourceMesh, candidate.edgeIndexA);
            Vector3D edgeDirectionB = ComputeEdgeDirection(targetMesh, candidate.edgeIndexB);
            Vector3D crossDirection = edgeDirectionA.CrossProduct(edgeDirectionB);
            Real crossNorm = crossDirection.GetL2Norm();
            if (crossNorm < modelSettings.ogcMinimumEdgeCrossNorm)
            {
                return false;
            }

            crossDirection *= 1. / crossNorm;
            Real normalAlignment = fabs(candidate.normal * crossDirection);
            return normalAlignment >= modelSettings.ogcNormalAlignmentTolerance;
        }

        bool IsVertexFaceTangentiallyFeasibleForOGC(const PotentialRigidMesh& sourceMesh, const PotentialRigidMesh& targetMesh,
            const PotentialContactCandidate& candidate, const PotentialModelSettings& modelSettings, Real tangentialActivationDistance)
        {
            const Real baryTolerance = 0.5 * modelSettings.ogcFaceInteriorTolerance;
            for (Index i = 0; i < 3; i++)
            {
                if (candidate.barycentricCoordinates[i] <= baryTolerance)
                {
                    return false;
                }
            }

            Vector3D faceNormal = ComputeTriangleNormal(targetMesh, candidate.triangleIndex);
            if (faceNormal.GetL2NormSquared() == 0.)
            {
                return false;
            }

            Real normalAlignment = fabs(candidate.normal * faceNormal);
            if (normalAlignment < 0.5 * modelSettings.ogcNormalAlignmentTolerance)
            {
                return false;
            }

            const auto& sourceVertex = sourceMesh.vertexKinematics[candidate.vertexIndex];
            const auto& triangle = targetMesh.triangles[candidate.triangleIndex];
            const Vector3D& x0 = targetMesh.vertexKinematics[triangle.vertices[0]].position;
            Real signedDistance = (sourceVertex.position - x0) * faceNormal;
            Real geometricDistance = fabs(signedDistance);
            return geometricDistance <= tangentialActivationDistance;
        }

        bool IsEdgeEdgeTangentiallyFeasibleForOGC(const PotentialRigidMesh& sourceMesh, const PotentialRigidMesh& targetMesh,
            const PotentialContactCandidate& candidate, const PotentialModelSettings& modelSettings)
        {
            const Real edgeTolerance = 0.5 * modelSettings.ogcEdgeInteriorTolerance;
            if (candidate.edgeCoordinateA <= edgeTolerance || candidate.edgeCoordinateA >= 1. - edgeTolerance ||
                candidate.edgeCoordinateB <= edgeTolerance || candidate.edgeCoordinateB >= 1. - edgeTolerance)
            {
                return false;
            }

            Vector3D edgeDirectionA = ComputeEdgeDirection(sourceMesh, candidate.edgeIndexA);
            Vector3D edgeDirectionB = ComputeEdgeDirection(targetMesh, candidate.edgeIndexB);
            Vector3D crossDirection = edgeDirectionA.CrossProduct(edgeDirectionB);
            Real crossNorm = crossDirection.GetL2Norm();
            if (crossNorm < 0.5 * modelSettings.ogcMinimumEdgeCrossNorm)
            {
                return false;
            }

            crossDirection *= 1. / crossNorm;
            Real normalAlignment = fabs(candidate.normal * crossDirection);
            return normalAlignment >= 0.5 * modelSettings.ogcNormalAlignmentTolerance;
        }

        void UpdateCollisionSetStatistics(PotentialCandidateType candidateType, const PotentialContactCandidate& candidate,
            PotentialContactStatistics* statistics)
        {
            if (statistics == nullptr)
            {
                return;
            }

            if (candidateType == PotentialCandidateType::VertexFace)
            {
                statistics->numberOfVertexFaceCandidates++;
            }
            else
            {
                statistics->numberOfEdgeEdgeCandidates++;
            }
            statistics->minimumDistance = EXUstd::Minimum(statistics->minimumDistance, candidate.distance);
        }

        Real GetTangentialActivationDistance(const PotentialModelSettings& modelSettings)
        {
            return EXUstd::Maximum(modelSettings.minimumDistance,
                EXUstd::Minimum(modelSettings.activationDistance, 0.5 * modelSettings.activationDistance));
        }

        bool IsTangentialCandidateEligible(const PotentialRigidMesh& sourceMesh, const PotentialRigidMesh& targetMesh,
            const PotentialContactCandidate& candidate, const PotentialModelSettings& modelSettings)
        {
            if (!modelSettings.enableFriction)
            {
                return false;
            }

            const Real tangentialActivationDistance = GetTangentialActivationDistance(modelSettings);
            if (candidate.distance > tangentialActivationDistance)
            {
                return false;
            }

            if (candidate.type == PotentialCandidateType::VertexFace)
            {
                if (candidate.barycentricCoordinates[0] <= 0. || candidate.barycentricCoordinates[1] <= 0. ||
                    candidate.barycentricCoordinates[2] <= 0.)
                {
                    return false;
                }

                if (modelSettings.modelType == PotentialModelType::OGC)
                {
                    return IsVertexFaceTangentiallyFeasibleForOGC(sourceMesh, targetMesh, candidate,
                        modelSettings, tangentialActivationDistance);
                }
                return true;
            }

            if (candidate.edgeCoordinateA <= 1e-8 || candidate.edgeCoordinateA >= 1. - 1e-8 ||
                candidate.edgeCoordinateB <= 1e-8 || candidate.edgeCoordinateB >= 1. - 1e-8)
            {
                return false;
            }
            if (modelSettings.modelType == PotentialModelType::OGC)
            {
                return IsEdgeEdgeTangentiallyFeasibleForOGC(sourceMesh, targetMesh, candidate, modelSettings);
            }
            return true;
        }

        void AppendTangentialCandidateIfEnabled(const PotentialRigidMesh& sourceMesh, const PotentialRigidMesh& targetMesh,
            const PotentialContactCandidate& candidate, const PotentialModelSettings& modelSettings,
            PotentialCollisionSet& collisionSet, PotentialContactStatistics* statistics)
        {
            if (!IsTangentialCandidateEligible(sourceMesh, targetMesh, candidate, modelSettings))
            {
                return;
            }

            collisionSet.tangentialCandidates.Append(candidate);
            if (statistics != nullptr)
            {
                statistics->numberOfTangentialCandidates++;
            }
        }
    }

    PotentialCollisionSetBuilderType PotentialCollisionSetBuilder::GetBuilderType(const PotentialModelSettings& modelSettings)
    {
        if (modelSettings.modelType == PotentialModelType::OGC)
        {
            return PotentialCollisionSetBuilderType::OGCFeasibleRegion;
        }
        return PotentialCollisionSetBuilderType::IPCCompatible;
    }

    void IPCCompatibleCollisionSetBuilder::BuildCollisionSet(const ResizableArray<PotentialRigidMesh*>& meshes, const PotentialCandidateSeeds& candidateSeeds,
        const PotentialModelSettings& modelSettings, PotentialCollisionSet& collisionSet, PotentialContactStatistics* statistics)
    {
        collisionSet.Reset();
        if (statistics != nullptr)
        {
            statistics->collisionSetBuilderType = PotentialCollisionSetBuilderType::IPCCompatible;
        }

        for (Index seedIndex = 0; seedIndex < candidateSeeds.vfSeeds.NumberOfItems(); seedIndex++)
        {
            const auto& seed = candidateSeeds.vfSeeds[seedIndex];
            PotentialContactCandidate candidate;
            if (!ComputeVertexFaceCandidate(*meshes[seed.meshV], seed.meshV, seed.vertexIndex,
                *meshes[seed.meshF], seed.meshF, seed.triangleIndex, modelSettings.activationDistance, candidate))
            {
                continue;
            }
            collisionSet.normalCandidates.Append(candidate);
            UpdateCollisionSetStatistics(PotentialCandidateType::VertexFace, candidate, statistics);
            AppendTangentialCandidateIfEnabled(*meshes[seed.meshV], *meshes[seed.meshF], candidate,
                modelSettings, collisionSet, statistics);
        }

        for (Index seedIndex = 0; seedIndex < candidateSeeds.eeSeeds.NumberOfItems(); seedIndex++)
        {
            const auto& seed = candidateSeeds.eeSeeds[seedIndex];
            PotentialContactCandidate candidate;
            if (!ComputeEdgeEdgeCandidate(*meshes[seed.meshA], seed.meshA, seed.edgeIndexA,
                *meshes[seed.meshB], seed.meshB, seed.edgeIndexB, modelSettings.activationDistance, candidate))
            {
                continue;
            }
            collisionSet.normalCandidates.Append(candidate);
            UpdateCollisionSetStatistics(PotentialCandidateType::EdgeEdge, candidate, statistics);
            AppendTangentialCandidateIfEnabled(*meshes[seed.meshA], *meshes[seed.meshB], candidate,
                modelSettings, collisionSet, statistics);
        }
    }

    void OGCCollisionSetBuilder::BuildCollisionSet(const ResizableArray<PotentialRigidMesh*>& meshes, const PotentialCandidateSeeds& candidateSeeds,
        const PotentialModelSettings& modelSettings, PotentialCollisionSet& collisionSet, PotentialContactStatistics* statistics)
    {
        collisionSet.Reset();
        if (statistics != nullptr)
        {
            statistics->collisionSetBuilderType = PotentialCollisionSetBuilderType::OGCFeasibleRegion;
        }

        for (Index seedIndex = 0; seedIndex < candidateSeeds.vfSeeds.NumberOfItems(); seedIndex++)
        {
            const auto& seed = candidateSeeds.vfSeeds[seedIndex];
            PotentialContactCandidate candidate;
            if (!ComputeVertexFaceCandidate(*meshes[seed.meshV], seed.meshV, seed.vertexIndex,
                *meshes[seed.meshF], seed.meshF, seed.triangleIndex, modelSettings.activationDistance, candidate))
            {
                continue;
            }
            AppendTangentialCandidateIfEnabled(*meshes[seed.meshV], *meshes[seed.meshF], candidate,
                modelSettings, collisionSet, statistics);
            if (!IsVertexFaceFeasibleForOGC(*meshes[seed.meshV], *meshes[seed.meshF], candidate, modelSettings))
            {
                if (statistics != nullptr) { statistics->numberOfCollisionSetRejects++; }
                continue;
            }
            collisionSet.normalCandidates.Append(candidate);
            UpdateCollisionSetStatistics(PotentialCandidateType::VertexFace, candidate, statistics);
        }

        for (Index seedIndex = 0; seedIndex < candidateSeeds.eeSeeds.NumberOfItems(); seedIndex++)
        {
            const auto& seed = candidateSeeds.eeSeeds[seedIndex];
            PotentialContactCandidate candidate;
            if (!ComputeEdgeEdgeCandidate(*meshes[seed.meshA], seed.meshA, seed.edgeIndexA,
                *meshes[seed.meshB], seed.meshB, seed.edgeIndexB, modelSettings.activationDistance, candidate))
            {
                continue;
            }
            AppendTangentialCandidateIfEnabled(*meshes[seed.meshA], *meshes[seed.meshB], candidate,
                modelSettings, collisionSet, statistics);
            if (!IsEdgeEdgeFeasibleForOGC(*meshes[seed.meshA], *meshes[seed.meshB], candidate, modelSettings))
            {
                if (statistics != nullptr) { statistics->numberOfCollisionSetRejects++; }
                continue;
            }
            collisionSet.normalCandidates.Append(candidate);
            UpdateCollisionSetStatistics(PotentialCandidateType::EdgeEdge, candidate, statistics);
        }
    }

    void PotentialCollisionSetBuilder::BuildCollisionSet(const ResizableArray<PotentialRigidMesh*>& meshes, const PotentialCandidateSeeds& candidateSeeds,
        const PotentialModelSettings& modelSettings, PotentialCollisionSet& collisionSet, PotentialContactStatistics* statistics)
    {
        if (GetBuilderType(modelSettings) == PotentialCollisionSetBuilderType::OGCFeasibleRegion)
        {
            OGCCollisionSetBuilder::BuildCollisionSet(meshes, candidateSeeds, modelSettings, collisionSet, statistics);
        }
        else
        {
            IPCCompatibleCollisionSetBuilder::BuildCollisionSet(meshes, candidateSeeds, modelSettings, collisionSet, statistics);
        }
    }
}
