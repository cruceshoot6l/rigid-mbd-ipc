/** ***********************************************************************************************
* @brief        Broad-phase helpers for potential contact candidate generation
*
************************************************************************************************ */

#include "System/PotentialBroadPhase.h"
#include "System/PotentialDistanceKernels.h"

#include <cmath>

namespace PotentialContact
{
    namespace
    {
        Box3D ComputeMeshBox(const PotentialRigidMesh& mesh)
        {
            Box3D box;
            for (Index i = 0; i < mesh.vertexKinematics.NumberOfItems(); i++)
            {
                box.Add(mesh.vertexKinematics[i].position);
            }
            return box;
        }

        bool BoxesOverlap(const Box3D& boxA, const Box3D& boxB)
        {
            for (Index i = 0; i < 3; i++)
            {
                if (boxA.PMax()[i] < boxB.PMin()[i] || boxB.PMax()[i] < boxA.PMin()[i])
                {
                    return false;
                }
            }
            return true;
        }

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

        bool IsVertexFaceSeedFeasibleForOGC(const PotentialRigidMesh& targetMesh, const PotentialContactCandidate& candidate,
            const PotentialModelSettings& modelSettings)
        {
            Vector3D faceNormal = ComputeTriangleNormal(targetMesh, candidate.triangleIndex);
            if (faceNormal.GetL2NormSquared() == 0.)
            {
                return false;
            }

            const Real alignmentTolerance = 0.5 * modelSettings.ogcNormalAlignmentTolerance;
            Real normalAlignment = fabs(candidate.normal * faceNormal);
            return normalAlignment >= alignmentTolerance;
        }

        bool IsEdgeEdgeSeedFeasibleForOGC(const PotentialRigidMesh& sourceMesh, const PotentialRigidMesh& targetMesh,
            const PotentialContactCandidate& candidate, const PotentialModelSettings& modelSettings)
        {
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

        void UpdateMeshPairStatistics(PotentialMeshPairBuilderType builderType, Index coarsePairs,
            Index rejectedPairs, Index keptPairs, PotentialContactStatistics* statistics)
        {
            if (statistics == nullptr)
            {
                return;
            }
            statistics->meshPairBuilderType = builderType;
            statistics->numberOfCoarseBroadPhasePairs = coarsePairs;
            statistics->numberOfBroadPhasePairRejects = rejectedPairs;
            statistics->numberOfBroadPhasePairs = keptPairs;
        }

        void UpdateSeedStatisticsVF(PotentialContactStatistics* statistics)
        {
            if (statistics != nullptr)
            {
                statistics->numberOfVertexFaceSeeds++;
            }
        }

        void UpdateSeedStatisticsEE(PotentialContactStatistics* statistics)
        {
            if (statistics != nullptr)
            {
                statistics->numberOfEdgeEdgeSeeds++;
            }
        }

        void BuildCoarseMeshPairs(const ResizableArray<PotentialRigidMesh*>& meshes, Real activationDistance,
            ResizableArray<PotentialBroadPhasePair>& coarsePairs)
        {
            coarsePairs.SetNumberOfItems(0);
            for (Index meshA = 0; meshA < meshes.NumberOfItems(); meshA++)
            {
                Box3D boxA = ExpandedBox(ComputeMeshBox(*meshes[meshA]), activationDistance);
                for (Index meshB = meshA + 1; meshB < meshes.NumberOfItems(); meshB++)
                {
                    Box3D boxB = ExpandedBox(ComputeMeshBox(*meshes[meshB]), activationDistance);
                    if (!BoxesOverlap(boxA, boxB))
                    {
                        continue;
                    }

                    PotentialBroadPhasePair pair;
                    pair.meshA = meshA;
                    pair.meshB = meshB;
                    pair.overlapBox = boxA;
                    coarsePairs.Append(pair);
                }
            }
        }

        bool HasAnyOGCFeasibleCandidateForPair(const PotentialRigidMesh& meshA, Index meshAIndex,
            const PotentialRigidMesh& meshB, Index meshBIndex, const PotentialModelSettings& modelSettings)
        {
            PotentialContactCandidate candidate;

            for (Index vertexIndex = 0; vertexIndex < meshA.vertexKinematics.NumberOfItems(); vertexIndex++)
            {
                for (Index triangleIndex = 0; triangleIndex < meshB.triangleAABBs.NumberOfItems(); triangleIndex++)
                {
                    if (!ExpandedBox(meshB.triangleAABBs[triangleIndex], modelSettings.activationDistance).IsInside(
                        meshA.vertexKinematics[vertexIndex].position))
                    {
                        continue;
                    }
                    if (!ComputeVertexFaceCandidate(meshA, meshAIndex, vertexIndex, meshB, meshBIndex, triangleIndex,
                        modelSettings.activationDistance, candidate))
                    {
                        continue;
                    }
                    if (IsVertexFaceSeedFeasibleForOGC(meshB, candidate, modelSettings))
                    {
                        return true;
                    }
                }
            }

            for (Index vertexIndex = 0; vertexIndex < meshB.vertexKinematics.NumberOfItems(); vertexIndex++)
            {
                for (Index triangleIndex = 0; triangleIndex < meshA.triangleAABBs.NumberOfItems(); triangleIndex++)
                {
                    if (!ExpandedBox(meshA.triangleAABBs[triangleIndex], modelSettings.activationDistance).IsInside(
                        meshB.vertexKinematics[vertexIndex].position))
                    {
                        continue;
                    }
                    if (!ComputeVertexFaceCandidate(meshB, meshBIndex, vertexIndex, meshA, meshAIndex, triangleIndex,
                        modelSettings.activationDistance, candidate))
                    {
                        continue;
                    }
                    if (IsVertexFaceSeedFeasibleForOGC(meshA, candidate, modelSettings))
                    {
                        return true;
                    }
                }
            }

            for (Index edgeIndexA = 0; edgeIndexA < meshA.edgeAABBs.NumberOfItems(); edgeIndexA++)
            {
                Box3D expandedA = ExpandedBox(meshA.edgeAABBs[edgeIndexA], modelSettings.activationDistance);
                for (Index edgeIndexB = 0; edgeIndexB < meshB.edgeAABBs.NumberOfItems(); edgeIndexB++)
                {
                    Box3D expandedB = ExpandedBox(meshB.edgeAABBs[edgeIndexB], modelSettings.activationDistance);
                    if (!BoxesOverlap(expandedA, expandedB))
                    {
                        continue;
                    }
                    if (!ComputeEdgeEdgeCandidate(meshA, meshAIndex, edgeIndexA, meshB, meshBIndex, edgeIndexB,
                        modelSettings.activationDistance, candidate))
                    {
                        continue;
                    }
                    if (IsEdgeEdgeSeedFeasibleForOGC(meshA, meshB, candidate, modelSettings))
                    {
                        return true;
                    }
                }
            }
            return false;
        }
    }

    PotentialMeshPairBuilderType PotentialBroadPhaseBuilder::GetMeshPairBuilderType(const PotentialModelSettings& modelSettings)
    {
        if (modelSettings.modelType == PotentialModelType::OGC)
        {
            return PotentialMeshPairBuilderType::OGCFeasibleRegion;
        }
        return PotentialMeshPairBuilderType::IPCCompatible;
    }

    PotentialSeedBuilderType PotentialBroadPhaseBuilder::GetSeedBuilderType(const PotentialModelSettings& modelSettings)
    {
        if (modelSettings.modelType == PotentialModelType::OGC)
        {
            return PotentialSeedBuilderType::OGCFeasibleRegion;
        }
        return PotentialSeedBuilderType::IPCCompatible;
    }

    void IPCCompatibleMeshPairBuilder::BuildMeshPairs(const ResizableArray<PotentialRigidMesh*>& meshes,
        const PotentialModelSettings& modelSettings, ResizableArray<PotentialBroadPhasePair>& meshPairs,
        PotentialContactStatistics* statistics)
    {
        BuildCoarseMeshPairs(meshes, modelSettings.activationDistance, meshPairs);
        UpdateMeshPairStatistics(PotentialMeshPairBuilderType::IPCCompatible,
            meshPairs.NumberOfItems(), 0, meshPairs.NumberOfItems(), statistics);
    }

    void OGCMeshPairBuilder::BuildMeshPairs(const ResizableArray<PotentialRigidMesh*>& meshes,
        const PotentialModelSettings& modelSettings, ResizableArray<PotentialBroadPhasePair>& meshPairs,
        PotentialContactStatistics* statistics)
    {
        ResizableArray<PotentialBroadPhasePair> coarsePairs;
        BuildCoarseMeshPairs(meshes, modelSettings.activationDistance, coarsePairs);

        meshPairs.SetNumberOfItems(0);
        Index rejectedPairs = 0;
        for (Index pairIndex = 0; pairIndex < coarsePairs.NumberOfItems(); pairIndex++)
        {
            const auto& pair = coarsePairs[pairIndex];
            const auto& meshA = *meshes[pair.meshA];
            const auto& meshB = *meshes[pair.meshB];
            if (!HasAnyOGCFeasibleCandidateForPair(meshA, pair.meshA, meshB, pair.meshB, modelSettings))
            {
                rejectedPairs++;
                continue;
            }
            meshPairs.Append(pair);
        }

        UpdateMeshPairStatistics(PotentialMeshPairBuilderType::OGCFeasibleRegion,
            coarsePairs.NumberOfItems(), rejectedPairs, meshPairs.NumberOfItems(), statistics);
    }

    void PotentialBroadPhaseBuilder::BuildMeshPairs(const ResizableArray<PotentialRigidMesh*>& meshes,
        const PotentialModelSettings& modelSettings, ResizableArray<PotentialBroadPhasePair>& meshPairs,
        PotentialContactStatistics* statistics)
    {
        if (GetMeshPairBuilderType(modelSettings) == PotentialMeshPairBuilderType::OGCFeasibleRegion)
        {
            OGCMeshPairBuilder::BuildMeshPairs(meshes, modelSettings, meshPairs, statistics);
        }
        else
        {
            IPCCompatibleMeshPairBuilder::BuildMeshPairs(meshes, modelSettings, meshPairs, statistics);
        }
    }

    void IPCCompatibleSeedBuilder::BuildCandidateSeeds(const ResizableArray<PotentialRigidMesh*>& meshes,
        const ResizableArray<PotentialBroadPhasePair>& meshPairs, const PotentialModelSettings& modelSettings,
        PotentialCandidateSeeds& candidateSeeds, PotentialContactStatistics* statistics)
    {
        candidateSeeds.Reset();
        if (statistics != nullptr)
        {
            statistics->seedBuilderType = PotentialSeedBuilderType::IPCCompatible;
        }

        for (Index pairIndex = 0; pairIndex < meshPairs.NumberOfItems(); pairIndex++)
        {
            const auto& pair = meshPairs[pairIndex];
            const auto& meshA = *meshes[pair.meshA];
            const auto& meshB = *meshes[pair.meshB];

            for (Index vertexIndex = 0; vertexIndex < meshA.vertexKinematics.NumberOfItems(); vertexIndex++)
            {
                for (Index triangleIndex = 0; triangleIndex < meshB.triangleAABBs.NumberOfItems(); triangleIndex++)
                {
                    if (ExpandedBox(meshB.triangleAABBs[triangleIndex], modelSettings.activationDistance).IsInside(
                        meshA.vertexKinematics[vertexIndex].position))
                    {
                        candidateSeeds.vfSeeds.Append(PotentialCandidateSeedVF{ pair.meshA, vertexIndex, pair.meshB, triangleIndex });
                        UpdateSeedStatisticsVF(statistics);
                    }
                }
            }

            for (Index vertexIndex = 0; vertexIndex < meshB.vertexKinematics.NumberOfItems(); vertexIndex++)
            {
                for (Index triangleIndex = 0; triangleIndex < meshA.triangleAABBs.NumberOfItems(); triangleIndex++)
                {
                    if (ExpandedBox(meshA.triangleAABBs[triangleIndex], modelSettings.activationDistance).IsInside(
                        meshB.vertexKinematics[vertexIndex].position))
                    {
                        candidateSeeds.vfSeeds.Append(PotentialCandidateSeedVF{ pair.meshB, vertexIndex, pair.meshA, triangleIndex });
                        UpdateSeedStatisticsVF(statistics);
                    }
                }
            }

            for (Index edgeIndexA = 0; edgeIndexA < meshA.edgeAABBs.NumberOfItems(); edgeIndexA++)
            {
                Box3D expandedA = ExpandedBox(meshA.edgeAABBs[edgeIndexA], modelSettings.activationDistance);
                for (Index edgeIndexB = 0; edgeIndexB < meshB.edgeAABBs.NumberOfItems(); edgeIndexB++)
                {
                    Box3D expandedB = ExpandedBox(meshB.edgeAABBs[edgeIndexB], modelSettings.activationDistance);
                    if (BoxesOverlap(expandedA, expandedB))
                    {
                        candidateSeeds.eeSeeds.Append(PotentialCandidateSeedEE{ pair.meshA, edgeIndexA, pair.meshB, edgeIndexB });
                        UpdateSeedStatisticsEE(statistics);
                    }
                }
            }
        }
    }

    void OGCSeedBuilder::BuildCandidateSeeds(const ResizableArray<PotentialRigidMesh*>& meshes,
        const ResizableArray<PotentialBroadPhasePair>& meshPairs, const PotentialModelSettings& modelSettings,
        PotentialCandidateSeeds& candidateSeeds, PotentialContactStatistics* statistics)
    {
        candidateSeeds.Reset();
        if (statistics != nullptr)
        {
            statistics->seedBuilderType = PotentialSeedBuilderType::OGCFeasibleRegion;
        }

        for (Index pairIndex = 0; pairIndex < meshPairs.NumberOfItems(); pairIndex++)
        {
            const auto& pair = meshPairs[pairIndex];
            const auto& meshA = *meshes[pair.meshA];
            const auto& meshB = *meshes[pair.meshB];

            for (Index vertexIndex = 0; vertexIndex < meshA.vertexKinematics.NumberOfItems(); vertexIndex++)
            {
                for (Index triangleIndex = 0; triangleIndex < meshB.triangleAABBs.NumberOfItems(); triangleIndex++)
                {
                    if (!ExpandedBox(meshB.triangleAABBs[triangleIndex], modelSettings.activationDistance).IsInside(
                        meshA.vertexKinematics[vertexIndex].position))
                    {
                        continue;
                    }

                    PotentialContactCandidate candidate;
                    if (!ComputeVertexFaceCandidate(meshA, pair.meshA, vertexIndex, meshB, pair.meshB, triangleIndex,
                        modelSettings.activationDistance, candidate))
                    {
                        if (statistics != nullptr) { statistics->numberOfSeedRejects++; }
                        continue;
                    }
                    if (!IsVertexFaceSeedFeasibleForOGC(meshB, candidate, modelSettings))
                    {
                        if (statistics != nullptr) { statistics->numberOfSeedRejects++; }
                        continue;
                    }

                    candidateSeeds.vfSeeds.Append(PotentialCandidateSeedVF{ pair.meshA, vertexIndex, pair.meshB, triangleIndex });
                    UpdateSeedStatisticsVF(statistics);
                }
            }

            for (Index vertexIndex = 0; vertexIndex < meshB.vertexKinematics.NumberOfItems(); vertexIndex++)
            {
                for (Index triangleIndex = 0; triangleIndex < meshA.triangleAABBs.NumberOfItems(); triangleIndex++)
                {
                    if (!ExpandedBox(meshA.triangleAABBs[triangleIndex], modelSettings.activationDistance).IsInside(
                        meshB.vertexKinematics[vertexIndex].position))
                    {
                        continue;
                    }

                    PotentialContactCandidate candidate;
                    if (!ComputeVertexFaceCandidate(meshB, pair.meshB, vertexIndex, meshA, pair.meshA, triangleIndex,
                        modelSettings.activationDistance, candidate))
                    {
                        if (statistics != nullptr) { statistics->numberOfSeedRejects++; }
                        continue;
                    }
                    if (!IsVertexFaceSeedFeasibleForOGC(meshA, candidate, modelSettings))
                    {
                        if (statistics != nullptr) { statistics->numberOfSeedRejects++; }
                        continue;
                    }

                    candidateSeeds.vfSeeds.Append(PotentialCandidateSeedVF{ pair.meshB, vertexIndex, pair.meshA, triangleIndex });
                    UpdateSeedStatisticsVF(statistics);
                }
            }

            for (Index edgeIndexA = 0; edgeIndexA < meshA.edgeAABBs.NumberOfItems(); edgeIndexA++)
            {
                Box3D expandedA = ExpandedBox(meshA.edgeAABBs[edgeIndexA], modelSettings.activationDistance);
                for (Index edgeIndexB = 0; edgeIndexB < meshB.edgeAABBs.NumberOfItems(); edgeIndexB++)
                {
                    Box3D expandedB = ExpandedBox(meshB.edgeAABBs[edgeIndexB], modelSettings.activationDistance);
                    if (!BoxesOverlap(expandedA, expandedB))
                    {
                        continue;
                    }

                    PotentialContactCandidate candidate;
                    if (!ComputeEdgeEdgeCandidate(meshA, pair.meshA, edgeIndexA, meshB, pair.meshB, edgeIndexB,
                        modelSettings.activationDistance, candidate))
                    {
                        if (statistics != nullptr) { statistics->numberOfSeedRejects++; }
                        continue;
                    }
                    if (!IsEdgeEdgeSeedFeasibleForOGC(meshA, meshB, candidate, modelSettings))
                    {
                        if (statistics != nullptr) { statistics->numberOfSeedRejects++; }
                        continue;
                    }

                    candidateSeeds.eeSeeds.Append(PotentialCandidateSeedEE{ pair.meshA, edgeIndexA, pair.meshB, edgeIndexB });
                    UpdateSeedStatisticsEE(statistics);
                }
            }
        }
    }

    void PotentialBroadPhaseBuilder::BuildCandidateSeeds(const ResizableArray<PotentialRigidMesh*>& meshes,
        const PotentialModelSettings& modelSettings, PotentialCandidateSeeds& candidateSeeds,
        ResizableArray<PotentialBroadPhasePair>* broadPhasePairs, PotentialContactStatistics* statistics)
    {
        ResizableArray<PotentialBroadPhasePair> localPairs;
        ResizableArray<PotentialBroadPhasePair>& meshPairs = broadPhasePairs ? *broadPhasePairs : localPairs;
        BuildMeshPairs(meshes, modelSettings, meshPairs, statistics);

        if (GetSeedBuilderType(modelSettings) == PotentialSeedBuilderType::OGCFeasibleRegion)
        {
            OGCSeedBuilder::BuildCandidateSeeds(meshes, meshPairs, modelSettings, candidateSeeds, statistics);
        }
        else
        {
            IPCCompatibleSeedBuilder::BuildCandidateSeeds(meshes, meshPairs, modelSettings, candidateSeeds, statistics);
        }
    }
}
