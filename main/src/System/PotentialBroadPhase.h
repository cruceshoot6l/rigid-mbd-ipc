/** ***********************************************************************************************
* @brief        Broad-phase helpers for potential contact candidate generation
*
************************************************************************************************ */
#ifndef POTENTIALBROADPHASE__H
#define POTENTIALBROADPHASE__H

#include "System/PotentialSurfaceMesh.h"

namespace PotentialContact
{
    struct PotentialBroadPhasePair
    {
        Index meshA;
        Index meshB;
        Box3D overlapBox;

        PotentialBroadPhasePair()
        {
            meshA = EXUstd::InvalidIndex;
            meshB = EXUstd::InvalidIndex;
        }
    };

    struct PotentialCandidateSeedVF
    {
        Index meshV;
        Index vertexIndex;
        Index meshF;
        Index triangleIndex;
    };

    struct PotentialCandidateSeedEE
    {
        Index meshA;
        Index edgeIndexA;
        Index meshB;
        Index edgeIndexB;
    };

    struct PotentialCandidateSeeds
    {
        ResizableArray<PotentialCandidateSeedVF> vfSeeds;
        ResizableArray<PotentialCandidateSeedEE> eeSeeds;

        void Reset(bool freeMemory = false)
        {
            if (freeMemory)
            {
                vfSeeds.Flush();
                eeSeeds.Flush();
            }
            else
            {
                vfSeeds.SetNumberOfItems(0);
                eeSeeds.SetNumberOfItems(0);
            }
        }
    };

    class PotentialBroadPhaseBuilder
    {
    public:
        static PotentialMeshPairBuilderType GetMeshPairBuilderType(const PotentialModelSettings& modelSettings);
        static PotentialSeedBuilderType GetSeedBuilderType(const PotentialModelSettings& modelSettings);
        static void BuildMeshPairs(const ResizableArray<PotentialRigidMesh*>& meshes, const PotentialModelSettings& modelSettings,
            ResizableArray<PotentialBroadPhasePair>& meshPairs, PotentialContactStatistics* statistics = nullptr);
        static void BuildCandidateSeeds(const ResizableArray<PotentialRigidMesh*>& meshes, const PotentialModelSettings& modelSettings,
            PotentialCandidateSeeds& candidateSeeds, ResizableArray<PotentialBroadPhasePair>* broadPhasePairs = nullptr,
            PotentialContactStatistics* statistics = nullptr);
    };

    class IPCCompatibleMeshPairBuilder
    {
    public:
        static void BuildMeshPairs(const ResizableArray<PotentialRigidMesh*>& meshes, const PotentialModelSettings& modelSettings,
            ResizableArray<PotentialBroadPhasePair>& meshPairs, PotentialContactStatistics* statistics = nullptr);
    };

    class OGCMeshPairBuilder
    {
    public:
        static void BuildMeshPairs(const ResizableArray<PotentialRigidMesh*>& meshes, const PotentialModelSettings& modelSettings,
            ResizableArray<PotentialBroadPhasePair>& meshPairs, PotentialContactStatistics* statistics = nullptr);
    };

    class IPCCompatibleSeedBuilder
    {
    public:
        static void BuildCandidateSeeds(const ResizableArray<PotentialRigidMesh*>& meshes,
            const ResizableArray<PotentialBroadPhasePair>& meshPairs, const PotentialModelSettings& modelSettings,
            PotentialCandidateSeeds& candidateSeeds, PotentialContactStatistics* statistics = nullptr);
    };

    class OGCSeedBuilder
    {
    public:
        static void BuildCandidateSeeds(const ResizableArray<PotentialRigidMesh*>& meshes,
            const ResizableArray<PotentialBroadPhasePair>& meshPairs, const PotentialModelSettings& modelSettings,
            PotentialCandidateSeeds& candidateSeeds, PotentialContactStatistics* statistics = nullptr);
    };
}

#endif
