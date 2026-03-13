/** ***********************************************************************************************
* @brief        Collision-set assembly helpers for potential contact
*
************************************************************************************************ */
#ifndef POTENTIALCOLLISIONSET__H
#define POTENTIALCOLLISIONSET__H

#include "System/PotentialBroadPhase.h"
#include "System/PotentialDistanceKernels.h"

namespace PotentialContact
{
    struct PotentialCollisionSet
    {
        ResizableArray<PotentialContactCandidate> normalCandidates;
        ResizableArray<PotentialContactCandidate> tangentialCandidates;

        void Reset(bool freeMemory = false)
        {
            if (freeMemory)
            {
                normalCandidates.Flush();
                tangentialCandidates.Flush();
            }
            else
            {
                normalCandidates.SetNumberOfItems(0);
                tangentialCandidates.SetNumberOfItems(0);
            }
        }
    };

    class PotentialCollisionSetBuilder
    {
    public:
        static PotentialCollisionSetBuilderType GetBuilderType(const PotentialModelSettings& modelSettings);
        static void BuildCollisionSet(const ResizableArray<PotentialRigidMesh*>& meshes, const PotentialCandidateSeeds& candidateSeeds,
            const PotentialModelSettings& modelSettings, PotentialCollisionSet& collisionSet, PotentialContactStatistics* statistics = nullptr);
    };

    class IPCCompatibleCollisionSetBuilder
    {
    public:
        static void BuildCollisionSet(const ResizableArray<PotentialRigidMesh*>& meshes, const PotentialCandidateSeeds& candidateSeeds,
            const PotentialModelSettings& modelSettings, PotentialCollisionSet& collisionSet, PotentialContactStatistics* statistics = nullptr);
    };

    class OGCCollisionSetBuilder
    {
    public:
        static void BuildCollisionSet(const ResizableArray<PotentialRigidMesh*>& meshes, const PotentialCandidateSeeds& candidateSeeds,
            const PotentialModelSettings& modelSettings, PotentialCollisionSet& collisionSet, PotentialContactStatistics* statistics = nullptr);
    };
}

#endif
