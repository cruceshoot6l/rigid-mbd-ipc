/** ***********************************************************************************************
* @brief        Shared enums and settings for potential-based rigid contact
* @details      Stage-A split-out of common data used by IPC/GCP/OGC formulations
*
************************************************************************************************ */
#ifndef POTENTIALCONTACTCOMMON__H
#define POTENTIALCONTACTCOMMON__H

#include "Linalg/BasicLinalg.h"
#include "Linalg/SearchTree.h"

namespace PotentialContact
{
    enum class PotentialMeshPairBuilderType
    {
        IPCCompatible = 0,
        OGCFeasibleRegion = 1
    };

    enum class PotentialSeedBuilderType
    {
        IPCCompatible = 0,
        OGCFeasibleRegion = 1
    };

    enum class PotentialCollisionSetBuilderType
    {
        IPCCompatible = 0,
        OGCFeasibleRegion = 1
    };

    struct EvaluationSummary
    {
        Index numberOfCoarseBroadPhasePairs;
        Index numberOfBroadPhasePairs;
        Index numberOfBroadPhasePairRejects;
        Index numberOfVertexFaceSeeds;
        Index numberOfEdgeEdgeSeeds;
        Index numberOfSeedRejects;
        Index numberOfVertexFaceCandidates;
        Index numberOfEdgeEdgeCandidates;
        Index numberOfTangentialCandidates;
        Index numberOfCandidates;
        Index numberOfCollisionSetRejects;
        Real minimumDistance;
        Real accumulatedNormalEnergy;
        Real accumulatedFrictionEnergy;
        bool usedGaussNewtonHessian;
        PotentialMeshPairBuilderType meshPairBuilderType;
        PotentialSeedBuilderType seedBuilderType;
        PotentialCollisionSetBuilderType collisionSetBuilderType;

        EvaluationSummary()
        {
            Reset();
        }

        void Reset()
        {
            numberOfCoarseBroadPhasePairs = 0;
            numberOfBroadPhasePairs = 0;
            numberOfBroadPhasePairRejects = 0;
            numberOfVertexFaceSeeds = 0;
            numberOfEdgeEdgeSeeds = 0;
            numberOfSeedRejects = 0;
            numberOfVertexFaceCandidates = 0;
            numberOfEdgeEdgeCandidates = 0;
            numberOfTangentialCandidates = 0;
            numberOfCandidates = 0;
            numberOfCollisionSetRejects = 0;
            minimumDistance = EXUstd::MAXREAL;
            accumulatedNormalEnergy = 0.;
            accumulatedFrictionEnergy = 0.;
            usedGaussNewtonHessian = true;
            meshPairBuilderType = PotentialMeshPairBuilderType::IPCCompatible;
            seedBuilderType = PotentialSeedBuilderType::IPCCompatible;
            collisionSetBuilderType = PotentialCollisionSetBuilderType::IPCCompatible;
        }
    };

    struct PotentialContactStatistics
    {
        Index numberOfCoarseBroadPhasePairs;
        Index numberOfBroadPhasePairs;
        Index numberOfBroadPhasePairRejects;
        Index numberOfVertexFaceSeeds;
        Index numberOfEdgeEdgeSeeds;
        Index numberOfSeedRejects;
        Index numberOfVertexFaceCandidates;
        Index numberOfEdgeEdgeCandidates;
        Index numberOfTangentialCandidates;
        Index numberOfCollisionSetRejects;
        Real minimumDistance;
        Real accumulatedNormalEnergy;
        Real accumulatedFrictionEnergy;
        Index numberOfClippedSteps;
        Index numberOfTrustRegionRejects;
        PotentialMeshPairBuilderType meshPairBuilderType;
        PotentialSeedBuilderType seedBuilderType;
        PotentialCollisionSetBuilderType collisionSetBuilderType;

        PotentialContactStatistics()
        {
            Reset();
        }

        void Reset()
        {
            numberOfCoarseBroadPhasePairs = 0;
            numberOfBroadPhasePairs = 0;
            numberOfBroadPhasePairRejects = 0;
            numberOfVertexFaceSeeds = 0;
            numberOfEdgeEdgeSeeds = 0;
            numberOfSeedRejects = 0;
            numberOfVertexFaceCandidates = 0;
            numberOfEdgeEdgeCandidates = 0;
            numberOfTangentialCandidates = 0;
            numberOfCollisionSetRejects = 0;
            minimumDistance = EXUstd::MAXREAL;
            accumulatedNormalEnergy = 0.;
            accumulatedFrictionEnergy = 0.;
            numberOfClippedSteps = 0;
            numberOfTrustRegionRejects = 0;
            meshPairBuilderType = PotentialMeshPairBuilderType::IPCCompatible;
            seedBuilderType = PotentialSeedBuilderType::IPCCompatible;
            collisionSetBuilderType = PotentialCollisionSetBuilderType::IPCCompatible;
        }
    };

    enum class PotentialCandidateType
    {
        VertexFace = 0,
        EdgeEdge = 1
    };

    enum class PotentialModelType
    {
        IPC = 0,
        GCP = 1,
        OGC = 2
    };

    enum class PotentialStepControllerType
    {
        CCDLineSearch = 0,
        TrustRegion = 1
    };

    inline STDstring GetPotentialStepControllerTypeString(PotentialStepControllerType type)
    {
        if (type == PotentialStepControllerType::CCDLineSearch) { return "CCDLineSearch"; }
        if (type == PotentialStepControllerType::TrustRegion) { return "TrustRegion"; }
        CHECKandTHROWstring("PotentialContact::GetPotentialStepControllerTypeString: invalid controller type");
        return "Invalid";
    }

    inline STDstring GetPotentialMeshPairBuilderTypeString(PotentialMeshPairBuilderType type)
    {
        if (type == PotentialMeshPairBuilderType::IPCCompatible) { return "IPCCompatible"; }
        if (type == PotentialMeshPairBuilderType::OGCFeasibleRegion) { return "OGCFeasibleRegion"; }
        CHECKandTHROWstring("PotentialContact::GetPotentialMeshPairBuilderTypeString: invalid builder type");
        return "Invalid";
    }

    inline STDstring GetPotentialCollisionSetBuilderTypeString(PotentialCollisionSetBuilderType type)
    {
        if (type == PotentialCollisionSetBuilderType::IPCCompatible) { return "IPCCompatible"; }
        if (type == PotentialCollisionSetBuilderType::OGCFeasibleRegion) { return "OGCFeasibleRegion"; }
        CHECKandTHROWstring("PotentialContact::GetPotentialCollisionSetBuilderTypeString: invalid builder type");
        return "Invalid";
    }

    inline STDstring GetPotentialSeedBuilderTypeString(PotentialSeedBuilderType type)
    {
        if (type == PotentialSeedBuilderType::IPCCompatible) { return "IPCCompatible"; }
        if (type == PotentialSeedBuilderType::OGCFeasibleRegion) { return "OGCFeasibleRegion"; }
        CHECKandTHROWstring("PotentialContact::GetPotentialSeedBuilderTypeString: invalid builder type");
        return "Invalid";
    }

    struct PotentialModelSettings
    {
        PotentialModelType modelType;
        Real activationDistance;
        Real minimumDistance;
        Real stiffness;
        bool useGaussNewtonHessian;
        bool enableFriction;
        Index gcpBarrierPower;
        Real gcpAlphaT;
        Real gcpBetaT;
        Real gcpAlphaN;
        Real gcpBetaN;
        Real gcpInteriorEpsilon;
        Real ogcFaceInteriorTolerance;
        Real ogcEdgeInteriorTolerance;
        Real ogcNormalAlignmentTolerance;
        Real ogcMinimumEdgeCrossNorm;

        PotentialModelSettings()
        {
            Reset();
        }

        void Reset()
        {
            modelType = PotentialModelType::GCP;
            activationDistance = 1e-3;
            minimumDistance = 1e-8;
            stiffness = 1.;
            useGaussNewtonHessian = true;
            enableFriction = false;
            gcpBarrierPower = 2;
            gcpAlphaT = 1.;
            gcpBetaT = 0.;
            gcpAlphaN = 0.1;
            gcpBetaN = 0.;
            gcpInteriorEpsilon = 0.05;
            ogcFaceInteriorTolerance = 0.02;
            ogcEdgeInteriorTolerance = 0.05;
            ogcNormalAlignmentTolerance = 0.75;
            ogcMinimumEdgeCrossNorm = 0.1;
        }
    };

    struct PotentialStepSettings
    {
        PotentialStepControllerType controllerType;
        Real minimumDistanceTolerance;
        Real currentDistanceSlackFactor;
        Real reductionFactor;
        Real alphaMinimum;
        Index maximumReductionSteps;
        Index maximumBisectionSteps;
        Real trustRegionRadius;
        Real trustRegionExpandFactor;
        Real trustRegionShrinkFactor;

        PotentialStepSettings()
        {
            Reset();
        }

        void Reset()
        {
            controllerType = PotentialStepControllerType::CCDLineSearch;
            minimumDistanceTolerance = 1e-6;
            currentDistanceSlackFactor = 1.0;
            reductionFactor = 0.5;
            alphaMinimum = 1e-8;
            maximumReductionSteps = 16;
            maximumBisectionSteps = 8;
            trustRegionRadius = 1.;
            trustRegionExpandFactor = 1.5;
            trustRegionShrinkFactor = 0.5;
        }
    };
}

#endif
