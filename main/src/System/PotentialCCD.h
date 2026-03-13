/** ***********************************************************************************************
* @brief		Helpers for nonlinear CCD style feasible-step filtering
* @details		Week-5 adds conservative backtracking and minimum-distance evaluation
*
************************************************************************************************ */
#ifndef POTENTIALCCD__H
#define POTENTIALCCD__H

#include <functional>

#include "Utilities/BasicDefinitions.h"
#include "System/PotentialSurfaceMesh.h"

namespace PotentialCCD
{
	struct BacktrackingSettings
	{
		Real minimumDistanceTolerance;
		Real currentDistanceSlackFactor;
		Real reductionFactor;
		Real alphaMinimum;
		Index maximumReductionSteps;
		Index maximumBisectionSteps;

		BacktrackingSettings()
		{
			Reset();
		}

		void Reset()
		{
			minimumDistanceTolerance = 1e-6;
			currentDistanceSlackFactor = 1.0;
			reductionFactor = 0.5;
			alphaMinimum = 1e-8;
			maximumReductionSteps = 16;
			maximumBisectionSteps = 8;
		}
	};

	struct FeasibleStepResult
	{
		PotentialContact::PotentialStepControllerType controllerType;
		Real alphaMax;
		Real minimumDistance;
		Index numberOfDistanceEvaluations;
		Index numberOfStepReductions;
		Index numberOfExpansionSteps;
		Index numberOfTrustRegionRejects;
		Real trustRegionRadius;
		Real acceptedDistanceMargin;
		bool stepWasClipped;
		bool collisionFree;
		bool hadFailure;

		FeasibleStepResult()
		{
			Reset();
		}

		void Reset()
		{
			controllerType = PotentialContact::PotentialStepControllerType::CCDLineSearch;
			alphaMax = 1.;
			minimumDistance = EXUstd::MAXREAL;
			numberOfDistanceEvaluations = 0;
			numberOfStepReductions = 0;
			numberOfExpansionSteps = 0;
			numberOfTrustRegionRejects = 0;
			trustRegionRadius = 1.;
			acceptedDistanceMargin = 0.;
			stepWasClipped = false;
			collisionFree = true;
			hadFailure = false;
		}
	};

	Real ComputeMinimumNormalDistance(const ResizableArray<PotentialContact::PotentialRigidMesh*>& potentialRigidMeshes,
		Index* numberOfCandidates = nullptr);

	inline Real ComputeMinimumVertexFaceDistance(const ResizableArray<PotentialContact::PotentialRigidMesh*>& potentialRigidMeshes,
		Index* numberOfCandidates = nullptr);

	bool FindFeasibleStep(const BacktrackingSettings& settings, const std::function<Real(Real)>& minimumDistanceEvaluator,
		FeasibleStepResult& result);

	inline STDstring GetModuleVersionTag()
	{
		return "PotentialCCD-Week8";
	}

	inline Real ComputeMinimumVertexFaceDistance(const ResizableArray<PotentialContact::PotentialRigidMesh*>& potentialRigidMeshes,
		Index* numberOfCandidates)
	{
		return ComputeMinimumNormalDistance(potentialRigidMeshes, numberOfCandidates);
	}
}

#endif
