/** ***********************************************************************************************
* @brief		Implementation for nonlinear CCD style feasible-step filtering
*
************************************************************************************************ */

#include "System/PotentialCCD.h"
#include "System/PotentialDistanceKernels.h"

#include <array>
#include <limits>

#include "Linalg/Geometry.h"

namespace PotentialCCD
{
	namespace
	{
		Real ComputeSceneSearchDistance(const ResizableArray<PotentialContact::PotentialRigidMesh*>& potentialRigidMeshes)
		{
			Box3D sceneBox;
			bool hasVertices = false;
			for (Index meshIndex = 0; meshIndex < potentialRigidMeshes.NumberOfItems(); meshIndex++)
			{
				const auto& mesh = *potentialRigidMeshes[meshIndex];
				for (Index vertexIndex = 0; vertexIndex < mesh.vertexKinematics.NumberOfItems(); vertexIndex++)
				{
					sceneBox.Add(mesh.vertexKinematics[vertexIndex].position);
					hasVertices = true;
				}
			}
			if (!hasVertices)
			{
				return 0.;
			}
			return (sceneBox.PMax() - sceneBox.PMin()).GetL2Norm() + 1.;
		}
	}

	Real ComputeMinimumNormalDistance(const ResizableArray<PotentialContact::PotentialRigidMesh*>& potentialRigidMeshes,
		Index* numberOfCandidates)
	{
		if (numberOfCandidates != nullptr)
		{
			*numberOfCandidates = 0;
		}

		Real minimumDistance = EXUstd::MAXREAL;
		PotentialContact::PotentialContactCandidate candidate;
		const Real searchDistance = ComputeSceneSearchDistance(potentialRigidMeshes);

		for (Index meshA = 0; meshA < potentialRigidMeshes.NumberOfItems(); meshA++)
		{
			const auto& sourceMesh = *potentialRigidMeshes[meshA];
			for (Index meshB = meshA + 1; meshB < potentialRigidMeshes.NumberOfItems(); meshB++)
			{
				const auto& targetMesh = *potentialRigidMeshes[meshB];

				for (Index vertexIndex = 0; vertexIndex < sourceMesh.vertexKinematics.NumberOfItems(); vertexIndex++)
				{
					for (Index triangleIndex = 0; triangleIndex < targetMesh.triangles.NumberOfItems(); triangleIndex++)
					{
						if (!PotentialContact::ComputeVertexFaceCandidate(sourceMesh, meshA, vertexIndex,
							targetMesh, meshB, triangleIndex, searchDistance, candidate))
						{
							continue;
						}

						if (numberOfCandidates != nullptr)
						{
							(*numberOfCandidates)++;
						}
						minimumDistance = EXUstd::Minimum(minimumDistance, candidate.distance);
					}
				}

				for (Index vertexIndex = 0; vertexIndex < targetMesh.vertexKinematics.NumberOfItems(); vertexIndex++)
				{
					for (Index triangleIndex = 0; triangleIndex < sourceMesh.triangles.NumberOfItems(); triangleIndex++)
					{
						if (!PotentialContact::ComputeVertexFaceCandidate(targetMesh, meshB, vertexIndex,
							sourceMesh, meshA, triangleIndex, searchDistance, candidate))
						{
							continue;
						}

						if (numberOfCandidates != nullptr)
						{
							(*numberOfCandidates)++;
						}
						minimumDistance = EXUstd::Minimum(minimumDistance, candidate.distance);
					}
				}

				for (Index edgeIndexA = 0; edgeIndexA < sourceMesh.edgeKinematics.NumberOfItems(); edgeIndexA++)
				{
					for (Index edgeIndexB = 0; edgeIndexB < targetMesh.edgeKinematics.NumberOfItems(); edgeIndexB++)
					{
						if (!PotentialContact::ComputeEdgeEdgeCandidate(sourceMesh, meshA, edgeIndexA,
							targetMesh, meshB, edgeIndexB, searchDistance, candidate))
						{
							continue;
						}

						if (numberOfCandidates != nullptr)
						{
							(*numberOfCandidates)++;
						}
						minimumDistance = EXUstd::Minimum(minimumDistance, candidate.distance);
					}
				}
			}
		}

		return minimumDistance;
	}

	bool FindFeasibleStep(const BacktrackingSettings& settings, const std::function<Real(Real)>& minimumDistanceEvaluator,
		FeasibleStepResult& result)
	{
		result.Reset();

		const Real targetTolerance = settings.currentDistanceSlackFactor * settings.minimumDistanceTolerance;
		Real currentDistance = minimumDistanceEvaluator(0.);
		result.numberOfDistanceEvaluations++;
		result.minimumDistance = currentDistance;
		if (currentDistance < targetTolerance)
		{
			result.alphaMax = 0.;
			result.collisionFree = false;
			result.hadFailure = true;
			return false;
		}

		Real fullStepDistance = minimumDistanceEvaluator(1.);
		result.numberOfDistanceEvaluations++;
		if (fullStepDistance >= targetTolerance)
		{
			result.alphaMax = 1.;
			result.minimumDistance = fullStepDistance;
			return true;
		}

		result.stepWasClipped = true;
		Real lowerAlpha = 0.;
		Real lowerDistance = currentDistance;
		Real lowerPhi = currentDistance - targetTolerance;
		Real upperAlpha = 1.;
		Real upperDistance = fullStepDistance;
		Real upperPhi = fullStepDistance - targetTolerance;
		bool foundBracket = true;

		for (Index reductionStep = 0; reductionStep < settings.maximumReductionSteps; reductionStep++)
		{
			if (upperAlpha - lowerAlpha <= settings.alphaMinimum)
			{
				break;
			}

			Real alpha = lowerAlpha + (upperAlpha - lowerAlpha) * lowerPhi / (lowerPhi - upperPhi);
			Real safeguard = settings.reductionFactor;
			Real lowerLimit = lowerAlpha + safeguard * (upperAlpha - lowerAlpha);
			Real upperLimit = upperAlpha - safeguard * (upperAlpha - lowerAlpha);
			if (!(alpha > lowerAlpha && alpha < upperAlpha))
			{
				alpha = 0.5 * (lowerAlpha + upperAlpha);
			}
			else
			{
				alpha = EXUstd::Maximum(lowerLimit, EXUstd::Minimum(upperLimit, alpha));
			}

			Real trialDistance = minimumDistanceEvaluator(alpha);
			result.numberOfDistanceEvaluations++;
			Real trialPhi = trialDistance - targetTolerance;
			if (trialPhi >= 0.)
			{
				lowerAlpha = alpha;
				lowerDistance = trialDistance;
				lowerPhi = trialPhi;
			}
			else
			{
				upperAlpha = alpha;
				upperDistance = trialDistance;
				upperPhi = trialPhi;
				result.numberOfStepReductions++;
			}

			if (upperAlpha - lowerAlpha <= settings.alphaMinimum)
			{
				foundBracket = true;
				break;
			}
			if (fabs(upperPhi - lowerPhi) <= 1e-15)
			{
				break;
			}
		}

		if (lowerAlpha == 0.)
		{
			Real fallbackUpper = 1.;
			for (Index reductionStep = 0; reductionStep < settings.maximumReductionSteps; reductionStep++)
			{
				fallbackUpper *= settings.reductionFactor;
				result.numberOfStepReductions++;
				if (fallbackUpper < settings.alphaMinimum)
				{
					foundBracket = false;
					break;
				}
				Real trialDistance = minimumDistanceEvaluator(fallbackUpper);
				result.numberOfDistanceEvaluations++;
				if (trialDistance >= targetTolerance)
				{
					lowerAlpha = fallbackUpper;
					lowerDistance = trialDistance;
					lowerPhi = trialDistance - targetTolerance;
					upperAlpha = EXUstd::Minimum(1., fallbackUpper / settings.reductionFactor);
					upperDistance = upperAlpha == 1. ? fullStepDistance : upperDistance;
					upperPhi = upperDistance - targetTolerance;
					foundBracket = true;
					break;
				}
			}
		}

		if (!foundBracket || lowerAlpha == 0.)
		{
			result.alphaMax = 0.;
			result.minimumDistance = currentDistance;
			result.hadFailure = true;
			return false;
		}

		for (Index bisectionStep = 0; bisectionStep < settings.maximumBisectionSteps; bisectionStep++)
		{
			if (upperAlpha - lowerAlpha <= settings.alphaMinimum)
			{
				break;
			}

			Real trialAlpha = 0.5 * (lowerAlpha + upperAlpha);
			if (trialAlpha <= lowerAlpha || trialAlpha >= upperAlpha)
			{
				break;
			}

			Real trialDistance = minimumDistanceEvaluator(trialAlpha);
			result.numberOfDistanceEvaluations++;
			if (trialDistance >= targetTolerance)
			{
				lowerAlpha = trialAlpha;
				lowerDistance = trialDistance;
				lowerPhi = trialDistance - targetTolerance;
			}
			else
			{
				upperAlpha = trialAlpha;
				upperDistance = trialDistance;
				upperPhi = trialDistance - targetTolerance;
			}
		}

		result.alphaMax = lowerAlpha;
		result.minimumDistance = lowerDistance;
		return true;
	}
}
