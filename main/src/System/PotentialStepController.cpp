/** ***********************************************************************************************
* @brief        Step-controller dispatch for potential contact formulations
*
************************************************************************************************ */

#include "System/PotentialStepController.h"

namespace PotentialStepController
{
    namespace
    {
        bool FindFeasibleStepTrustRegion(const PotentialContact::PotentialStepSettings& settings,
            const std::function<Real(Real)>& minimumDistanceEvaluator,
            PotentialCCD::FeasibleStepResult& result)
        {
            result.Reset();
            result.controllerType = PotentialContact::PotentialStepControllerType::TrustRegion;

            const Real targetTolerance = settings.currentDistanceSlackFactor * settings.minimumDistanceTolerance;
            auto ComputeViolation = [&](Real distance) -> Real
            {
                return EXUstd::Maximum(0., targetTolerance - distance);
            };

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

            Real trustRegionRadius = EXUstd::Maximum(settings.alphaMinimum,
                EXUstd::Minimum(1., settings.trustRegionRadius));
            Real lowerAlpha = 0.;
            Real lowerDistance = currentDistance;
            Real upperDistance = targetTolerance;
            Real upperAlpha = 1.;
            bool hasUpperBound = false;

            for (Index attempt = 0; attempt < settings.maximumReductionSteps; attempt++)
            {
                Real trialAlpha = EXUstd::Minimum(1., trustRegionRadius);
                Real trialDistance = minimumDistanceEvaluator(trialAlpha);
                result.numberOfDistanceEvaluations++;
                if (trialDistance >= targetTolerance)
                {
                    lowerAlpha = trialAlpha;
                    lowerDistance = trialDistance;
                    break;
                }

                hasUpperBound = true;
                upperAlpha = trialAlpha;
                upperDistance = trialDistance;
                result.numberOfTrustRegionRejects++;
                result.numberOfStepReductions++;
                trustRegionRadius *= settings.trustRegionShrinkFactor;
                if (trustRegionRadius < settings.alphaMinimum)
                {
                    break;
                }
            }

            if (lowerAlpha == 0.)
            {
                result.alphaMax = 0.;
                result.minimumDistance = currentDistance;
                result.stepWasClipped = true;
                result.hadFailure = true;
                return false;
            }

            if (!hasUpperBound && lowerAlpha < 1.)
            {
                for (Index attempt = 0; attempt < settings.maximumReductionSteps; attempt++)
                {
                    Real expandedRadius = EXUstd::Minimum(1., EXUstd::Maximum(
                        lowerAlpha + settings.alphaMinimum, trustRegionRadius * settings.trustRegionExpandFactor));
                    if (expandedRadius <= lowerAlpha)
                    {
                        break;
                    }

                    Real trialDistance = minimumDistanceEvaluator(expandedRadius);
                    result.numberOfDistanceEvaluations++;
                    if (trialDistance >= targetTolerance)
                    {
                        trustRegionRadius = expandedRadius;
                        lowerAlpha = expandedRadius;
                        lowerDistance = trialDistance;
                        result.numberOfExpansionSteps++;
                        if (expandedRadius >= 1.)
                        {
                            break;
                        }
                    }
                    else
                    {
                        hasUpperBound = true;
                        upperAlpha = expandedRadius;
                        upperDistance = trialDistance;
                        result.numberOfTrustRegionRejects++;
                        break;
                    }
                }
            }

            if (hasUpperBound && upperAlpha > lowerAlpha)
            {
                for (Index attempt = 0; attempt < settings.maximumBisectionSteps; attempt++)
                {
                    Real trialAlpha = 0.5 * (lowerAlpha + upperAlpha);
                    Real denominator = upperDistance - lowerDistance;
                    if (denominator != 0.)
                    {
                        Real secantAlpha = lowerAlpha +
                            (targetTolerance - lowerDistance) * (upperAlpha - lowerAlpha) / denominator;
                        if (secantAlpha > lowerAlpha && secantAlpha < upperAlpha)
                        {
                            trialAlpha = secantAlpha;
                        }
                    }
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
                    }
                    else
                    {
                        upperAlpha = trialAlpha;
                        upperDistance = trialDistance;
                        result.numberOfTrustRegionRejects++;
                    }
                }
            }

            result.alphaMax = lowerAlpha;
            result.minimumDistance = lowerDistance;
            result.acceptedDistanceMargin = lowerDistance - targetTolerance;
            result.trustRegionRadius = EXUstd::Maximum(settings.alphaMinimum,
                EXUstd::Minimum(1., EXUstd::Maximum(lowerAlpha, trustRegionRadius)));
            result.stepWasClipped = lowerAlpha < 1.;
            return true;
        }
    }

    bool FindFeasibleStep(const PotentialContact::PotentialStepSettings& settings,
        const std::function<Real(Real)>& minimumDistanceEvaluator,
        PotentialCCD::FeasibleStepResult& result)
    {
        if (settings.controllerType == PotentialContact::PotentialStepControllerType::TrustRegion)
        {
            return FindFeasibleStepTrustRegion(settings, minimumDistanceEvaluator, result);
        }

        PotentialCCD::BacktrackingSettings backtrackingSettings;
        backtrackingSettings.minimumDistanceTolerance = settings.minimumDistanceTolerance;
        backtrackingSettings.currentDistanceSlackFactor = settings.currentDistanceSlackFactor;
        backtrackingSettings.reductionFactor = settings.reductionFactor;
        backtrackingSettings.alphaMinimum = settings.alphaMinimum;
        backtrackingSettings.maximumReductionSteps = settings.maximumReductionSteps;
        backtrackingSettings.maximumBisectionSteps = settings.maximumBisectionSteps;

        bool success = PotentialCCD::FindFeasibleStep(backtrackingSettings, minimumDistanceEvaluator, result);
        result.controllerType = PotentialContact::PotentialStepControllerType::CCDLineSearch;
        result.trustRegionRadius = 1.;
        return success;
    }
}
