/** ***********************************************************************************************
* @brief        Step-controller dispatch for potential contact formulations
* @details      Separates CCD line search from trust-region / filter-step control
*
************************************************************************************************ */
#ifndef POTENTIALSTEPCONTROLLER__H
#define POTENTIALSTEPCONTROLLER__H

#include <functional>

#include "System/PotentialCCD.h"
#include "System/PotentialContactCommon.h"

namespace PotentialStepController
{
    bool FindFeasibleStep(const PotentialContact::PotentialStepSettings& settings,
        const std::function<Real(Real)>& minimumDistanceEvaluator,
        PotentialCCD::FeasibleStepResult& result);

    inline STDstring GetModuleVersionTag()
    {
        return "PotentialStepController-Week8";
    }
}

#endif
