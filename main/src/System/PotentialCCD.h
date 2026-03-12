/** ***********************************************************************************************
* @brief		Placeholder declarations for nonlinear CCD based feasible-step filtering
* @details		Week-1 scaffold for barrier-based solver integration
*
************************************************************************************************ */
#ifndef POTENTIALCCD__H
#define POTENTIALCCD__H

#include "Utilities/BasicDefinitions.h"

namespace PotentialCCD
{
	struct FeasibleStepResult
	{
		Real alphaMax;
		Real minimumDistance;
		bool collisionFree;

		FeasibleStepResult()
		{
			Reset();
		}

		void Reset()
		{
			alphaMax = 1.;
			minimumDistance = EXUstd::MAXREAL;
			collisionFree = true;
		}
	};

	inline STDstring GetModuleVersionTag()
	{
		return "PotentialCCD-Week1";
	}
}

#endif
