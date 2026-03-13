/** ***********************************************************************************************
* @brief        Normal-potential model dispatch for IPC / GCP / OGC style rigid contact
* @details      Stage-B split-out of formulation-specific normal energies from distance kernels
*
************************************************************************************************ */
#ifndef POTENTIALNORMALPOTENTIAL__H
#define POTENTIALNORMALPOTENTIAL__H

#include "System/PotentialDistanceKernels.h"

namespace PotentialContact
{
    class IPCNormalPotential
    {
    public:
        static bool Evaluate(const PotentialContactCandidate& candidate,
            const ResizableArray<PotentialRigidMesh*>& meshes,
            const PotentialModelSettings& settings, PotentialContactEvaluation& evaluation);
    };

    class GCPNormalPotential
    {
    public:
        static bool Evaluate(const PotentialContactCandidate& candidate,
            const ResizableArray<PotentialRigidMesh*>& meshes,
            const PotentialModelSettings& settings, PotentialContactEvaluation& evaluation);
    };

    class OGCNormalPotential
    {
    public:
        static bool Evaluate(const PotentialContactCandidate& candidate,
            const ResizableArray<PotentialRigidMesh*>& meshes,
            const PotentialModelSettings& settings, PotentialContactEvaluation& evaluation);
    };

    bool EvaluateNormalPotential(const PotentialContactCandidate& candidate,
        const ResizableArray<PotentialRigidMesh*>& meshes, const PotentialModelSettings& settings,
        PotentialContactEvaluation& evaluation);

    inline STDstring GetPotentialModelTypeString(PotentialModelType type)
    {
        if (type == PotentialModelType::IPC) { return "IPC"; }
        if (type == PotentialModelType::GCP) { return "GCP"; }
        if (type == PotentialModelType::OGC) { return "OGC"; }
        CHECKandTHROWstring("PotentialContact::GetPotentialModelTypeString: invalid model type");
        return "Invalid";
    }
}

#endif
