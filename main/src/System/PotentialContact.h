/** ***********************************************************************************************
* @brief        Aggregated interface for potential-based rigid contact integration
* @details      Stage-A split-out of shared modules while keeping legacy includes stable
*
************************************************************************************************ */
#ifndef POTENTIALCONTACT__H
#define POTENTIALCONTACT__H

#include "System/PotentialContactCommon.h"
#include "System/PotentialSurfaceMesh.h"
#include "System/PotentialDistanceKernels.h"
#include "System/PotentialNormalPotential.h"
#include "System/PotentialBroadPhase.h"
#include "System/PotentialCollisionSet.h"
#include "System/PotentialStepController.h"

namespace PotentialContact
{
    void LocalVerticesToMatrix(const ResizableArray<PotentialMeshVertexLocal>& vertices, Matrix& matrix);
    void CurrentVerticesToMatrix(const ResizableArray<PotentialSurfacePointKinematics>& vertices, Matrix& matrix);
    void CurrentVelocitiesToMatrix(const ResizableArray<PotentialSurfacePointKinematics>& vertices, Matrix& matrix);
    void TrianglesToMatrixI(const ResizableArray<PotentialMeshTriangle>& triangles, MatrixI& matrix);
    void TriangleBoxesToMatrices(const ResizableArray<Box3D>& boxes, Matrix& pMin, Matrix& pMax);

    inline STDstring GetModuleVersionTag()
    {
        return "PotentialContact-Week4";
    }
}

#endif
