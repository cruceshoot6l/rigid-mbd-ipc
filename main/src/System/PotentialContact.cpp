/** ***********************************************************************************************
* @brief        Utility implementation for potential-based rigid contact integration
*
************************************************************************************************ */

#include "System/PotentialContact.h"

namespace PotentialContact
{
    void LocalVerticesToMatrix(const ResizableArray<PotentialMeshVertexLocal>& vertices, Matrix& matrix)
    {
        matrix.SetNumberOfRowsAndColumns(vertices.NumberOfItems(), 3);
        for (Index i = 0; i < vertices.NumberOfItems(); i++)
        {
            for (Index j = 0; j < 3; j++)
            {
                matrix(i, j) = vertices[i].xLocal[j];
            }
        }
    }

    void CurrentVerticesToMatrix(const ResizableArray<PotentialSurfacePointKinematics>& vertices, Matrix& matrix)
    {
        matrix.SetNumberOfRowsAndColumns(vertices.NumberOfItems(), 3);
        for (Index i = 0; i < vertices.NumberOfItems(); i++)
        {
            for (Index j = 0; j < 3; j++)
            {
                matrix(i, j) = vertices[i].position[j];
            }
        }
    }

    void CurrentVelocitiesToMatrix(const ResizableArray<PotentialSurfacePointKinematics>& vertices, Matrix& matrix)
    {
        matrix.SetNumberOfRowsAndColumns(vertices.NumberOfItems(), 3);
        for (Index i = 0; i < vertices.NumberOfItems(); i++)
        {
            for (Index j = 0; j < 3; j++)
            {
                matrix(i, j) = vertices[i].velocity[j];
            }
        }
    }

    void TrianglesToMatrixI(const ResizableArray<PotentialMeshTriangle>& triangles, MatrixI& matrix)
    {
        matrix.SetNumberOfRowsAndColumns(triangles.NumberOfItems(), 3);
        for (Index i = 0; i < triangles.NumberOfItems(); i++)
        {
            for (Index j = 0; j < 3; j++)
            {
                matrix(i, j) = triangles[i].vertices[j];
            }
        }
    }

    void TriangleBoxesToMatrices(const ResizableArray<Box3D>& boxes, Matrix& pMin, Matrix& pMax)
    {
        pMin.SetNumberOfRowsAndColumns(boxes.NumberOfItems(), 3);
        pMax.SetNumberOfRowsAndColumns(boxes.NumberOfItems(), 3);
        for (Index i = 0; i < boxes.NumberOfItems(); i++)
        {
            for (Index j = 0; j < 3; j++)
            {
                pMin(i, j) = boxes[i].PMin()[j];
                pMax(i, j) = boxes[i].PMax()[j];
            }
        }
    }
}
