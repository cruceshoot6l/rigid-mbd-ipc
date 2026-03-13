/** ***********************************************************************************************
* @brief        Distance kernels and barrier evaluation for potential contact
*
************************************************************************************************ */

#include "System/PotentialDistanceKernels.h"

#include <array>
#include <limits>

#include "Linalg/Geometry.h"

namespace PotentialContact
{
    namespace
    {
        void ClosestPointsOnSegments(const Vector3D& p0, const Vector3D& p1, const Vector3D& q0, const Vector3D& q1,
            Real& s, Real& t, Vector3D& c0, Vector3D& c1)
        {
            const Real eps = 1e-15;
            Vector3D d1 = p1 - p0;
            Vector3D d2 = q1 - q0;
            Vector3D r = p0 - q0;
            Real a = d1 * d1;
            Real e = d2 * d2;
            Real f = d2 * r;

            if (a <= eps && e <= eps)
            {
                s = 0.;
                t = 0.;
                c0 = p0;
                c1 = q0;
                return;
            }
            if (a <= eps)
            {
                s = 0.;
                t = EXUstd::Maximum(0., EXUstd::Minimum(1., f / e));
            }
            else
            {
                Real c = d1 * r;
                if (e <= eps)
                {
                    t = 0.;
                    s = EXUstd::Maximum(0., EXUstd::Minimum(1., -c / a));
                }
                else
                {
                    Real b = d1 * d2;
                    Real denom = a * e - b * b;
                    if (denom != 0.)
                    {
                        s = EXUstd::Maximum(0., EXUstd::Minimum(1., (b * f - c * e) / denom));
                    }
                    else
                    {
                        s = 0.;
                    }

                    t = (b * s + f) / e;
                    if (t < 0.)
                    {
                        t = 0.;
                        s = EXUstd::Maximum(0., EXUstd::Minimum(1., -c / a));
                    }
                    else if (t > 1.)
                    {
                        t = 1.;
                        s = EXUstd::Maximum(0., EXUstd::Minimum(1., (b - c) / a));
                    }
                }
            }

            c0 = p0 + s * d1;
            c1 = q0 + t * d2;
        }
    }

    bool VertexFaceDistanceKernel::BuildCandidate(const PotentialRigidMesh& sourceMesh, Index sourceMeshIndex, Index vertexIndex,
        const PotentialRigidMesh& targetMesh, Index targetMeshIndex, Index triangleIndex, Real activationDistance,
        PotentialContactCandidate& candidate)
    {
        candidate.Reset();

        CHECKandTHROW(sourceMesh.vertexKinematics.IsValidIndex(vertexIndex),
            "PotentialContact::ComputeVertexFaceCandidate: invalid source vertex index");
        CHECKandTHROW(targetMesh.triangles.IsValidIndex(triangleIndex),
            "PotentialContact::ComputeVertexFaceCandidate: invalid target triangle index");

        const PotentialSurfacePointKinematics& sourceVertex = sourceMesh.vertexKinematics[vertexIndex];
        Box3D expandedBox = ExpandedBox(targetMesh.triangleAABBs[triangleIndex], activationDistance);
        if (!expandedBox.IsInside(sourceVertex.position))
        {
            return false;
        }

        const PotentialMeshTriangle& triangle = targetMesh.triangles[triangleIndex];
        const Vector3D& x0 = targetMesh.vertexKinematics[triangle.vertices[0]].position;
        const Vector3D& x1 = targetMesh.vertexKinematics[triangle.vertices[1]].position;
        const Vector3D& x2 = targetMesh.vertexKinematics[triangle.vertices[2]].position;

        Vector3D closestPoint;
        Index inside = 0;
        Real distance = EGeometry::MinDistTP(x0, x1, x2, sourceVertex.position, closestPoint, inside);
        if (inside != 1 || distance > activationDistance)
        {
            return false;
        }

        Real lam1 = 0.;
        Real lam2 = 0.;
        EGeometry::LocalTriangleCoordinates(x1 - x0, x2 - x0, closestPoint - x0, lam1, lam2);

        Vector3D barycentricCoordinates({ 1. - lam1 - lam2, lam1, lam2 });
        const Real baryTol = 1e-12;
        if (barycentricCoordinates[0] < -baryTol || barycentricCoordinates[1] < -baryTol || barycentricCoordinates[2] < -baryTol)
        {
            return false;
        }

        Vector3D normal = sourceVertex.position - closestPoint;
        if (distance > std::numeric_limits<Real>::epsilon())
        {
            normal *= 1. / distance;
        }
        else
        {
            std::array<Vector3D, 3> trianglePoints = { x0, x1, x2 };
            normal = EGeometry::ComputeTriangleNormal(trianglePoints);
            if ((sourceVertex.position - x0) * normal < 0.)
            {
                normal *= -1.;
            }
        }

        candidate.type = PotentialCandidateType::VertexFace;
        candidate.sourceMeshIndex = sourceMeshIndex;
        candidate.targetMeshIndex = targetMeshIndex;
        candidate.vertexIndex = vertexIndex;
        candidate.triangleIndex = triangleIndex;
        candidate.closestPoint = closestPoint;
        candidate.normal = normal;
        candidate.barycentricCoordinates = barycentricCoordinates;
        candidate.distance = distance;
        return true;
    }

    bool EdgeEdgeDistanceKernel::BuildCandidate(const PotentialRigidMesh& sourceMesh, Index sourceMeshIndex, Index edgeIndexA,
        const PotentialRigidMesh& targetMesh, Index targetMeshIndex, Index edgeIndexB, Real activationDistance,
        PotentialContactCandidate& candidate)
    {
        candidate.Reset();
        CHECKandTHROW(sourceMesh.edgeKinematics.IsValidIndex(edgeIndexA),
            "PotentialContact::ComputeEdgeEdgeCandidate: invalid source edge index");
        CHECKandTHROW(targetMesh.edgeKinematics.IsValidIndex(edgeIndexB),
            "PotentialContact::ComputeEdgeEdgeCandidate: invalid target edge index");

        Box3D expandedBox = ExpandedBox(targetMesh.edgeAABBs[edgeIndexB], activationDistance);
        if (!(expandedBox.IsInside(sourceMesh.edgeKinematics[edgeIndexA].point0) ||
            expandedBox.IsInside(sourceMesh.edgeKinematics[edgeIndexA].point1) ||
            ExpandedBox(sourceMesh.edgeAABBs[edgeIndexA], activationDistance).IsInside(targetMesh.edgeKinematics[edgeIndexB].point0) ||
            ExpandedBox(sourceMesh.edgeAABBs[edgeIndexA], activationDistance).IsInside(targetMesh.edgeKinematics[edgeIndexB].point1)))
        {
            return false;
        }

        Real s = 0.;
        Real t = 0.;
        Vector3D closestPointA;
        Vector3D closestPointB;
        ClosestPointsOnSegments(sourceMesh.edgeKinematics[edgeIndexA].point0, sourceMesh.edgeKinematics[edgeIndexA].point1,
            targetMesh.edgeKinematics[edgeIndexB].point0, targetMesh.edgeKinematics[edgeIndexB].point1,
            s, t, closestPointA, closestPointB);

        const Real interiorTol = 1e-12;
        if (s <= interiorTol || s >= 1. - interiorTol || t <= interiorTol || t >= 1. - interiorTol)
        {
            return false;
        }

        Vector3D normal = closestPointA - closestPointB;
        Real distance = normal.GetL2Norm();
        if (distance > activationDistance)
        {
            return false;
        }

        if (distance > std::numeric_limits<Real>::epsilon())
        {
            normal *= 1. / distance;
        }
        else
        {
            Vector3D edgeDirectionA = sourceMesh.edgeKinematics[edgeIndexA].point1 - sourceMesh.edgeKinematics[edgeIndexA].point0;
            Vector3D edgeDirectionB = targetMesh.edgeKinematics[edgeIndexB].point1 - targetMesh.edgeKinematics[edgeIndexB].point0;
            normal = edgeDirectionA.CrossProduct(edgeDirectionB);
            if (normal.GetL2NormSquared() == 0.)
            {
                normal.SetVector({ 1., 0., 0. });
            }
            else
            {
                normal.Normalize();
            }
        }

        candidate.type = PotentialCandidateType::EdgeEdge;
        candidate.sourceMeshIndex = sourceMeshIndex;
        candidate.targetMeshIndex = targetMeshIndex;
        candidate.edgeIndexA = edgeIndexA;
        candidate.edgeIndexB = edgeIndexB;
        candidate.closestPointA = closestPointA;
        candidate.closestPointB = closestPointB;
        candidate.normal = normal;
        candidate.edgeCoordinateA = s;
        candidate.edgeCoordinateB = t;
        candidate.distance = distance;
        return true;
    }

    bool ComputeVertexFaceCandidate(const PotentialRigidMesh& sourceMesh, Index sourceMeshIndex, Index vertexIndex,
        const PotentialRigidMesh& targetMesh, Index targetMeshIndex, Index triangleIndex, Real activationDistance,
        PotentialContactCandidate& candidate)
    {
        return VertexFaceDistanceKernel::BuildCandidate(sourceMesh, sourceMeshIndex, vertexIndex,
            targetMesh, targetMeshIndex, triangleIndex, activationDistance, candidate);
    }

    bool ComputeEdgeEdgeCandidate(const PotentialRigidMesh& sourceMesh, Index sourceMeshIndex, Index edgeIndexA,
        const PotentialRigidMesh& targetMesh, Index targetMeshIndex, Index edgeIndexB, Real activationDistance,
        PotentialContactCandidate& candidate)
    {
        return EdgeEdgeDistanceKernel::BuildCandidate(sourceMesh, sourceMeshIndex, edgeIndexA,
            targetMesh, targetMeshIndex, edgeIndexB, activationDistance, candidate);
    }
}
