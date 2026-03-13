/** ***********************************************************************************************
* @brief        Normal-potential model dispatch for IPC / GCP / OGC style rigid contact
*
************************************************************************************************ */

#include "System/PotentialNormalPotential.h"

#include <cmath>
#include <limits>

namespace PotentialContact
{
    namespace
    {
        constexpr Real kSmallDistanceFactor = 1. + 1e-12;
        constexpr Real kSmallGeometry = 1e-15;

        Real ClampedLogBarrier(Real d, Real dhat)
        {
            if (d <= 0.)
            {
                return std::numeric_limits<Real>::infinity();
            }
            if (d >= dhat)
            {
                return 0.;
            }
            Real delta = d - dhat;
            return -(delta * delta) * log(d / dhat);
        }

        Real ClampedLogBarrierDerivative(Real d, Real dhat)
        {
            if (d <= 0. || d >= dhat)
            {
                return 0.;
            }
            return (dhat - d) * (2. * log(d / dhat) - dhat / d + 1.);
        }

        Real ClampedLogBarrierSecondDerivative(Real d, Real dhat)
        {
            if (d <= 0. || d >= dhat)
            {
                return 0.;
            }
            Real dhatOverD = dhat / d;
            return (dhatOverD + 2.) * dhatOverD - 2. * log(d / dhat) - 3.;
        }

        Real CubicSpline(Real x)
        {
            Real ax = fabs(x);
            if (ax >= 1.)
            {
                return 0.;
            }
            if (ax >= 0.5)
            {
                return (4. / 3.) * EXUstd::Cube(1. - ax);
            }
            return 2. / 3. - 4. * x * x * (1. - ax);
        }

        Real CubicSplineDerivative(Real x)
        {
            Real ax = fabs(x);
            if (ax >= 1.)
            {
                return 0.;
            }
            if (ax >= 0.5)
            {
                return -4. * EXUstd::Square(1. - ax) * (x >= 0. ? 1. : -1.);
            }
            return 4. * x * (3. * ax - 2.);
        }

        Real CubicSplineSecondDerivative(Real x)
        {
            Real ax = fabs(x);
            if (ax >= 1.)
            {
                return 0.;
            }
            if (ax >= 0.5)
            {
                return 8. * (1. - ax);
            }
            return 8. * (3. * ax - 1.);
        }

        Real InverseBarrier(Real x, Index r)
        {
            if (x <= 0.)
            {
                return std::numeric_limits<Real>::infinity();
            }
            if (x >= 1.)
            {
                return 0.;
            }
            return CubicSpline(x) / std::pow(x, (Real)r);
        }

        Real InverseBarrierDerivative(Real x, Index r)
        {
            if (x <= 0. || x >= 1.)
            {
                return 0.;
            }
            Real xr = std::pow(x, (Real)r);
            return (CubicSplineDerivative(x) - CubicSpline(x) * ((Real)r) / x) / xr;
        }

        Real InverseBarrierSecondDerivative(Real x, Index r)
        {
            if (x <= 0. || x >= 1.)
            {
                return 0.;
            }
            Real xr = std::pow(x, (Real)r);
            return (CubicSplineSecondDerivative(x)
                + (-2. * CubicSplineDerivative(x)
                    + (((Real)r) + 1.) * CubicSpline(x) / x) * ((Real)r) / x) / xr;
        }

        Real SmoothHeavisideStandard(Real x)
        {
            if (x <= -3.)
            {
                return 0.;
            }
            if (x <= -2.)
            {
                return EXUstd::Cube(3. + x) / 6.;
            }
            if (x <= -1.)
            {
                return (((-2. * x - 9.) * x - 9.) * x + 3.) / 6.;
            }
            if (x < 0.)
            {
                return EXUstd::Cube(x) / 6. + 1.;
            }
            return 1.;
        }

        Real SmoothHeaviside(Real x, Real alpha, Real beta)
        {
            Real scale = 3. / EXUstd::Maximum(alpha + beta, 1e-12);
            return SmoothHeavisideStandard((x - beta) * scale);
        }

        Real ComputeTriangleArea(const PotentialRigidMesh& mesh, Index triangleIndex)
        {
            const auto& triangle = mesh.triangles[triangleIndex];
            const Vector3D& x0 = mesh.vertexKinematics[triangle.vertices[0]].position;
            const Vector3D& x1 = mesh.vertexKinematics[triangle.vertices[1]].position;
            const Vector3D& x2 = mesh.vertexKinematics[triangle.vertices[2]].position;
            return 0.5 * (x1 - x0).CrossProduct(x2 - x0).GetL2Norm();
        }

        Vector3D ComputeTriangleUnitNormal(const PotentialRigidMesh& mesh, Index triangleIndex)
        {
            const auto& triangle = mesh.triangles[triangleIndex];
            const Vector3D& x0 = mesh.vertexKinematics[triangle.vertices[0]].position;
            const Vector3D& x1 = mesh.vertexKinematics[triangle.vertices[1]].position;
            const Vector3D& x2 = mesh.vertexKinematics[triangle.vertices[2]].position;
            Vector3D normal = (x1 - x0).CrossProduct(x2 - x0);
            Real norm = normal.GetL2Norm();
            if (norm <= kSmallGeometry)
            {
                normal.SetAll(0.);
                return normal;
            }
            normal *= 1. / norm;
            return normal;
        }

        Real ComputeVertexSmoothWeight(const PotentialRigidMesh& mesh, Index vertexIndex,
            const Vector3D& tangentialDirection, const Vector3D& normalDirection,
            const PotentialModelSettings& settings)
        {
            if (!mesh.vertexKinematics.IsValidIndex(vertexIndex))
            {
                return 0.;
            }

            const Vector3D& xVertex = mesh.vertexKinematics[vertexIndex].position;
            Real tangentWeight = 1.;
            Real summedEdgeLengthSquared = 0.;
            Index numberOfIncidentEdges = 0;

            for (Index edgeIndex = 0; edgeIndex < mesh.edges.NumberOfItems(); edgeIndex++)
            {
                const auto& edge = mesh.edges[edgeIndex];
                Index otherVertex = EXUstd::InvalidIndex;
                if (edge.vertices[0] == vertexIndex)
                {
                    otherVertex = edge.vertices[1];
                }
                else if (edge.vertices[1] == vertexIndex)
                {
                    otherVertex = edge.vertices[0];
                }
                if (otherVertex == EXUstd::InvalidIndex)
                {
                    continue;
                }

                Vector3D tangent = mesh.vertexKinematics[otherVertex].position - xVertex;
                Real tangentNorm = tangent.GetL2Norm();
                if (tangentNorm <= kSmallGeometry)
                {
                    continue;
                }

                tangentWeight *= SmoothHeaviside((tangentialDirection * tangent) / tangentNorm,
                    settings.gcpAlphaT, settings.gcpBetaT);
                if (tangentWeight <= 0.)
                {
                    return 0.;
                }
                summedEdgeLengthSquared += tangentNorm * tangentNorm;
                numberOfIncidentEdges++;
            }

            if (numberOfIncidentEdges == 0)
            {
                return 0.;
            }

            Real normalAccumulator = 0.;
            Index numberOfIncidentTriangles = 0;
            for (Index triangleIndex = 0; triangleIndex < mesh.triangles.NumberOfItems(); triangleIndex++)
            {
                const auto& triangle = mesh.triangles[triangleIndex];
                if (triangle.vertices[0] != vertexIndex && triangle.vertices[1] != vertexIndex &&
                    triangle.vertices[2] != vertexIndex)
                {
                    continue;
                }

                Vector3D triangleNormal = ComputeTriangleUnitNormal(mesh, triangleIndex);
                if (triangleNormal.GetL2NormSquared() <= kSmallGeometry)
                {
                    continue;
                }
                normalAccumulator += SmoothHeaviside(normalDirection * triangleNormal,
                    settings.gcpAlphaN, settings.gcpBetaN);
                numberOfIncidentTriangles++;
            }

            Real normalWeight = 1.;
            if (numberOfIncidentTriangles > 0)
            {
                normalWeight = SmoothHeaviside(normalAccumulator - 1., 1., 0.);
                if (normalWeight <= 0.)
                {
                    return 0.;
                }
            }

            return (summedEdgeLengthSquared / 3.) * tangentWeight * normalWeight;
        }

        Real ComputeEdgeSmoothWeight(const PotentialRigidMesh& mesh, Index edgeIndex,
            const Vector3D& tangentialDirection, const Vector3D& normalDirection,
            const PotentialModelSettings& settings)
        {
            if (!mesh.edges.IsValidIndex(edgeIndex))
            {
                return 0.;
            }

            const auto& edge = mesh.edges[edgeIndex];
            const Vector3D& e0 = mesh.vertexKinematics[edge.vertices[0]].position;
            const Vector3D& e1 = mesh.vertexKinematics[edge.vertices[1]].position;
            Vector3D edgeVector = e1 - e0;
            Real edgeLengthSquared = edgeVector.GetL2NormSquared();
            if (edgeLengthSquared <= kSmallGeometry)
            {
                return 0.;
            }

            Real tangentWeight = 1.;
            Real normalAccumulator = 0.;
            Index adjacentTriangleCount = 0;

            for (Index triangleIndex = 0; triangleIndex < mesh.triangles.NumberOfItems(); triangleIndex++)
            {
                const auto& triangle = mesh.triangles[triangleIndex];
                Index oppositeVertex = EXUstd::InvalidIndex;
                Index matches = 0;
                for (Index localVertex = 0; localVertex < 3; localVertex++)
                {
                    Index v = triangle.vertices[localVertex];
                    if (v == edge.vertices[0] || v == edge.vertices[1])
                    {
                        matches++;
                    }
                    else
                    {
                        oppositeVertex = v;
                    }
                }
                if (matches != 2 || oppositeVertex == EXUstd::InvalidIndex)
                {
                    continue;
                }

                const Vector3D& xOpposite = mesh.vertexKinematics[oppositeVertex].position;
                Real alpha = ((xOpposite - e0) * edgeVector) / edgeLengthSquared;
                Vector3D projectedPoint = e0 + alpha * edgeVector;
                Vector3D triangleDirection = xOpposite - projectedPoint;
                Real triangleDirectionNorm = triangleDirection.GetL2Norm();
                if (triangleDirectionNorm > kSmallGeometry)
                {
                    tangentWeight *= SmoothHeaviside(
                        (tangentialDirection * triangleDirection) / triangleDirectionNorm,
                        settings.gcpAlphaT, settings.gcpBetaT);
                    if (tangentWeight <= 0.)
                    {
                        return 0.;
                    }
                }

                Vector3D triangleNormal = ComputeTriangleUnitNormal(mesh, triangleIndex);
                if (triangleNormal.GetL2NormSquared() > kSmallGeometry)
                {
                    normalAccumulator += SmoothHeaviside(normalDirection * triangleNormal,
                        settings.gcpAlphaN, settings.gcpBetaN);
                    adjacentTriangleCount++;
                }
            }

            Real normalWeight = 1.;
            if (adjacentTriangleCount > 0)
            {
                normalWeight = SmoothHeaviside(normalAccumulator - 1., 1., 0.);
                if (normalWeight <= 0.)
                {
                    return 0.;
                }
            }

            return edgeLengthSquared * tangentWeight * normalWeight;
        }

        Real ComputeVFInteriorWeight(const PotentialContactCandidate& candidate, const PotentialModelSettings& settings)
        {
            Real interiorWeight = 1.;
            Real epsilon = EXUstd::Maximum(settings.gcpInteriorEpsilon, 1e-12);
            for (Index i = 0; i < 3; i++)
            {
                interiorWeight *= SmoothHeaviside(candidate.barycentricCoordinates[i] - epsilon, epsilon, 0.);
            }
            return interiorWeight;
        }

        Real ComputeEEInteriorWeight(const PotentialContactCandidate& candidate, const PotentialModelSettings& settings)
        {
            Real epsilon = EXUstd::Maximum(settings.gcpInteriorEpsilon, 1e-12);
            Real weightA = SmoothHeaviside(candidate.edgeCoordinateA - epsilon, epsilon, 0.) *
                SmoothHeaviside((1. - epsilon) - candidate.edgeCoordinateA, epsilon, 0.);
            Real weightB = SmoothHeaviside(candidate.edgeCoordinateB - epsilon, epsilon, 0.) *
                SmoothHeaviside((1. - epsilon) - candidate.edgeCoordinateB, epsilon, 0.);
            return weightA * weightB;
        }

        Real ComputeGCPGeometricWeight(const PotentialContactCandidate& candidate,
            const ResizableArray<PotentialRigidMesh*>& meshes, const PotentialModelSettings& settings)
        {
            CHECKandTHROW(meshes.IsValidIndex(candidate.sourceMeshIndex) && meshes.IsValidIndex(candidate.targetMeshIndex),
                "PotentialContact::ComputeGCPGeometricWeight: invalid source/target mesh indices");

            const auto& sourceMesh = *meshes[candidate.sourceMeshIndex];
            const auto& targetMesh = *meshes[candidate.targetMeshIndex];

            if (candidate.type == PotentialCandidateType::VertexFace)
            {
                Real faceArea = ComputeTriangleArea(targetMesh, candidate.triangleIndex);
                Real pointWeight = ComputeVertexSmoothWeight(sourceMesh, candidate.vertexIndex,
                    candidate.normal, -candidate.normal, settings);
                Real interiorWeight = ComputeVFInteriorWeight(candidate, settings);
                return faceArea * pointWeight * interiorWeight;
            }

            Real sourceEdgeWeight = ComputeEdgeSmoothWeight(sourceMesh, candidate.edgeIndexA,
                candidate.normal, -candidate.normal, settings);
            Real targetEdgeWeight = ComputeEdgeSmoothWeight(targetMesh, candidate.edgeIndexB,
                -candidate.normal, candidate.normal, settings);
            Real interiorWeight = ComputeEEInteriorWeight(candidate, settings);
            return sourceEdgeWeight * targetEdgeWeight * interiorWeight;
        }

        void AppendUniqueLocalVertexReference(ResizableArray<PotentialLocalVertexReference>& references,
            Index meshIndex, Index vertexIndex)
        {
            if (meshIndex == EXUstd::InvalidIndex || vertexIndex == EXUstd::InvalidIndex)
            {
                return;
            }
            for (Index i = 0; i < references.NumberOfItems(); i++)
            {
                if (references[i].meshIndex == meshIndex && references[i].vertexIndex == vertexIndex)
                {
                    return;
                }
            }
            references.Append(PotentialLocalVertexReference());
            references.Last().meshIndex = meshIndex;
            references.Last().vertexIndex = vertexIndex;
        }

        void AppendIncidentTriangleVerticesForVertex(const PotentialRigidMesh& mesh, Index meshIndex, Index vertexIndex,
            ResizableArray<PotentialLocalVertexReference>& references)
        {
            for (Index triangleIndex = 0; triangleIndex < mesh.triangles.NumberOfItems(); triangleIndex++)
            {
                const auto& triangle = mesh.triangles[triangleIndex];
                if (triangle.vertices[0] != vertexIndex && triangle.vertices[1] != vertexIndex &&
                    triangle.vertices[2] != vertexIndex)
                {
                    continue;
                }
                for (Index i = 0; i < 3; i++)
                {
                    AppendUniqueLocalVertexReference(references, meshIndex, triangle.vertices[i]);
                }
            }
        }

        void AppendIncidentTriangleVerticesForEdge(const PotentialRigidMesh& mesh, Index meshIndex, Index edgeIndex,
            ResizableArray<PotentialLocalVertexReference>& references)
        {
            if (!mesh.edges.IsValidIndex(edgeIndex))
            {
                return;
            }
            const auto& edge = mesh.edges[edgeIndex];
            for (Index triangleIndex = 0; triangleIndex < mesh.triangles.NumberOfItems(); triangleIndex++)
            {
                const auto& triangle = mesh.triangles[triangleIndex];
                Index matches = 0;
                for (Index i = 0; i < 3; i++)
                {
                    Index vertexIndex = triangle.vertices[i];
                    if (vertexIndex == edge.vertices[0] || vertexIndex == edge.vertices[1])
                    {
                        matches++;
                    }
                }
                if (matches != 2)
                {
                    continue;
                }
                for (Index i = 0; i < 3; i++)
                {
                    AppendUniqueLocalVertexReference(references, meshIndex, triangle.vertices[i]);
                }
            }
        }

        void ClonePotentialSurfacePointKinematics(const PotentialSurfacePointKinematics& source,
            PotentialSurfacePointKinematics& target)
        {
            target.position = source.position;
            target.velocity = source.velocity;
            target.positionJacobian.CopyFrom(source.positionJacobian);
        }

        void ClonePotentialSurfaceEdgeKinematics(const PotentialSurfaceEdgeKinematics& source,
            PotentialSurfaceEdgeKinematics& target)
        {
            target.point0 = source.point0;
            target.point1 = source.point1;
            target.velocity0 = source.velocity0;
            target.velocity1 = source.velocity1;
            target.point0Jacobian.CopyFrom(source.point0Jacobian);
            target.point1Jacobian.CopyFrom(source.point1Jacobian);
        }

        void ClonePotentialRigidMeshState(const PotentialRigidMeshState& source, PotentialRigidMeshState& target)
        {
            target.markerPosition = source.markerPosition;
            target.markerOrientation = source.markerOrientation;
            target.markerVelocity = source.markerVelocity;
            target.markerAngularVelocityLocal = source.markerAngularVelocityLocal;
            target.markerPositionJacobian.CopyFrom(source.markerPositionJacobian);
            target.markerRotationJacobian.CopyFrom(source.markerRotationJacobian);
            target.markerLTG.CopyFrom(source.markerLTG);
        }

        void ClonePotentialRigidMesh(const PotentialRigidMesh& source, PotentialRigidMesh& target)
        {
            target.Reset(false);
            target.markerIndex = source.markerIndex;
            target.frictionMaterialIndex = source.frictionMaterialIndex;
            target.staticMesh = source.staticMesh;
            target.verticesLocal.CopyFrom(source.verticesLocal);
            target.edges.CopyFrom(source.edges);
            target.triangles.CopyFrom(source.triangles);
            target.edgeAABBs.CopyFrom(source.edgeAABBs);
            target.triangleAABBs.CopyFrom(source.triangleAABBs);

            target.vertexKinematics.SetNumberOfItems(source.vertexKinematics.NumberOfItems());
            for (Index i = 0; i < source.vertexKinematics.NumberOfItems(); i++)
            {
                ClonePotentialSurfacePointKinematics(source.vertexKinematics[i], target.vertexKinematics[i]);
            }

            target.edgeKinematics.SetNumberOfItems(source.edgeKinematics.NumberOfItems());
            for (Index i = 0; i < source.edgeKinematics.NumberOfItems(); i++)
            {
                ClonePotentialSurfaceEdgeKinematics(source.edgeKinematics[i], target.edgeKinematics[i]);
            }

            ClonePotentialRigidMeshState(source.state, target.state);
        }

        void CollectGCPLocalVertexReferences(const PotentialContactCandidate& candidate,
            const ResizableArray<PotentialRigidMesh*>& meshes,
            ResizableArray<PotentialLocalVertexReference>& references)
        {
            references.SetNumberOfItems(0);
            CHECKandTHROW(meshes.IsValidIndex(candidate.sourceMeshIndex) && meshes.IsValidIndex(candidate.targetMeshIndex),
                "PotentialContact::CollectGCPLocalVertexReferences: invalid source/target mesh indices");

            const auto& sourceMesh = *meshes[candidate.sourceMeshIndex];
            const auto& targetMesh = *meshes[candidate.targetMeshIndex];

            if (candidate.type == PotentialCandidateType::VertexFace)
            {
                AppendUniqueLocalVertexReference(references, candidate.sourceMeshIndex, candidate.vertexIndex);
                AppendIncidentTriangleVerticesForVertex(sourceMesh, candidate.sourceMeshIndex, candidate.vertexIndex, references);

                if (targetMesh.triangles.IsValidIndex(candidate.triangleIndex))
                {
                    const auto& triangle = targetMesh.triangles[candidate.triangleIndex];
                    for (Index i = 0; i < 3; i++)
                    {
                        AppendUniqueLocalVertexReference(references, candidate.targetMeshIndex, triangle.vertices[i]);
                    }
                }
                return;
            }

            if (sourceMesh.edges.IsValidIndex(candidate.edgeIndexA))
            {
                const auto& edgeA = sourceMesh.edges[candidate.edgeIndexA];
                AppendUniqueLocalVertexReference(references, candidate.sourceMeshIndex, edgeA.vertices[0]);
                AppendUniqueLocalVertexReference(references, candidate.sourceMeshIndex, edgeA.vertices[1]);
                AppendIncidentTriangleVerticesForEdge(sourceMesh, candidate.sourceMeshIndex, candidate.edgeIndexA, references);
            }

            if (targetMesh.edges.IsValidIndex(candidate.edgeIndexB))
            {
                const auto& edgeB = targetMesh.edges[candidate.edgeIndexB];
                AppendUniqueLocalVertexReference(references, candidate.targetMeshIndex, edgeB.vertices[0]);
                AppendUniqueLocalVertexReference(references, candidate.targetMeshIndex, edgeB.vertices[1]);
                AppendIncidentTriangleVerticesForEdge(targetMesh, candidate.targetMeshIndex, candidate.edgeIndexB, references);
            }
        }

        bool EvaluateGCPScalarPotential(const PotentialContactCandidate& candidate,
            const ResizableArray<PotentialRigidMesh*>& meshes, const PotentialModelSettings& settings,
            PotentialContactEvaluation& evaluation)
        {
            evaluation.Reset();
            const Real activationDistance = settings.activationDistance;
            if (activationDistance <= 0.)
            {
                return false;
            }

            Real geometricWeight = ComputeGCPGeometricWeight(candidate, meshes, settings);
            if (geometricWeight <= 0.)
            {
                return false;
            }

            Real distanceClamped = EXUstd::Maximum(candidate.distance, settings.minimumDistance * kSmallDistanceFactor);
            if (distanceClamped >= activationDistance)
            {
                return false;
            }

            Real normalizedDistance = distanceClamped / activationDistance;
            Real inverseBarrier = InverseBarrier(normalizedDistance, settings.gcpBarrierPower);
            if (!std::isfinite(inverseBarrier) || inverseBarrier <= 0.)
            {
                return false;
            }

            Real inverseBarrierDerivative = InverseBarrierDerivative(normalizedDistance, settings.gcpBarrierPower);
            Real inverseBarrierSecondDerivative = InverseBarrierSecondDerivative(normalizedDistance, settings.gcpBarrierPower);

            evaluation.modelType = PotentialModelType::GCP;
            evaluation.energy = settings.stiffness * geometricWeight * inverseBarrier;
            evaluation.derivativeWRTDistance =
                settings.stiffness * geometricWeight * inverseBarrierDerivative / activationDistance;
            evaluation.secondDerivativeWRTDistance =
                settings.stiffness * geometricWeight * inverseBarrierSecondDerivative /
                (activationDistance * activationDistance);
            evaluation.distanceClamped = distanceClamped;
            evaluation.effectiveActivationDistance = activationDistance;
            evaluation.active = true;
            return true;
        }

        void SetLocalVertexPositions(const ResizableArray<PotentialLocalVertexReference>& references,
            const ResizableVector& localPositions, PotentialRigidMesh& sourceMesh, Index sourceMeshIndex,
            PotentialRigidMesh* targetMesh, Index targetMeshIndex)
        {
            CHECKandTHROW(localPositions.NumberOfItems() == 3 * references.NumberOfItems(),
                "PotentialContact::SetLocalVertexPositions: inconsistent coordinate size");

            for (Index i = 0; i < references.NumberOfItems(); i++)
            {
                const auto& reference = references[i];
                Vector3D position({ localPositions[3 * i], localPositions[3 * i + 1], localPositions[3 * i + 2] });

                if (reference.meshIndex == sourceMeshIndex)
                {
                    CHECKandTHROW(sourceMesh.vertexKinematics.IsValidIndex(reference.vertexIndex),
                        "PotentialContact::SetLocalVertexPositions: invalid source vertex index");
                    sourceMesh.vertexKinematics[reference.vertexIndex].position = position;
                    continue;
                }
                CHECKandTHROW(targetMesh != nullptr && reference.meshIndex == targetMeshIndex,
                    "PotentialContact::SetLocalVertexPositions: invalid target mesh selection");
                CHECKandTHROW(targetMesh->vertexKinematics.IsValidIndex(reference.vertexIndex),
                    "PotentialContact::SetLocalVertexPositions: invalid target vertex index");
                targetMesh->vertexKinematics[reference.vertexIndex].position = position;
            }
        }

        bool RebuildCandidateForLocalCoordinates(const PotentialContactCandidate& templateCandidate,
            const ResizableArray<PotentialRigidMesh*>& meshes, Real activationDistance, PotentialContactCandidate& candidate)
        {
            CHECKandTHROW(meshes.IsValidIndex(templateCandidate.sourceMeshIndex) &&
                meshes.IsValidIndex(templateCandidate.targetMeshIndex),
                "PotentialContact::RebuildCandidateForLocalCoordinates: invalid mesh index");

            const auto& sourceMesh = *meshes[templateCandidate.sourceMeshIndex];
            const auto& targetMesh = *meshes[templateCandidate.targetMeshIndex];
            if (templateCandidate.type == PotentialCandidateType::VertexFace)
            {
                return ComputeVertexFaceCandidate(sourceMesh, templateCandidate.sourceMeshIndex, templateCandidate.vertexIndex,
                    targetMesh, templateCandidate.targetMeshIndex, templateCandidate.triangleIndex,
                    activationDistance, candidate);
            }
            return ComputeEdgeEdgeCandidate(sourceMesh, templateCandidate.sourceMeshIndex, templateCandidate.edgeIndexA,
                targetMesh, templateCandidate.targetMeshIndex, templateCandidate.edgeIndexB,
                activationDistance, candidate);
        }

        bool EvaluateGCPEnergyAtLocalCoordinates(const PotentialContactCandidate& templateCandidate,
            const ResizableArray<PotentialRigidMesh*>& meshes, const PotentialModelSettings& settings,
            const ResizableArray<PotentialLocalVertexReference>& references, const ResizableVector& localPositions,
            Real& energy)
        {
            CHECKandTHROW(meshes.IsValidIndex(templateCandidate.sourceMeshIndex) &&
                meshes.IsValidIndex(templateCandidate.targetMeshIndex),
                "PotentialContact::EvaluateGCPEnergyAtLocalCoordinates: invalid mesh index");

            PotentialRigidMesh sourceCopy;
            ClonePotentialRigidMesh(*meshes[templateCandidate.sourceMeshIndex], sourceCopy);
            PotentialRigidMesh targetCopy;
            PotentialRigidMesh* targetCopyPtr = nullptr;
            if (templateCandidate.targetMeshIndex == templateCandidate.sourceMeshIndex)
            {
                targetCopyPtr = &sourceCopy;
            }
            else
            {
                ClonePotentialRigidMesh(*meshes[templateCandidate.targetMeshIndex], targetCopy);
                targetCopyPtr = &targetCopy;
            }

            SetLocalVertexPositions(references, localPositions, sourceCopy, templateCandidate.sourceMeshIndex,
                targetCopyPtr, templateCandidate.targetMeshIndex);
            PotentialSurfaceMeshRegistry::UpdateDerivedKinematics(sourceCopy, false);
            if (targetCopyPtr != &sourceCopy)
            {
                PotentialSurfaceMeshRegistry::UpdateDerivedKinematics(*targetCopyPtr, false);
            }

            ResizableArray<PotentialRigidMesh*> localMeshes;
            localMeshes.SetNumberOfItems(meshes.NumberOfItems());
            for (Index i = 0; i < meshes.NumberOfItems(); i++)
            {
                localMeshes[i] = meshes[i];
            }
            localMeshes[templateCandidate.sourceMeshIndex] = &sourceCopy;
            localMeshes[templateCandidate.targetMeshIndex] = targetCopyPtr;

            PotentialContactCandidate rebuiltCandidate;
            if (!RebuildCandidateForLocalCoordinates(templateCandidate, localMeshes, settings.activationDistance, rebuiltCandidate))
            {
                energy = 0.;
                return false;
            }

            PotentialContactEvaluation scalarEvaluation;
            if (!EvaluateGCPScalarPotential(rebuiltCandidate, localMeshes, settings, scalarEvaluation))
            {
                energy = 0.;
                return false;
            }

            energy = scalarEvaluation.energy;
            return true;
        }

        Real ComputeFiniteDifferenceStep(Real coordinateValue, const PotentialModelSettings& settings)
        {
            Real scale = EXUstd::Maximum(settings.activationDistance, settings.minimumDistance);
            scale = EXUstd::Maximum(scale, fabs(coordinateValue));
            scale = EXUstd::Maximum(scale, 1.);
            return 1e-6 * scale;
        }

        bool ComputeGCPLocalDerivatives(const PotentialContactCandidate& candidate,
            const ResizableArray<PotentialRigidMesh*>& meshes, const PotentialModelSettings& settings,
            PotentialContactEvaluation& evaluation)
        {
            ResizableArray<PotentialLocalVertexReference> references;
            CollectGCPLocalVertexReferences(candidate, meshes, references);
            if (references.NumberOfItems() == 0)
            {
                return false;
            }

            ResizableVector localPositions;
            localPositions.SetNumberOfItems(3 * references.NumberOfItems());
            for (Index i = 0; i < references.NumberOfItems(); i++)
            {
                const auto& reference = references[i];
                CHECKandTHROW(meshes.IsValidIndex(reference.meshIndex) &&
                    meshes[reference.meshIndex]->vertexKinematics.IsValidIndex(reference.vertexIndex),
                    "PotentialContact::ComputeGCPLocalDerivatives: invalid local reference");
                const Vector3D& position = meshes[reference.meshIndex]->vertexKinematics[reference.vertexIndex].position;
                localPositions[3 * i] = position[0];
                localPositions[3 * i + 1] = position[1];
                localPositions[3 * i + 2] = position[2];
            }

            const Index coordinateCount = localPositions.NumberOfItems();
            if (coordinateCount == 0)
            {
                return false;
            }

            Real baseEnergy = evaluation.energy;
            evaluation.localVertexReferences = references;
            evaluation.localGradient.SetNumberOfItems(coordinateCount);
            evaluation.localGradient.SetAll(0.);

            for (Index i = 0; i < coordinateCount; i++)
            {
                Real step = ComputeFiniteDifferenceStep(localPositions[i], settings);
                ResizableVector xPlus(localPositions);
                ResizableVector xMinus(localPositions);
                xPlus[i] += step;
                xMinus[i] -= step;

                Real energyPlus = 0.;
                Real energyMinus = 0.;
                EvaluateGCPEnergyAtLocalCoordinates(candidate, meshes, settings, references, xPlus, energyPlus);
                EvaluateGCPEnergyAtLocalCoordinates(candidate, meshes, settings, references, xMinus, energyMinus);
                evaluation.localGradient[i] = (energyPlus - energyMinus) / (2. * step);
            }

            evaluation.hasLocalDerivatives = true;
            if (settings.useGaussNewtonHessian)
            {
                evaluation.hasLocalHessian = false;
                evaluation.localHessian.SetNumberOfRowsAndColumns(0, 0);
                return true;
            }

            evaluation.localHessian.SetNumberOfRowsAndColumns(coordinateCount, coordinateCount);
            evaluation.localHessian.SetAll(0.);
            for (Index i = 0; i < coordinateCount; i++)
            {
                Real stepI = ComputeFiniteDifferenceStep(localPositions[i], settings);

                ResizableVector xPlus(localPositions);
                ResizableVector xMinus(localPositions);
                xPlus[i] += stepI;
                xMinus[i] -= stepI;
                Real energyPlus = 0.;
                Real energyMinus = 0.;
                EvaluateGCPEnergyAtLocalCoordinates(candidate, meshes, settings, references, xPlus, energyPlus);
                EvaluateGCPEnergyAtLocalCoordinates(candidate, meshes, settings, references, xMinus, energyMinus);

                evaluation.localHessian(i, i) = (energyPlus - 2. * baseEnergy + energyMinus) / (stepI * stepI);
                for (Index j = i + 1; j < coordinateCount; j++)
                {
                    Real stepJ = ComputeFiniteDifferenceStep(localPositions[j], settings);
                    ResizableVector xPP(localPositions);
                    ResizableVector xPM(localPositions);
                    ResizableVector xMP(localPositions);
                    ResizableVector xMM(localPositions);
                    xPP[i] += stepI; xPP[j] += stepJ;
                    xPM[i] += stepI; xPM[j] -= stepJ;
                    xMP[i] -= stepI; xMP[j] += stepJ;
                    xMM[i] -= stepI; xMM[j] -= stepJ;

                    Real ePP = 0.;
                    Real ePM = 0.;
                    Real eMP = 0.;
                    Real eMM = 0.;
                    EvaluateGCPEnergyAtLocalCoordinates(candidate, meshes, settings, references, xPP, ePP);
                    EvaluateGCPEnergyAtLocalCoordinates(candidate, meshes, settings, references, xPM, ePM);
                    EvaluateGCPEnergyAtLocalCoordinates(candidate, meshes, settings, references, xMP, eMP);
                    EvaluateGCPEnergyAtLocalCoordinates(candidate, meshes, settings, references, xMM, eMM);

                    Real mixedEntry = (ePP - ePM - eMP + eMM) / (4. * stepI * stepJ);
                    evaluation.localHessian(i, j) = mixedEntry;
                    evaluation.localHessian(j, i) = mixedEntry;
                }
            }
            evaluation.hasLocalHessian = true;
            return true;
        }

        bool EvaluateIPCStyleBarrier(PotentialModelType modelType, Real distance,
            const PotentialModelSettings& settings, PotentialContactEvaluation& evaluation)
        {
            evaluation.Reset();
            const Real activationDistance = settings.activationDistance;
            const Real minimumDistance = settings.minimumDistance;
            if (activationDistance <= 0.)
            {
                return false;
            }

            Real distanceClamped = EXUstd::Maximum(distance, minimumDistance * kSmallDistanceFactor);
            const Real barrierActivationArgument = (2. * minimumDistance + activationDistance) * activationDistance;
            const Real barrierArgument = distanceClamped * distanceClamped - minimumDistance * minimumDistance;
            if (barrierArgument >= barrierActivationArgument)
            {
                return false;
            }

            evaluation.modelType = modelType;
            evaluation.energy = settings.stiffness * ClampedLogBarrier(barrierArgument, barrierActivationArgument);
            Real barrierDerivative = ClampedLogBarrierDerivative(barrierArgument, barrierActivationArgument);
            Real barrierSecondDerivative = ClampedLogBarrierSecondDerivative(barrierArgument, barrierActivationArgument);
            evaluation.derivativeWRTDistance = settings.stiffness * barrierDerivative * 2. * distanceClamped;
            evaluation.secondDerivativeWRTDistance = settings.stiffness *
                (barrierSecondDerivative * 4. * distanceClamped * distanceClamped + barrierDerivative * 2.);
            evaluation.distanceClamped = distanceClamped;
            evaluation.effectiveActivationDistance = minimumDistance + activationDistance;
            evaluation.active = true;
            return true;
        }
    }

    bool IPCNormalPotential::Evaluate(const PotentialContactCandidate& candidate,
        const ResizableArray<PotentialRigidMesh*>& meshes, const PotentialModelSettings& settings,
        PotentialContactEvaluation& evaluation)
    {
        return EvaluateIPCStyleBarrier(PotentialModelType::IPC, candidate.distance, settings, evaluation);
    }

    bool GCPNormalPotential::Evaluate(const PotentialContactCandidate& candidate,
        const ResizableArray<PotentialRigidMesh*>& meshes, const PotentialModelSettings& settings,
        PotentialContactEvaluation& evaluation)
    {
        if (!EvaluateGCPScalarPotential(candidate, meshes, settings, evaluation))
        {
            return false;
        }
        ComputeGCPLocalDerivatives(candidate, meshes, settings, evaluation);
        return true;
    }

    bool OGCNormalPotential::Evaluate(const PotentialContactCandidate& candidate,
        const ResizableArray<PotentialRigidMesh*>& meshes, const PotentialModelSettings& settings,
        PotentialContactEvaluation& evaluation)
    {
        // OGC in IPC Toolkit keeps the barrier potential and changes the collision
        // pipeline / step controller instead of introducing a distinct normal energy.
        return EvaluateIPCStyleBarrier(PotentialModelType::OGC, candidate.distance, settings, evaluation);
    }

    bool EvaluateNormalPotential(const PotentialContactCandidate& candidate,
        const ResizableArray<PotentialRigidMesh*>& meshes, const PotentialModelSettings& settings,
        PotentialContactEvaluation& evaluation)
    {
        if (settings.modelType == PotentialModelType::IPC)
        {
            return IPCNormalPotential::Evaluate(candidate, meshes, settings, evaluation);
        }
        if (settings.modelType == PotentialModelType::OGC)
        {
            return OGCNormalPotential::Evaluate(candidate, meshes, settings, evaluation);
        }
        return GCPNormalPotential::Evaluate(candidate, meshes, settings, evaluation);
    }
}
