/** ***********************************************************************************************
* @brief		Utility implementation for potential-based rigid contact integration
*
************************************************************************************************ */

#include "System/PotentialContact.h"
#include "Linalg/Geometry.h"
#include "Utilities/RigidBodyMath.h"

namespace PotentialContact
{
	Vector3D ComputeVertexPosition(const PotentialRigidMeshState& state, const Vector3D& localPosition)
	{
		return state.markerPosition + state.markerOrientation * localPosition;
	}

	Vector3D ComputeVertexVelocity(const PotentialRigidMeshState& state, const Vector3D& localPosition)
	{
		return state.markerVelocity + state.markerOrientation * state.markerAngularVelocityLocal.CrossProduct(localPosition);
	}

	void ComputeVertexPositionJacobian(const PotentialRigidMeshState& state, const Vector3D& localPosition, ResizableMatrix& positionJacobian)
	{
		positionJacobian.CopyFrom(state.markerPositionJacobian);
		if (state.markerRotationJacobian.NumberOfColumns() == 0)
		{
			return;
		}

		ResizableMatrix temp;
		Vector3D globalOffset = state.markerOrientation * localPosition;
		EXUmath::MultMatrixMatrixTemplate<Matrix3D, ResizableMatrix, ResizableMatrix>(
			RigidBodyMath::Vector2SkewMatrix(-globalOffset), state.markerRotationJacobian, temp);

		CHECKandTHROW(temp.NumberOfRows() == positionJacobian.NumberOfRows() &&
			temp.NumberOfColumns() == positionJacobian.NumberOfColumns(),
			"PotentialContact::ComputeVertexPositionJacobian: inconsistent jacobian dimensions");

		for (Index i = 0; i < positionJacobian.NumberOfRows(); i++)
		{
			for (Index j = 0; j < positionJacobian.NumberOfColumns(); j++)
			{
				positionJacobian(i, j) += temp(i, j);
			}
		}
	}

	void UpdateTriangleAABBs(PotentialRigidMesh& mesh)
	{
		mesh.triangleAABBs.SetNumberOfItems(mesh.triangles.NumberOfItems());
		for (Index i = 0; i < mesh.triangles.NumberOfItems(); i++)
		{
			const PotentialMeshTriangle& triangle = mesh.triangles[i];
			Box3D box;
			for (Index j = 0; j < triangle.vertices.NumberOfItems(); j++)
			{
				CHECKandTHROW(mesh.vertexKinematics.IsValidIndex(triangle.vertices[j]),
					"PotentialContact::UpdateTriangleAABBs: invalid triangle vertex index");
				box.Add(mesh.vertexKinematics[triangle.vertices[j]].position);
			}
			mesh.triangleAABBs[i] = box;
		}
	}

	Box3D ExpandedBox(const Box3D& box, Real margin)
	{
		Box3D expandedBox(box);
		expandedBox.Increase(margin);
		return expandedBox;
	}

	bool ComputeVertexFaceCandidate(const PotentialRigidMesh& sourceMesh, Index sourceMeshIndex, Index vertexIndex,
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

	bool EvaluateBarrierPotential(const PotentialContactCandidate& candidate, Real activationDistance,
		Real minimumDistance, Real stiffness, PotentialContactEvaluation& evaluation)
	{
		evaluation.Reset();
		if (activationDistance <= 0. || candidate.distance >= activationDistance)
		{
			return false;
		}

		Real distanceClamped = EXUstd::Maximum(candidate.distance, minimumDistance);
		distanceClamped = EXUstd::Minimum(distanceClamped, activationDistance - 1e-15);
		Real gap = activationDistance - distanceClamped;
		Real logTerm = log(activationDistance / distanceClamped);

		evaluation.energy = stiffness * gap * gap * logTerm;
		evaluation.derivativeWRTDistance = stiffness * (-2. * gap * logTerm - gap * gap / distanceClamped);
		evaluation.distanceClamped = distanceClamped;
		evaluation.active = true;
		return true;
	}

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
