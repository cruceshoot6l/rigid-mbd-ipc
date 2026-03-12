/** ***********************************************************************************************
* @brief		Data structures for potential-based rigid contact integration
* @details		Week-3 adds minimal vertex-face candidates and barrier evaluation helpers
*
************************************************************************************************ */
#ifndef POTENTIALCONTACT__H
#define POTENTIALCONTACT__H

#include "Linalg/BasicLinalg.h"
#include "Linalg/SearchTree.h"

namespace PotentialContact
{
	struct EvaluationSummary
	{
		Index numberOfCandidates;
		Real minimumDistance;
		bool usedGaussNewtonHessian;

		EvaluationSummary()
		{
			Reset();
		}

		void Reset()
		{
			numberOfCandidates = 0;
			minimumDistance = EXUstd::MAXREAL;
			usedGaussNewtonHessian = true;
		}
	};

	enum class PotentialCandidateType
	{
		VertexFace = 0
	};

	struct PotentialMeshVertexLocal
	{
		Vector3D xLocal;

		PotentialMeshVertexLocal()
		{
			xLocal.SetAll(0.);
		}
	};

	struct PotentialMeshTriangle
	{
		Index3 vertices;

		PotentialMeshTriangle()
		{
			vertices = Index3({ 0,0,0 });
		}
	};

	struct PotentialSurfacePointKinematics
	{
		Vector3D position;
		Vector3D velocity;
		ResizableMatrix positionJacobian;

		PotentialSurfacePointKinematics()
		{
			Reset(false);
		}

		void Reset(bool freeMemory)
		{
			position.SetAll(0.);
			velocity.SetAll(0.);
			if (freeMemory) { positionJacobian.Flush(); }
			else { positionJacobian.SetNumberOfRowsAndColumns(0, 0); }
		}
	};

	struct PotentialRigidMeshState
	{
		Vector3D markerPosition;
		Matrix3D markerOrientation;
		Vector3D markerVelocity;
		Vector3D markerAngularVelocityLocal;
		ResizableMatrix markerPositionJacobian;
		ResizableMatrix markerRotationJacobian;
		ArrayIndex markerLTG;

		PotentialRigidMeshState()
		{
			Reset(false);
		}

		void Reset(bool freeMemory)
		{
			markerPosition.SetAll(0.);
			markerOrientation = EXUmath::unitMatrix3D;
			markerVelocity.SetAll(0.);
			markerAngularVelocityLocal.SetAll(0.);
			if (freeMemory)
			{
				markerPositionJacobian.Flush();
				markerRotationJacobian.Flush();
				markerLTG.Flush();
			}
			else
			{
				markerPositionJacobian.SetNumberOfRowsAndColumns(0, 0);
				markerRotationJacobian.SetNumberOfRowsAndColumns(0, 0);
				markerLTG.SetNumberOfItems(0);
			}
		}
	};

	struct PotentialRigidMesh
	{
		Index markerIndex;
		Index frictionMaterialIndex;
		bool staticMesh;
		ResizableArray<PotentialMeshVertexLocal> verticesLocal;
		ResizableArray<PotentialMeshTriangle> triangles;
		ResizableArray<PotentialSurfacePointKinematics> vertexKinematics;
		ResizableArray<Box3D> triangleAABBs;
		PotentialRigidMeshState state;

		PotentialRigidMesh()
		{
			Reset(false);
		}

		void Reset(bool freeMemory)
		{
			markerIndex = EXUstd::InvalidIndex;
			frictionMaterialIndex = 0;
			staticMesh = false;
			if (freeMemory)
			{
				verticesLocal.Flush();
				triangles.Flush();
				vertexKinematics.Flush();
				triangleAABBs.Flush();
			}
			else
			{
				verticesLocal.SetNumberOfItems(0);
				triangles.SetNumberOfItems(0);
				vertexKinematics.SetNumberOfItems(0);
				triangleAABBs.SetNumberOfItems(0);
			}
			state.Reset(freeMemory);
		}

		Index NumberOfVertices() const { return verticesLocal.NumberOfItems(); }
		Index NumberOfTriangles() const { return triangles.NumberOfItems(); }
	};

	struct PotentialContactCandidate
	{
		PotentialCandidateType type;
		Index sourceMeshIndex;
		Index targetMeshIndex;
		Index vertexIndex;
		Index triangleIndex;
		Vector3D closestPoint;
		Vector3D normal;
		Vector3D barycentricCoordinates;
		Real distance;

		PotentialContactCandidate()
		{
			Reset();
		}

		void Reset()
		{
			type = PotentialCandidateType::VertexFace;
			sourceMeshIndex = EXUstd::InvalidIndex;
			targetMeshIndex = EXUstd::InvalidIndex;
			vertexIndex = EXUstd::InvalidIndex;
			triangleIndex = EXUstd::InvalidIndex;
			closestPoint.SetAll(0.);
			normal.SetAll(0.);
			barycentricCoordinates.SetAll(0.);
			distance = EXUstd::MAXREAL;
		}
	};

	struct PotentialContactEvaluation
	{
		Real energy;
		Real derivativeWRTDistance;
		Real distanceClamped;
		bool active;

		PotentialContactEvaluation()
		{
			Reset();
		}

		void Reset()
		{
			energy = 0.;
			derivativeWRTDistance = 0.;
			distanceClamped = 0.;
			active = false;
		}
	};

	Vector3D ComputeVertexPosition(const PotentialRigidMeshState& state, const Vector3D& localPosition);
	Vector3D ComputeVertexVelocity(const PotentialRigidMeshState& state, const Vector3D& localPosition);
	void ComputeVertexPositionJacobian(const PotentialRigidMeshState& state, const Vector3D& localPosition, ResizableMatrix& positionJacobian);
	void UpdateTriangleAABBs(PotentialRigidMesh& mesh);
	Box3D ExpandedBox(const Box3D& box, Real margin);
	bool ComputeVertexFaceCandidate(const PotentialRigidMesh& sourceMesh, Index sourceMeshIndex, Index vertexIndex,
		const PotentialRigidMesh& targetMesh, Index targetMeshIndex, Index triangleIndex, Real activationDistance,
		PotentialContactCandidate& candidate);
	bool EvaluateBarrierPotential(const PotentialContactCandidate& candidate, Real activationDistance,
		Real minimumDistance, Real stiffness, PotentialContactEvaluation& evaluation);
	void LocalVerticesToMatrix(const ResizableArray<PotentialMeshVertexLocal>& vertices, Matrix& matrix);
	void CurrentVerticesToMatrix(const ResizableArray<PotentialSurfacePointKinematics>& vertices, Matrix& matrix);
	void CurrentVelocitiesToMatrix(const ResizableArray<PotentialSurfacePointKinematics>& vertices, Matrix& matrix);
	void TrianglesToMatrixI(const ResizableArray<PotentialMeshTriangle>& triangles, MatrixI& matrix);
	void TriangleBoxesToMatrices(const ResizableArray<Box3D>& boxes, Matrix& pMin, Matrix& pMax);

	inline STDstring GetModuleVersionTag()
	{
		return "PotentialContact-Week3";
	}
}

#endif
