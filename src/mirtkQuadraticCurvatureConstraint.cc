/*
 * Medical Image Registration ToolKit (MIRTK)
 *
 * Copyright 2013-2015 Imperial College London
 * Copyright 2013-2015 Andreas Schuh
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <mirtkQuadraticCurvatureConstraint.h>

#include <mirtkMemory.h>
#include <mirtkVector.h>
#include <mirtkMatrix.h>
#include <mirtkParallel.h>
#include <mirtkProfiling.h>

#include <vtkMath.h>
#include <vtkPoints.h>
#include <vtkDataArray.h>

#include <vector>


namespace mirtk {


// =============================================================================
// Auxiliaries
// =============================================================================

namespace QuadraticCurvatureConstraintUtils {


// -----------------------------------------------------------------------------
/// Evaluate error of quadratic fit in normal direction
struct ComputeErrorOfQuadraticFit
{
  typedef RegisteredPointSet::NodeNeighbors NodeNeighbors;

  vtkPoints           *_Points;
  vtkDataArray        *_Normals;
  const NodeNeighbors *_Neighbors;
  std::vector<double> *_Residuals;

  void operator ()(const blocked_range<int> &ptIds) const
  {
    int       numNbrPts;
    const int *nbrPtIds;
    double     c[3], p[3], n[3], e[3], b;

    Vector h; // Distances of neighbors to tangent plane
    Matrix r; // Squared radial distances of neighbors to central node

    for (int ptId = ptIds.begin(); ptId != ptIds.end(); ++ptId) {
      b = .0;
      _Points   ->GetPoint(ptId, c);
      _Normals  ->GetTuple(ptId, n);
      _Neighbors->GetConnectedPoints(ptId, numNbrPts, nbrPtIds);
      if (numNbrPts > 0) {
        h.Initialize(numNbrPts);
        r.Initialize(numNbrPts, 2);
        for (int i = 0; i < numNbrPts; ++i) {
          _Points->GetPoint(nbrPtIds[i], p);
          vtkMath::Subtract(p, c, e);
          h(i)    = vtkMath::Dot(e, n);
          r(i, 0) = vtkMath::Dot(e, e) - h(i) * h(i);
          r(i, 1) = 1.0;
        }
        r.PseudoInvert();
        for (int i = 0; i < numNbrPts; ++i) {
          b += r(1, i) * h(i);
        }
      }
      (*_Residuals)[ptId] = b;
    }
  }
};

// -----------------------------------------------------------------------------
/// Evaluate energy of quadratic curvature term
struct Evaluate
{
  const std::vector<double> *_Residuals;
  double                     _Sum;

  Evaluate() : _Sum(.0) {}

  Evaluate(const Evaluate &other, split)
  :
    _Residuals(other._Residuals),
    _Sum(.0)
  {}

  void join(const Evaluate &other)
  {
    _Sum += other._Sum;
  }

  void operator ()(const blocked_range<int> &ptIds)
  {
    double b;
    for (int ptId = ptIds.begin(); ptId != ptIds.end(); ++ptId) {
      b = (*_Residuals)[ptId];
      _Sum += b * b;
    }
  }
};

// -----------------------------------------------------------------------------
/// Evaluate gradient of quadratic curvature term (i.e., negative force)
struct EvaluateGradient
{
  typedef QuadraticCurvatureConstraint::GradientType GradientType;

  vtkDataArray              *_Status;
  vtkDataArray              *_Normals;
  const std::vector<double> *_Residuals;
  GradientType              *_Gradient;

  void operator ()(const blocked_range<int> &ptIds) const
  {
    double b, n[3];
    for (int ptId = ptIds.begin(); ptId != ptIds.end(); ++ptId) {
      if (_Status && _Status->GetComponent(ptId, 0) == .0) continue;
      b = (*_Residuals)[ptId];
      _Normals->GetTuple(ptId, n);
      _Gradient[ptId] = -b * GradientType(n[0], n[1], n[2]);
    }
  }
};


} // namespace QuadraticCurvatureConstraintUtils
using namespace QuadraticCurvatureConstraintUtils;

// =============================================================================
// Construction/Destruction
// =============================================================================

// -----------------------------------------------------------------------------
QuadraticCurvatureConstraint
::QuadraticCurvatureConstraint(const char *name, double weight)
:
  SurfaceConstraint(name, weight)
{
  // QuadraticCurvatureConstraint specific prefixes
  _ParameterPrefix.push_back("Quadratic surface curvature ");
  _ParameterPrefix.push_back("Quadratic surface mesh curvature ");
  _ParameterPrefix.push_back("Quadratic mesh curvature ");
  // Alternative CurvatureConstraint prefixes
  _ParameterPrefix.push_back("Surface curvature ");
  _ParameterPrefix.push_back("Surface bending ");
  _ParameterPrefix.push_back("Mesh curvature ");
  _ParameterPrefix.push_back("Mesh bending ");
  _ParameterPrefix.push_back("Surface mesh curvature ");
  _ParameterPrefix.push_back("Surface mesh bending ");
}

// -----------------------------------------------------------------------------
void QuadraticCurvatureConstraint
::CopyAttributes(const QuadraticCurvatureConstraint &other)
{
  _Residuals = other._Residuals;
}

// -----------------------------------------------------------------------------
QuadraticCurvatureConstraint
::QuadraticCurvatureConstraint(const QuadraticCurvatureConstraint &other)
:
  SurfaceConstraint(other)
{
  CopyAttributes(other);
}

// -----------------------------------------------------------------------------
QuadraticCurvatureConstraint &QuadraticCurvatureConstraint
::operator =(const QuadraticCurvatureConstraint &other)
{
  if (this != &other) {
    SurfaceConstraint::operator =(other);
    CopyAttributes(other);
  }
  return *this;
}

// -----------------------------------------------------------------------------
QuadraticCurvatureConstraint::~QuadraticCurvatureConstraint()
{
}

// =============================================================================
// Evaluation
// =============================================================================

// -----------------------------------------------------------------------------
void QuadraticCurvatureConstraint::Update(bool gradient)
{
  if (_NumberOfPoints == 0) return;

  // Update base class
  SurfaceConstraint::Update(gradient);

  // Compute normal coefficients of quadratic fit
  MIRTK_START_TIMING();
  _Residuals.resize(_NumberOfPoints);
  ComputeErrorOfQuadraticFit fit;
  fit._Points    = _PointSet->SurfacePoints();
  fit._Normals   = _PointSet->SurfaceNormals();
  fit._Neighbors = _PointSet->SurfaceNeighbors();
  fit._Residuals = &_Residuals;
  parallel_for(blocked_range<int>(0, _NumberOfPoints), fit);
  MIRTK_DEBUG_TIMING(3, "quadratic curvature fitting");
}

// -----------------------------------------------------------------------------
double QuadraticCurvatureConstraint::Evaluate()
{
  if (_NumberOfPoints == 0) return .0;
  MIRTK_START_TIMING();
  QuadraticCurvatureConstraintUtils::Evaluate eval;
  eval._Residuals = &_Residuals;
  parallel_reduce(blocked_range<int>(0, _NumberOfPoints), eval);
  MIRTK_DEBUG_TIMING(3, "evaluation of quadratic curvature penalty");
  return eval._Sum / _NumberOfPoints;
}

// -----------------------------------------------------------------------------
void QuadraticCurvatureConstraint
::EvaluateGradient(double *gradient, double step, double weight)
{
  if (_NumberOfPoints == 0) return;

  MIRTK_START_TIMING();
  memset(_Gradient, 0, _NumberOfPoints * sizeof(GradientType));

  QuadraticCurvatureConstraintUtils::EvaluateGradient eval;
  eval._Status    = _PointSet->SurfaceStatus();
  eval._Normals   = _PointSet->SurfaceNormals();
  eval._Residuals = &_Residuals;
  eval._Gradient  = _Gradient;
  parallel_for(blocked_range<int>(0, _NumberOfPoints), eval);

  SurfaceConstraint::EvaluateGradient(gradient, step, 2.0 * weight / _NumberOfPoints);
  MIRTK_DEBUG_TIMING(3, "evaluation of quadratic curvature force");
}


} // namespace mirtk
