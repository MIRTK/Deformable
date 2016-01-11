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

#include <mirtkImplicitSurfaceSpringForce.h>

#include <mirtkMath.h>
#include <mirtkMemory.h>
#include <mirtkParallel.h>
#include <mirtkProfiling.h>
#include <mirtkVector.h>
#include <mirtkMatrix.h>

#include <vtkMath.h>
#include <vtkPoints.h>
#include <vtkDataArray.h>
#include <vtkFloatArray.h>


namespace mirtk {


// =============================================================================
// Auxiliary functions
// =============================================================================

namespace ImplicitSurfaceSpringForceUtils {

const double distance_delta_sigma2 = .25 * .25;

// -----------------------------------------------------------------------------
// Typedefs
typedef RegisteredPointSet::NodeNeighbors        NodeNeighbors;
typedef ImplicitSurfaceSpringForce::GradientType NodeGradient;

// -----------------------------------------------------------------------------
/// Evaluate error of quadratic fit in normal direction
struct ComputeErrorOfQuadraticFit
{
  vtkPoints           *_Points;
  vtkDataArray        *_Normals;
  vtkDataArray        *_Distances;
  const NodeNeighbors *_Neighbors;
  vtkDataArray        *_Residuals;

  void operator ()(const blocked_range<int> &ptIds) const
  {
    int       numNbrPts;
    const int *nbrPtIds;
    double     c[3], p[3], n[3], e[3], b, d, delta, w, wsum;

    Vector h; // Distances of neighbors to tangent plane
    Matrix r; // Squared radial distances of neighbors to central node

    for (int ptId = ptIds.begin(); ptId != ptIds.end(); ++ptId) {
      b = .0;
      _Neighbors->GetConnectedPoints(ptId, numNbrPts, nbrPtIds);
      if (numNbrPts > 0) {
        _Points ->GetPoint(ptId, c);
        _Normals->GetTuple(ptId, n);
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
        wsum = .0;
        d = abs(_Distances->GetComponent(ptId, 0));
        for (int i = 0; i < numNbrPts; ++i) {
          delta = (abs(_Distances->GetComponent(nbrPtIds[i], 0)) - d) / (d + 1e-6);
          wsum += (w = exp(-.5 * delta * delta / distance_delta_sigma2));
          b += w * r(1, i) * h(i);
        }
        if (wsum) b /= wsum;
      }
      _Residuals->SetComponent(ptId, 0, b);
    }
  }
};

// -----------------------------------------------------------------------------
/// Evaluate energy of quadratic curvature term
struct ComputeQuadraticCurvatureEnergy
{
  vtkPoints    *_Points;
  vtkDataArray *_Residuals;
  double        _Offset;
  double        _Sum;

  ComputeQuadraticCurvatureEnergy() : _Sum(.0) {}

  ComputeQuadraticCurvatureEnergy(const ComputeQuadraticCurvatureEnergy &other, split)
  :
    _Points(other._Points),
    _Residuals(other._Residuals),
    _Offset(other._Offset),
    _Sum(.0)
  {}

  void join(const ComputeQuadraticCurvatureEnergy &other)
  {
    _Sum += other._Sum;
  }

  void operator ()(const blocked_range<int> &ptIds)
  {
    double b;
    for (int ptId = ptIds.begin(); ptId != ptIds.end(); ++ptId) {
      b = _Residuals->GetComponent(ptId, 0);
      _Sum += b * b;
    }
  }
};

// -----------------------------------------------------------------------------
/// Evaluate gradient of quadratic curvature term (i.e., negative force)
struct ComputeQuadraticCurvatureGradient
{
  typedef ImplicitSurfaceSpringForce::GradientType GradientType;

  vtkDataArray *_Status;
  vtkDataArray *_Normals;
  vtkDataArray *_Residuals;
  GradientType *_Gradient;

  void operator ()(const blocked_range<int> &ptIds) const
  {
    double n[3], b;
    for (int ptId = ptIds.begin(); ptId != ptIds.end(); ++ptId) {
      if (_Status && _Status->GetComponent(ptId, 0) == .0) continue;
      _Normals->GetTuple(ptId, n);
      b = _Residuals->GetComponent(ptId, 0);
      _Gradient[ptId] = -b * GradientType(n[0], n[1], n[2]);
    }
  }
};

// -----------------------------------------------------------------------------
/// Evaluate gradient of vanishing spring force term (i.e., negative force)
struct ComputeSpringForceGradient
{
  vtkPoints           *_Points;
  vtkDataArray        *_Status;
  vtkDataArray        *_Normals;
  vtkDataArray        *_Distances;
  const NodeNeighbors *_Neighbors;
  NodeGradient        *_Gradient;

  void operator ()(const blocked_range<int> &ptIds) const
  {
    int        numNbrPts;
    const int *nbrPtIds;
    double     c[3], p[3], d, delta, w, wsum;

    NodeGradient *g = _Gradient + ptIds.begin();
    for (int ptId = ptIds.begin(); ptId != ptIds.end(); ++ptId, ++g) {
      if (_Status && _Status->GetComponent(ptId, 0) == .0) continue;
      _Neighbors->GetConnectedPoints(ptId, numNbrPts, nbrPtIds);
      if (numNbrPts > 0) {
        wsum = .0;
        _Points->GetPoint(ptId, c);
        d = abs(_Distances->GetComponent(ptId, 0));
        for (int i = 0; i < numNbrPts; ++i) {
          _Points->GetPoint(nbrPtIds[i], p);
          delta = (abs(_Distances->GetComponent(nbrPtIds[i], 0)) - d) / (d + 1e-6);
          wsum += (w = exp(-.5 * delta * delta / distance_delta_sigma2));
          g->_x += w * (c[0] - p[0]);
          g->_y += w * (c[1] - p[1]);
          g->_z += w * (c[2] - p[2]);
        }
        if (wsum) (*g) /= wsum;
      }
    }
  }
};

// -----------------------------------------------------------------------------
/// Scale spring force such that it vanishes close to the isosurface
/// and it only deforms the nodes towards this isosurface, not away from it
struct ScaleForce
{
  ImplicitSurfaceForce *_Force;
  vtkPoints            *_Points;
  vtkDataArray         *_Normals;
  double                _Scale;
  double                _MaxDistance;
  NodeGradient         *_Gradient;

  void operator ()(const blocked_range<int> &ptIds) const
  {
    double p[3], n[3], d, dp;
    NodeGradient *g = _Gradient + ptIds.begin();
    for (int ptId = ptIds.begin(); ptId != ptIds.end(); ++ptId, ++g) {
      if (g->_x || g->_y || g->_z) {
        _Points ->GetPoint(ptId, p);
        _Normals->GetTuple(ptId, n);
        d  = _Force->Distance(p, n);
        dp = g->_x * n[0] + g->_y * n[1] + g->_z * n[2];
        if (d * dp < .0) {
          (*g) = .0;
        } else if (_MaxDistance > .0) { // e.g. _Force->DistanceMeasure() == DM_Normal
          d /= _MaxDistance;
          (*g) *= 1.0 - exp(- _Scale * (d * d));
        } else { // _Force->DistanceMeasure() == DM_Minimum
          if (abs(d) < 1.0) (*g) *= 1.0 - exp(- _Scale * (d * d));
        }
      }
    }
  }
};

// -----------------------------------------------------------------------------
/// Weight normal and tangential component
struct WeightComponents
{
  vtkDataArray *_Normals;
  NodeGradient *_Gradient;
  double        _NormalWeight;
  double        _TangentialWeight;

  void operator ()(const blocked_range<int> &ptIds) const
  {
    double n[3], nc;
    NodeGradient *g = _Gradient + ptIds.begin();
    for (int ptId = ptIds.begin(); ptId != ptIds.end(); ++ptId, ++g) {
      if (g->_x || g->_y || g->_z) {
        _Normals->GetTuple(ptId, n);
        nc = g->_x * n[0] + g->_y * n[1] + g->_z * n[2];
        vtkMath::MultiplyScalar(n, nc);
        g->_x = _TangentialWeight * (g->_x - n[0]) + _NormalWeight * n[0];
        g->_y = _TangentialWeight * (g->_y - n[1]) + _NormalWeight * n[1];
        g->_z = _TangentialWeight * (g->_z - n[2]) + _NormalWeight * n[2];
      }
    }
  }
};


} // namespace ImplicitSurfaceSpringForceUtils
using namespace ImplicitSurfaceSpringForceUtils;

// =============================================================================
// Construction/Destruction
// =============================================================================

// -----------------------------------------------------------------------------
ImplicitSurfaceSpringForce::ImplicitSurfaceSpringForce(const char *name, double weight)
:
  ImplicitSurfaceForce(name, weight),
  _Sigma(1.0)
{
  _MaxDistance = 5.0;
}

// -----------------------------------------------------------------------------
void ImplicitSurfaceSpringForce::CopyAttributes(const ImplicitSurfaceSpringForce &other)
{
  _Sigma = other._Sigma;
}

// -----------------------------------------------------------------------------
ImplicitSurfaceSpringForce::ImplicitSurfaceSpringForce(const ImplicitSurfaceSpringForce &other)
:
  ImplicitSurfaceForce(other)
{
  CopyAttributes(other);
}

// -----------------------------------------------------------------------------
ImplicitSurfaceSpringForce &ImplicitSurfaceSpringForce::operator =(const ImplicitSurfaceSpringForce &other)
{
  if (this != &other) {
    ImplicitSurfaceSpringForce::operator =(other);
    CopyAttributes(other);
  }
  return *this;
}

// -----------------------------------------------------------------------------
ImplicitSurfaceSpringForce::~ImplicitSurfaceSpringForce()
{
}

// -----------------------------------------------------------------------------
void ImplicitSurfaceSpringForce::Initialize()
{
  ImplicitSurfaceForce::Initialize();
  InitializeDistances();
}

// =============================================================================
// Configuration
// =============================================================================

// -----------------------------------------------------------------------------
bool ImplicitSurfaceSpringForce::SetWithoutPrefix(const char *param, const char *value)
{
  if (strcmp(param, "Sigma") == 0) {
    return FromString(value, _Sigma) && _Sigma > .0;
  }
  return ImplicitSurfaceForce::SetWithoutPrefix(param, value);
}

// -----------------------------------------------------------------------------
ParameterList ImplicitSurfaceSpringForce::Parameter() const
{
  ParameterList params = ImplicitSurfaceForce::Parameter();
  InsertWithPrefix(params, "Sigma",  _Sigma);
  return params;
}

// =============================================================================
// Evaluation
// =============================================================================

// -----------------------------------------------------------------------------
void ImplicitSurfaceSpringForce::Update(bool gradient)
{
  if (_NumberOfPoints == 0) return;

  // Update base class
  ImplicitSurfaceForce::Update(gradient);

  // Compute surface distances in normal direction
  // TODO: Store modified flag for "global" point data arrays such that
  //       the values of these global arrays are only re-computed when needed.
  UpdateDistances();

  // Compute normal coefficients of quadratic fit
  MIRTK_START_TIMING();
  if (!_Residuals) _Residuals = vtkSmartPointer<vtkFloatArray>::New();
  _Residuals->SetNumberOfTuples(_NumberOfPoints);
  ComputeErrorOfQuadraticFit fit;
  fit._Points    = _PointSet->SurfacePoints();
  fit._Normals   = _PointSet->SurfaceNormals();
  fit._Distances = Distances();
  fit._Neighbors = _PointSet->SurfaceNeighbors();
  fit._Residuals = _Residuals;
  parallel_for(blocked_range<int>(0, _NumberOfPoints), fit);
  MIRTK_DEBUG_TIMING(3, "quadratic curvature fitting");
}

// -----------------------------------------------------------------------------
double ImplicitSurfaceSpringForce::Evaluate()
{
  if (_NumberOfPoints == 0) return .0;

  ComputeQuadraticCurvatureEnergy eval;
  eval._Points    = _PointSet->SurfacePoints();
  eval._Residuals = _Residuals;
  eval._Offset    = _Offset;
  parallel_reduce(blocked_range<int>(0, _NumberOfPoints), eval);

  return eval._Sum / _NumberOfPoints;
}

// -----------------------------------------------------------------------------
void ImplicitSurfaceSpringForce::EvaluateGradient(double *gradient, double step, double weight)
{
  if (_NumberOfPoints == 0) return;

  memset(_Gradient, 0, _NumberOfPoints * sizeof(GradientType));

#if 1
  ComputeQuadraticCurvatureGradient eval;
  eval._Status    = _PointSet->SurfaceStatus();
  eval._Normals   = _PointSet->SurfaceNormals();
  eval._Residuals = _Residuals;
  eval._Gradient  = _Gradient;
  parallel_for(blocked_range<int>(0, _NumberOfPoints), eval);
#else
  ComputeSpringForceGradient eval;
  eval._Points    = _PointSet->SurfacePoints();
  eval._Status    = _PointSet->SurfaceStatus();
  eval._Normals   = _PointSet->SurfaceNormals();
  eval._Distances = Distances();
  eval._Neighbors = _PointSet->SurfaceNeighbors();
  eval._Gradient  = _Gradient;
  parallel_for(blocked_range<int>(0, _NumberOfPoints), eval);
#endif

  ScaleForce scale;
  scale._Force       = this;
  scale._Points      = _PointSet->SurfacePoints();
  scale._Normals     = _PointSet->SurfaceNormals();
  scale._Scale       = .5 / (_Sigma * _Sigma);
  scale._MaxDistance = (_DistanceMeasure == DM_Minimum ? .0 : _MaxDistance);
  scale._Gradient    = _Gradient;
  parallel_for(blocked_range<int>(0, _NumberOfPoints), scale);

#if 0
  WeightComponents mul;
  mul._Normals          = _PointSet->SurfaceNormals();
  mul._NormalWeight     = .67;
  mul._TangentialWeight = .33;
  mul._Gradient         = _Gradient;
  parallel_for(blocked_range<int>(0, _NumberOfPoints), mul);
#endif

  ImplicitSurfaceForce::EvaluateGradient(gradient, step, weight / _NumberOfPoints);
}


} // namespace mirtk
