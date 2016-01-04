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

#include <mirtkImplicitSurfaceDistance.h>

#include <mirtkMath.h>
#include <mirtkMemory.h>
#include <mirtkParallel.h>
#include <mirtkDataStatistics.h>
#include <mirtkObjectFactory.h>

#include <vtkPoints.h>
#include <vtkDataArray.h>
#include <vtkFloatArray.h>


namespace mirtk {


// Register energy term with object factory during static initialization
mirtkAutoRegisterEnergyTermMacro(ImplicitSurfaceDistance);


// =============================================================================
// Auxiliary functions
// =============================================================================

namespace ImplicitSurfaceDistanceUtils {

// -----------------------------------------------------------------------------
/// Compute mean surface distance
struct ComputeEnergy
{
  ImplicitSurfaceForce *_Force;
  vtkPoints            *_Points;
  double                _Sum;

  ComputeEnergy() : _Sum(.0) {}

  ComputeEnergy(const ComputeEnergy &other, split)
  :
    _Force(other._Force),
    _Points(other._Points),
    _Sum(.0)
  {}

  void join(const ComputeEnergy &other)
  {
    _Sum += other._Sum;
  }

  void operator ()(const blocked_range<int> &ptIds)
  {
    double p[3];
    for (int ptId = ptIds.begin(); ptId != ptIds.end(); ++ptId) {
      _Points->GetPoint(ptId, p);
      _Sum += abs(_Force->Distance(p));
    }
  }
};

// -----------------------------------------------------------------------------
/// Compute gradient of implicit surface distance force term (i.e., negative force)
struct ComputeGradient
{
  typedef ImplicitSurfaceDistance::GradientType GradientType;

  ImplicitSurfaceForce *_Force;
  vtkPoints            *_Points;
  vtkDataArray         *_Status;
  vtkDataArray         *_Normals;
  vtkDataArray         *_Distances;
  double                _MaxDistance;
  GradientType         *_Gradient;

  void operator ()(const blocked_range<int> &ptIds) const
  {
    double p[3], n[3], d, maxd;
    GradientType *g = _Gradient + ptIds.begin();
    for (vtkIdType ptId = ptIds.begin(); ptId != ptIds.end(); ++ptId, ++g) {
      if (_Status && _Status->GetComponent(ptId, 0) == .0) continue;
      _Normals->GetTuple(ptId, n);
      d = _Distances->GetComponent(ptId, 0);
      if (!fequal(d, .0)) {
        _Points->GetPoint(ptId, p);
        maxd = .33 * _Force->IntersectWithRay(p, n, -d);
        if (maxd < abs(d)) d = copysign(maxd, d);
      }
      (*g) = (d / _MaxDistance) * GradientType(n[0], n[1], n[2]);
//      d = abs(d);
//      if (d < 1.0) {
//        // scale gradient only when point is moving orthogonal to the isosurface
//        // in this case, we expect to be close to the isosurface after a step
//        // of length d along the normal direction, otherwise we are more likely
//        // moving parallel to the isosurface and thus keep our constant speed
//        p2[0] = p[0] - g->_x * d;
//        p2[1] = p[1] - g->_y * d;
//        p2[2] = p[2] - g->_z * d;
//        d2 = _Force->EvaluateDistance(p2);
//        if (abs(d2) < .1) (*g) *= d;
//      }
    }
  }
};


} // namespace ImplicitSurfaceDistanceUtils
using namespace ImplicitSurfaceDistanceUtils;

// =============================================================================
// Construction/Destruction
// =============================================================================

// -----------------------------------------------------------------------------
ImplicitSurfaceDistance::ImplicitSurfaceDistance(const char *name, double weight)
:
  ImplicitSurfaceForce(name, weight)
{
  _MaxDistance = 5.0;
}

// -----------------------------------------------------------------------------
ImplicitSurfaceDistance::ImplicitSurfaceDistance(const ImplicitSurfaceDistance &other)
:
  ImplicitSurfaceForce(other)
{
}

// -----------------------------------------------------------------------------
ImplicitSurfaceDistance &ImplicitSurfaceDistance::operator =(const ImplicitSurfaceDistance &other)
{
  ImplicitSurfaceForce::operator =(other);
  return *this;
}

// -----------------------------------------------------------------------------
ImplicitSurfaceDistance::~ImplicitSurfaceDistance()
{
}

// -----------------------------------------------------------------------------
void ImplicitSurfaceDistance::Initialize()
{
  ImplicitSurfaceForce::Initialize();
  InitializeDistances();
}

// =============================================================================
// Evaluation
// =============================================================================

// -----------------------------------------------------------------------------
void ImplicitSurfaceDistance::Update(bool gradient)
{
  ImplicitSurfaceForce::Update(gradient);
  UpdateDistances();
}

// -----------------------------------------------------------------------------
double ImplicitSurfaceDistance::Evaluate()
{
  if (_NumberOfPoints == 0) return .0;
#if 1
  double sum = .0;
  vtkDataArray *distances = Distances();
  for (int ptId = 0; ptId < _NumberOfPoints; ++ptId) {
    sum += abs(distances->GetComponent(ptId, 0));
  }
  return sum / _NumberOfPoints;
#else
  ComputeEnergy eval;
  eval._Force  = this;
  eval._Points = _PointSet->SurfacePoints();
  parallel_reduce(blocked_range<int>(0, _NumberOfPoints), eval);

  return eval._Sum / _NumberOfPoints;
#endif
}

// -----------------------------------------------------------------------------
void ImplicitSurfaceDistance::EvaluateGradient(double *gradient, double step, double weight)
{
  if (_NumberOfPoints == 0) return;

  memset(_Gradient, 0, _NumberOfPoints * sizeof(GradientType));

  vtkDataArray *distances = Distances();
  double max_distance = data::statistic::AbsPercentile::Calculate(95, distances);

  ComputeGradient eval;
  eval._Force       = this;
  eval._Points      = _PointSet->SurfacePoints();
  eval._Status      = _PointSet->SurfaceStatus();
  eval._Normals     = _PointSet->SurfaceNormals();
  eval._Distances   = distances;
  eval._MaxDistance = max_distance;
  eval._Gradient    = _Gradient;
  // Attention: VTK cell locator used by SurfaceForce::IntersectWithRay
  //            is not thread-safe. Hence, execute this loop in main thread.
  //parallel_for(blocked_range<int>(0, _NumberOfPoints), eval);
  eval(blocked_range<int>(0, _NumberOfPoints));

  _GradientAveraging        = 2;
  _AverageGradientMagnitude = true;
  _AverageSignedGradients   = false;

  ImplicitSurfaceForce::EvaluateGradient(gradient, step, weight / _NumberOfPoints);
}


} // namespace mirtk
