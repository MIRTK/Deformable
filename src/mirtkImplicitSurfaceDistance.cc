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


namespace mirtk {


// Register energy term with object factory during static initialization
mirtkAutoRegisterEnergyTermMacro(ImplicitSurfaceDistance);


// =============================================================================
// Auxiliary functions
// =============================================================================

namespace ImplicitSurfaceDistanceUtils {


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
  double                _Scale;
  GradientType         *_Gradient;

  void operator ()(const blocked_range<int> &ptIds) const
  {
    double n[3], d;
    GradientType *g = _Gradient + ptIds.begin();
    for (vtkIdType ptId = ptIds.begin(); ptId != ptIds.end(); ++ptId, ++g) {
      if (_Status && _Status->GetComponent(ptId, 0) == .0) continue;
      _Normals->GetTuple(ptId, n);
      d = _Distances->GetComponent(ptId, 0);
      if (d < .0) (*g) = -GradientType(n[0], n[1], n[2]);
      else        (*g) =  GradientType(n[0], n[1], n[2]);

      if (_MaxDistance > .0) { // e.g. _Force->DistanceMeasure() == DM_Normal

        d /= _MaxDistance;
        d *= _Scale;
        d *= d;
        (*g) *= d / (1.0 + d);

      } else { // _Force->DistanceMeasure() == DM_Minimum

        d = fabs(d);
        if (d < 1.0) (*g) *= d;

      }
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
  double sum = .0;
  vtkDataArray *distances = Distances();
  for (int ptId = 0; ptId < _NumberOfPoints; ++ptId) {
    sum += abs(distances->GetComponent(ptId, 0));
  }
  return sum / _NumberOfPoints;
}

// -----------------------------------------------------------------------------
void ImplicitSurfaceDistance::EvaluateGradient(double *gradient, double step, double weight)
{
  if (_NumberOfPoints == 0) return;

  memset(_Gradient, 0, _NumberOfPoints * sizeof(GradientType));

  vtkDataArray *distances = this->Distances();

  double max_distance = .0;
  if (_DistanceMeasure != DM_Minimum) {
    data::statistic::AbsPercentile::Calculate(95, distances);
  }

  ComputeGradient eval;
  eval._Force       = this;
  eval._Points      = _PointSet->SurfacePoints();
  eval._Status      = _PointSet->SurfaceStatus();
  eval._Normals     = _PointSet->SurfaceNormals();
  eval._Distances   = distances;
  eval._MaxDistance = max_distance;
  eval._Scale       = 1.0;
  eval._Gradient    = _Gradient;
  parallel_for(blocked_range<int>(0, _NumberOfPoints), eval);

  ImplicitSurfaceForce::EvaluateGradient(gradient, step, weight / _NumberOfPoints);
}


} // namespace mirtk
