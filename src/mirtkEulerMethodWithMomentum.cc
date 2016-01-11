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

#include <mirtkEulerMethodWithMomentum.h>

#include <mirtkMath.h>
#include <mirtkParallel.h>
#include <mirtkDeformableSurfaceModel.h>
#include <mirtkObjectFactory.h>

#include <vtkPointData.h>
#include <vtkFloatArray.h>


namespace mirtk {


// Register energy term with object factory during static initialization
mirtkAutoRegisterOptimizerMacro(EulerMethodWithMomentum);


// =============================================================================
// Auxiliary functors
// =============================================================================

namespace EulerMethodWithMomentumUtils {


// -----------------------------------------------------------------------------
/// Compute node displacements given previous velocity, negated force and time step
class ComputeDisplacements
{
  const double *_Gradient;
  vtkDataArray *_PreviousDisplacement;
  double       *_Displacement;
  double        _Momentum;
  double        _Maximum;
  double        _StepLength;

public:

  ComputeDisplacements(double *dx, vtkDataArray *odx, const double *gradient,
                       double momentum, double max_dx, double dt, double norm)
  :
    _Gradient(gradient),
    _PreviousDisplacement(odx),
    _Displacement(dx),
    _Momentum(momentum),
    _Maximum(max_dx * max_dx),
    _StepLength(-dt / norm)
  {}

  void operator ()(const blocked_range<int> &ptIds) const
  {
    double norm;
    const double *g = _Gradient     + 3 * ptIds.begin();
    double       *d = _Displacement + 3 * ptIds.begin();
    for (int ptId = ptIds.begin(); ptId != ptIds.end(); ++ptId, g += 3, d += 3) {
      _PreviousDisplacement->GetTuple(ptId, d);
      d[0] = _StepLength * g[0] + _Momentum * d[0];
      d[1] = _StepLength * g[1] + _Momentum * d[1];
      d[2] = _StepLength * g[2] + _Momentum * d[2];
      norm = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
      if (norm > _Maximum) {
        norm = sqrt(_Maximum / norm);
        d[0] *= norm, d[1] *= norm, d[2] *= norm;
      }
      _PreviousDisplacement->SetTuple(ptId, d);
    }
  }
};


} // namespace EulerMethodWithMomentumUtils
using namespace EulerMethodWithMomentumUtils;

// =============================================================================
// Construction/Destruction
// =============================================================================

// -----------------------------------------------------------------------------
EulerMethodWithMomentum::EulerMethodWithMomentum(ObjectiveFunction *f)
:
  EulerMethod(f),
  _Momentum(.9)
{
}

// -----------------------------------------------------------------------------
void EulerMethodWithMomentum::CopyAttributes(const EulerMethodWithMomentum &other)
{
  _Momentum = other._Momentum;
}

// -----------------------------------------------------------------------------
EulerMethodWithMomentum::EulerMethodWithMomentum(const EulerMethodWithMomentum &other)
:
  EulerMethod(other)
{
  CopyAttributes(other);
}

// -----------------------------------------------------------------------------
EulerMethodWithMomentum &EulerMethodWithMomentum::operator =(const EulerMethodWithMomentum &other)
{
  if (this != &other) {
    EulerMethod::operator =(other);
    CopyAttributes(other);
  }
  return *this;
}

// -----------------------------------------------------------------------------
EulerMethodWithMomentum::~EulerMethodWithMomentum()
{
}

// =============================================================================
// Parameters
// =============================================================================

// -----------------------------------------------------------------------------
bool EulerMethodWithMomentum::Set(const char *name, const char *value)
{
  if (strcmp(name, "Deformable surface momentum") == 0) {
    return FromString(value, _Momentum) && .0 <= _Momentum && _Momentum <= 1.0;
  }
  if (strcmp(name, "Deformable surface damping") == 0) {
    double damping;
    if (!FromString(value, damping) || damping < .0 || damping > 1.0) return false;
    _Momentum = 1.0 - damping;
    return true;
  }
  return EulerMethod::Set(name, value);
}

// -----------------------------------------------------------------------------
ParameterList EulerMethodWithMomentum::Parameter() const
{
  ParameterList params = EulerMethod::Parameter();
  Insert(params, "Deformable surface momentum", _Momentum);
  return params;
}

// =============================================================================
// Execution
// =============================================================================

// -----------------------------------------------------------------------------
void EulerMethodWithMomentum::Initialize()
{
  // Initialize base class
  EulerMethod::Initialize();

  // Get model point data
  vtkPointData *modelPD = _Model->Output()->GetPointData();

  // Add point data array with initial node displacements such that these
  // are interpolated at new node positions during the remeshing
  //
  // An initial node "Displacement" can also be provided as input
  // (e.g., from a previous Euler integration with different parameters).
  vtkSmartPointer<vtkDataArray> displacement;
  displacement = modelPD->GetArray("Displacement");
  if (!displacement) {
    displacement = vtkSmartPointer<vtkFloatArray>::New();
    displacement->SetName("Displacement");
    displacement->SetNumberOfComponents(3);
    displacement->SetNumberOfTuples(_Model->NumberOfPoints());
    displacement->FillComponent(0, .0);
    displacement->FillComponent(1, .0);
    displacement->FillComponent(2, .0);
    modelPD->AddArray(displacement);
  }

  // Limit momentum factor to the interval [0, 1]
  _Momentum = max(.0, min(_Momentum, 1.0));
}

// -----------------------------------------------------------------------------
void EulerMethodWithMomentum::UpdateDisplacement()
{
  double norm   = this->GradientNorm();
  double max_dx = _MaximumDisplacement;
  if (max_dx <= .0) max_dx = _NormalizeStepLength ? _StepLength : 1.0;

  vtkPointData *modelPD      = _Model->Output()->GetPointData();
  vtkDataArray *displacement = modelPD->GetArray("Displacement");
  ComputeDisplacements eval(_Displacement, displacement, _Gradient,
                            _Momentum, max_dx, _StepLength, norm);
  parallel_for(blocked_range<int>(0, _Model->NumberOfPoints()), eval);
}


} // namespace mirtk
