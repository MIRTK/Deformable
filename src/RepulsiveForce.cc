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

#include "mirtk/RepulsiveForce.h"

#include "mirtk/Math.h"
#include "mirtk/Memory.h"
#include "mirtk/Parallel.h"
#include "mirtk/PointSetUtils.h"
#include "mirtk/ObjectFactory.h"
#include "mirtk/VtkMath.h"

#include "vtkSmartPointer.h"
#include "vtkIdList.h"
#include "vtkPoints.h"
#include "vtkAbstractPointLocator.h"
#include "vtkOctreePointLocator.h"


namespace mirtk {


// Register energy term with object factory during static initialization
mirtkAutoRegisterEnergyTermMacro(RepulsiveForce);


// =============================================================================
// Auxiliaries
// =============================================================================

namespace RepulsiveForceUtils {


// -----------------------------------------------------------------------------
/// Evaluate energy of repulsive force term
struct Evaluate
{
  typedef RegisteredPointSet::NodeNeighbors NodeNeighbors;

  vtkPoints               *_Points;
  const NodeNeighbors     *_Neighbors;
  vtkAbstractPointLocator *_Locator;
  double                   _Radius;
  int                      _Power;
  double                   _Epsilon;
  double                   _SSE;

  Evaluate() : _SSE(.0) {}

  Evaluate(const Evaluate &other, split)
  :
    _Points(other._Points),
    _Neighbors(other._Neighbors),
    _Locator(other._Locator),
    _Radius(other._Radius),
    _Power(other._Power),
    _Epsilon(other._Epsilon),
    _SSE(.0)
  {}

  void join(const Evaluate &other)
  {
    _SSE += other._SSE;
  }

  void operator ()(const blocked_range<int> &ptIds)
  {
    const int *nbrPtIds;
    int        numNbrPts, id, j, num;
    double     c[3], p[3], dist, sse;

    vtkSmartPointer<vtkIdList> ids = vtkSmartPointer<vtkIdList>::New();
    for (int ptId = ptIds.begin(); ptId != ptIds.end(); ++ptId) {
      _Points->GetPoint(ptId, c);
      _Locator->FindPointsWithinRadius(_Radius, c, ids);
      if (ids->GetNumberOfIds() > 0) {
        sse = .0, num = 0;
        _Neighbors->GetConnectedPoints(ptId, numNbrPts, nbrPtIds);
        for (vtkIdType i = 0; i < ids->GetNumberOfIds(); ++i) {
          id = static_cast<int>(ids->GetId(i));
          if (id == ptId) continue;
          for (j = 0; j < numNbrPts; ++j) {
            if (id == nbrPtIds[j]) break;
          }
          if (j < numNbrPts) continue;
          _Points->GetPoint(id, p);
          dist = sqrt(vtkMath::Distance2BetweenPoints(c, p));
          if (dist > .0) {
            sse += 1.0 / pow(dist / _Radius + _Epsilon, _Power);
            ++num;
          }
        }
        if (num > 0) _SSE += sse / num;
      }
    }
  }
};

// -----------------------------------------------------------------------------
/// Evaluate gradient of repulsive force term
struct EvaluateGradient
{
  typedef RegisteredPointSet::NodeNeighbors NodeNeighbors;
  typedef RepulsiveForce::GradientType      GradientType;

  vtkPoints               *_Points;
  vtkDataArray            *_Status;
  const NodeNeighbors     *_Neighbors;
  vtkAbstractPointLocator *_Locator;
  double                   _Radius;
  int                      _Power;
  double                   _Epsilon;
  GradientType            *_Gradient;

  void operator ()(const blocked_range<int> &ptIds) const
  {
    const int   *nbrPtIds;
    int          numNbrPts, id, j, num;
    double       c[3], p[3], dist, scale;
    GradientType gradient;

    vtkSmartPointer<vtkIdList> ids = vtkSmartPointer<vtkIdList>::New();
    for (int ptId = ptIds.begin(); ptId != ptIds.end(); ++ptId) {
      if (_Status && _Status->GetComponent(ptId, 0) == .0) continue;
      _Points->GetPoint(ptId, c);
      _Locator->FindPointsWithinRadius(_Radius, c, ids);
      if (ids->GetNumberOfIds() > 0) {
        gradient = .0, num = 0;
        _Neighbors->GetConnectedPoints(ptId, numNbrPts, nbrPtIds);
        for (vtkIdType i = 0; i < ids->GetNumberOfIds(); ++i) {
          id = static_cast<int>(ids->GetId(i));
          if (id == ptId) continue;
          for (j = 0; j < numNbrPts; ++j) {
            if (id == nbrPtIds[j]) break;
          }
          if (j < numNbrPts) continue;
          _Points->GetPoint(id, p);
          vtkMath::Subtract(c, p, p);
          dist = sqrt(vtkMath::Dot(p, p));
          if (dist > .0) {
            scale = 1.0 / (pow(dist / _Radius + _Epsilon, _Power) * dist);
            gradient._x -= scale * p[0];
            gradient._y -= scale * p[1];
            gradient._z -= scale * p[2];
            ++num;
          }
        }
        if (num > 0) _Gradient[ptId] = (gradient /= num);
      }
    }
  }
};


} // namespace RepulsiveForceUtils

// =============================================================================
// Construction/Destruction
// =============================================================================

// -----------------------------------------------------------------------------
RepulsiveForce::RepulsiveForce(const char *name, double weight)
:
  InternalForce(name, weight),
  _Radius(-1.0),
  _Power(6),
  _Magnitude(100.0),
  _Epsilon(.0)
{
  _ParameterPrefix.push_back("Repulsive force ");
  _ParameterPrefix.push_back("Node repulsion ");
}

// -----------------------------------------------------------------------------
void RepulsiveForce::CopyAttributes(const RepulsiveForce &other)
{
  _Radius    = other._Radius;
  _Power     = other._Power;
  _Magnitude = other._Magnitude;
  _Epsilon   = other._Epsilon;
  _Locator   = NULL;

  if (other._Locator) {
    _Locator.TakeReference(other._Locator->NewInstance());
    _Locator->SetDataSet(_PointSet->PointSet());
    _Locator->BuildLocator();
  }
}

// -----------------------------------------------------------------------------
RepulsiveForce::RepulsiveForce(const RepulsiveForce &other)
:
  InternalForce(other)
{
  CopyAttributes(other);
}

// -----------------------------------------------------------------------------
RepulsiveForce &RepulsiveForce::operator =(const RepulsiveForce &other)
{
  if (this != &other) {
    InternalForce::operator =(other);
    CopyAttributes(other);
  }
  return *this;
}

// -----------------------------------------------------------------------------
RepulsiveForce::~RepulsiveForce()
{
}

// -----------------------------------------------------------------------------
void RepulsiveForce::Initialize()
{
  _NumberOfPoints = 0;
  if (_Radius == .0) return;

  if (_Power <= 0) {
    cerr << "RepulsiveForce::Initialize: Power exponent must be positive" << endl;
    exit(1);
  }

  InternalForce::Initialize();

  if (_Radius < .0) {
    _Radius = abs(_Radius) * AverageEdgeLength(_PointSet->InputPointSet());
  }
  _Epsilon = pow(_Magnitude, -1.0 / _Power);

  _Locator = vtkSmartPointer<vtkOctreePointLocator>::New();
  _Locator->SetDataSet(_PointSet->PointSet());
}

// -----------------------------------------------------------------------------
void RepulsiveForce::Reinitialize()
{
  if (_Radius == .0) {
    _NumberOfPoints = 0;
    _Locator = NULL;
  } else {
    InternalForce::Reinitialize();
    _Locator = vtkSmartPointer<vtkOctreePointLocator>::New();
    _Locator->SetDataSet(_PointSet->PointSet());
  }
}

// =============================================================================
// Configuration
// =============================================================================

// -----------------------------------------------------------------------------
bool RepulsiveForce::SetWithoutPrefix(const char *name, const char *value)
{
  if (strcmp(name, "Radius") == 0) {
    double r;
    if (!FromString(value, r) || r == .0) return false;
    _Radius = r;
    return true;
  }
  if (strcmp(name, "Power") == 0 || strcmp(name, "Power exponent") == 0) {
    int p;
    if (!FromString(value, p) || p < 2) return false;
    _Power = p;
    return true;
  }
  if (strcmp(name, "Magnitude") == 0) {
    double m = .0;
    if (!FromString(value, m) || m <= .0) return false;
    _Magnitude = m;
    return true;
  }
  return PointSetForce::SetWithoutPrefix(name, value);
}

// -----------------------------------------------------------------------------
ParameterList RepulsiveForce::Parameter() const
{
  ParameterList params = PointSetForce::Parameter();
  InsertWithPrefix(params, "Radius",    _Radius);
  InsertWithPrefix(params, "Power",     _Power);
  InsertWithPrefix(params, "Magnitude", _Magnitude);
  return params;
}

// =============================================================================
// Evaluation
// =============================================================================

// -----------------------------------------------------------------------------
void RepulsiveForce::Update(bool gradient)
{
  if (_NumberOfPoints == .0) return;
  InternalForce::Update(gradient);
  if (gradient) _Locator->BuildLocator();
}

// -----------------------------------------------------------------------------
double RepulsiveForce::Evaluate()
{
  if (_NumberOfPoints == 0) return .0;
  _Locator->BuildLocator();

  RepulsiveForceUtils::Evaluate eval;
  eval._Points    = _PointSet->Points();
  eval._Neighbors = _PointSet->Neighbors();
  eval._Locator   = _Locator;
  eval._Radius    = _Radius;
  eval._Power     = _Power;
  eval._Epsilon   = _Epsilon;
  parallel_reduce(blocked_range<int>(0, _NumberOfPoints), eval);

  return eval._SSE / _NumberOfPoints;
}

// -----------------------------------------------------------------------------
void RepulsiveForce::EvaluateGradient(double *gradient, double step, double weight)
{
  if (_NumberOfPoints == 0) return;
  _Locator->BuildLocator();

  memset(_Gradient, 0, _NumberOfPoints * sizeof(GradientType));

  RepulsiveForceUtils::EvaluateGradient eval;
  eval._Points    = _PointSet->Points();
  eval._Status    = _PointSet->Status();
  eval._Neighbors = _PointSet->Neighbors();
  eval._Locator   = _Locator;
  eval._Radius    = _Radius;
  eval._Power     = _Power + 1; // power exponent of derivative
  eval._Epsilon   = _Epsilon;
  eval._Gradient  = _Gradient;
  parallel_for(blocked_range<int>(0, _NumberOfPoints), eval);

  InternalForce::EvaluateGradient(gradient, step, _Power * weight / _NumberOfPoints);
}


} // namespace mirtk
