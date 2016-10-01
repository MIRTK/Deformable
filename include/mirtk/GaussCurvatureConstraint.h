/*
 * Medical Image Registration ToolKit (MIRTK)
 *
 * Copyright 2016 Imperial College London
 * Copyright 2016 Andreas Schuh
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

#ifndef MIRTK_GaussCurvatureConstraint_H
#define MIRTK_GaussCurvatureConstraint_H

#include "mirtk/SurfaceConstraint.h"

#include "vtkSmartPointer.h"
#include "vtkDataArray.h"


namespace mirtk {


/**
 * This force attempts to remove saddle points with negative Gauss curvature
 */
class GaussCurvatureConstraint : public SurfaceConstraint
{
  mirtkEnergyTermMacro(GaussCurvatureConstraint, EM_GaussCurvature);

  // ---------------------------------------------------------------------------
  // Types

public:

  /// Enumeration of possible actions for reducing negative Gauss curvature
  enum Action
  {
    DefaultAction,
    NoAction,
    Deflate,
    Inflate
  };

  // ---------------------------------------------------------------------------
  // Attributes

private:

  /// Minimum Gauss curvature threshold
  mirtkPublicAttributeMacro(double, MinGaussCurvature);

  /// Maximum Gauss curvature threshold
  mirtkPublicAttributeMacro(double, MaxGaussCurvature);

  /// Action to take for negative Gauss curvature points
  mirtkPublicAttributeMacro(Action, NegativeGaussCurvatureAction);

  /// Action to take for positive Gauss curvature points
  mirtkPublicAttributeMacro(Action, PositiveGaussCurvatureAction);

  /// Whether to scale force proportional to mean curvature
  mirtkPublicAttributeMacro(bool, UseMeanCurvature);

  /// Copy attributes of this class from another instance
  void CopyAttributes(const GaussCurvatureConstraint &);

  // ---------------------------------------------------------------------------
  // Construction/Destruction

public:

  /// Constructor
  GaussCurvatureConstraint(const char * = "", double = 1.0);

  /// Copy constructor
  GaussCurvatureConstraint(const GaussCurvatureConstraint &);

  /// Assignment operator
  GaussCurvatureConstraint &operator =(const GaussCurvatureConstraint &);

  /// Destructor
  virtual ~GaussCurvatureConstraint();

  // ---------------------------------------------------------------------------
  // Evaluation

  /// Initialize force term once input and parameters have been set
  virtual void Initialize();

  /// Update internal force data structures
  virtual void Update(bool);

protected:

  /// Evaluate energy of internal force term
  virtual double Evaluate();

  /// Evaluate internal force w.r.t. transformation parameters or surface nodes
  virtual void EvaluateGradient(double *, double, double);

};

////////////////////////////////////////////////////////////////////////////////
// Enum <-> string conversion
////////////////////////////////////////////////////////////////////////////////

template <> bool FromString(const char *, enum GaussCurvatureConstraint::Action &);
template <> string ToString(const enum GaussCurvatureConstraint::Action &, int, char, bool);


} // namespace mirtk

#endif // MIRTK_GaussCurvatureConstraint_H
