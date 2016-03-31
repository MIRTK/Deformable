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

#ifndef MIRTK_ImplicitSurfaceSpringForce_H
#define MIRTK_ImplicitSurfaceSpringForce_H

#include "mirtk/ImplicitSurfaceForce.h"

#include "vtkSmartPointer.h"
#include "vtkDataArray.h"


namespace mirtk {


/**
 * Force used to smooth sharp edges during shrink wrapping an implicit surface
 *
 * This force vanishes nearby the target isosurface defined by the discretized
 * implicit surface distance function. Further away from the isosurface, it
 * is equivalent to the QuadraticCurvatureConstraint force which aims
 * to minimize the average distance of neighboring nodes to the tangent plane
 * by moving the tangent plane in normal direction.
 */
class ImplicitSurfaceSpringForce : public ImplicitSurfaceForce
{
  mirtkEnergyTermMacro(ImplicitSurfaceSpringForce, EM_ImplicitSurfaceSpringForce);

  // ---------------------------------------------------------------------------
  // Attributes

  /// Distance sigma value used to decrease strength of force when node
  /// approaches isosurface of implicit surface distance function
  mirtkPublicAttributeMacro(double, Sigma);

  /// Residual error of quadratic fit to node distance to tangent plane
  mirtkAttributeMacro(vtkSmartPointer<vtkDataArray>, Residuals);

  /// Copy attributes of this class from another instance
  void CopyAttributes(const ImplicitSurfaceSpringForce &);

  // ---------------------------------------------------------------------------
  // Construction/Destruction
public:

  /// Constructor
  ImplicitSurfaceSpringForce(const char * = "", double = 1.0);

  /// Copy constructor
  ImplicitSurfaceSpringForce(const ImplicitSurfaceSpringForce &);

  /// Assignment operator
  ImplicitSurfaceSpringForce &operator =(const ImplicitSurfaceSpringForce &);

  /// Destructor
  virtual ~ImplicitSurfaceSpringForce();

  /// Initialize force term once input and parameters have been set
  virtual void Initialize();

  // ---------------------------------------------------------------------------
  // Configuration

protected:

  /// Set parameter value from string
  virtual bool SetWithoutPrefix(const char *, const char *);

public:

  // Import other overloads
  using ImplicitSurfaceForce::Parameter;

  /// Get parameter name/value pairs
  virtual ParameterList Parameter() const;

  // ---------------------------------------------------------------------------
  // Evaluation

  /// Update internal force data structures
  virtual void Update(bool);

protected:

  /// Evaluate external force term
  virtual double Evaluate();

  /// Evaluate external force
  virtual void EvaluateGradient(double *, double, double);

};


} // namespace mirtk

#endif // MIRTK_ImplicitSurfaceSpringForce_H
