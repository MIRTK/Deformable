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

#ifndef MIRTK_ImplicitSurfaceForce_H
#define MIRTK_ImplicitSurfaceForce_H

#include <mirtkSurfaceForce.h>

#include <mirtkLinearInterpolateImageFunction.h>
#include <mirtkFastLinearImageGradientFunction.h>

class vtkDataArray;


namespace mirtk {


/**
 * Base class of external forces based on a target implicit surface
 *
 * The input _Image of these force terms is the discrete distance function of
 * the implicit surface, e.g., a signed Euclidean distance transform of a binary
 * object mask.
 */
class ImplicitSurfaceForce : public SurfaceForce
{
  mirtkAbstractMacro(ImplicitSurfaceForce);

  // ---------------------------------------------------------------------------
  // Types
public:

  typedef GenericLinearInterpolateImageFunction<ImageType>  ImageFunction;
  typedef GenericFastLinearImageGradientFunction<ImageType> ImageGradient;

  // ---------------------------------------------------------------------------
  // Attributes
private:

  /// Signed distance offset
  mirtkPublicAttributeMacro(double, Offset);

  /// Maximum implicit surface distance considered for ray casting
  mirtkPublicAttributeMacro(double, MaxDistance);

  /// Continuous implicit surface distance function
  ImageFunction _Distance;

  /// Continuous implicit surface distance function gradient
  ImageGradient _DistanceGradient;

  /// Copy attributes of this class from another instance
  void CopyAttributes(const ImplicitSurfaceForce &);

  // ---------------------------------------------------------------------------
  // Construction/Destruction
protected:

  /// Constructor
  ImplicitSurfaceForce(const char * = "", double = 1.0);

  /// Copy constructor
  ImplicitSurfaceForce(const ImplicitSurfaceForce &);

  /// Assignment operator
  ImplicitSurfaceForce &operator =(const ImplicitSurfaceForce &);

public:

  /// Destructor
  virtual ~ImplicitSurfaceForce();

  /// Initialize force term once input and parameters have been set
  virtual void Initialize();

  // ---------------------------------------------------------------------------
  // Configuration
protected:

  /// Set parameter value from string
  virtual bool SetWithPrefix(const char *, const char *);

  /// Set parameter value from string
  virtual bool SetWithoutPrefix(const char *, const char *);

public:

  // Import other overloads
  using SurfaceForce::Parameter;

  /// Get parameter name/value pairs
  virtual ParameterList Parameter() const;

  // ---------------------------------------------------------------------------
  // Surface distance

  /// Get self-distance value at given world position along specified direction
  double SelfDistance(const double p[3], const double n[3]) const;

  /// Get distance value at given world position
  ///
  /// \param[in] p Point at which to evaluate implicit surface distance function.
  ///
  /// \returns Interpolated implicit surface distance function value.
  double Distance(const double p[3]) const;

  /// Get distance value at given world position along specified direction
  ///
  /// \param[in] p Starting point for ray casting.
  /// \param[in] n Direction of ray (incl. opposite direction).
  ///
  /// \returns Distance of closest intersection of ray cast from point \p p in
  ///          direction \p n (opposite directions) of length \c _MaxDistance.
  ///          If no intersection occurs, \c _MaxDistance is returned.
  double Distance(const double p[3], const double n[3]) const;

  /// Get (normalized) distance gradient at given world position
  ///
  /// \param[in]  p         Point at which to evaluate implicit surface distance gradient.
  /// \param[out] g         Gradient of implicit surface distance.
  /// \param[in]  normalize Whether to normalize the gradient vector.
  void DistanceGradient(const double p[3], double g[3], bool normalize = false) const;

protected:

  /// Get pointer to point data array of minimum implicit surface distances
  vtkDataArray *MinimumDistances() const;

  /// Initialize point data array used to store minimum implicit surface distances
  void InitializeMinimumDistances();

  /// Update minimum distances to implicit surface
  void UpdateMinimumDistances();

  /// Get pointer to point data array of implicit surface distances in normal directions
  vtkDataArray *Distances() const;

  /// Initialize point data array used to store implicit surface distances in normal directions
  void InitializeDistances();

  /// Update distances from implicit surface in normal direction
  void UpdateDistances();

};


} // namespace mirtk

#endif // MIRTK_ImplicitSurfaceForce_H
