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

#ifndef MIRKT_ImageEdgeDistance_H
#define MIRKT_ImageEdgeDistance_H

#include "mirtk/SurfaceForce.h"


namespace mirtk {


/**
 * External surface force which attracts the surface to nearby image edges
 */
class ImageEdgeDistance : public SurfaceForce
{
  mirtkEnergyTermMacro(ImageEdgeDistance, EM_ImageEdgeDistance);

  // ---------------------------------------------------------------------------
  // Types

public:

  /// Enumeration of edge force modes based on directional derivative of image intensities
  enum EdgeType
  {
    Extremum,         ///< Attract points to closest extrema of same sign
    ClosestMinimum,   ///< Attract points to closest minima
    ClosestMaximum,   ///< Attract points to closest maxima
    ClosestExtremum,  ///< Attract points to closest extrema
    StrongestMinimum, ///< Attract points to strongest minima
    StrongestMaximum, ///< Attract points to strongest maxima
    StrongestExtremum ///< Attract points to strongest extrema
  };

  // ---------------------------------------------------------------------------
  // Attributes

private:

  /// Type of edge which points are attracted to
  mirtkPublicAttributeMacro(enum EdgeType, EdgeType);

  /// Minimum foreground intensity value
  mirtkPublicAttributeMacro(double, MinIntensity);

  /// Maximum edge point distance
  mirtkPublicAttributeMacro(double, MaxDistance);

  /// Radius of distance median filter
  mirtkPublicAttributeMacro(int, MedianFilterRadius);

  /// Number of edge distance smoothing iterations
  mirtkPublicAttributeMacro(int, DistanceSmoothing);

  /// Number of edge magnitude smoothing iterations
  mirtkPublicAttributeMacro(int, MagnitudeSmoothing);

  /// Step length used for ray casting
  mirtkPublicAttributeMacro(double, StepLength);

private:

  /// Copy attributes of this class from another instance
  void CopyAttributes(const ImageEdgeDistance &);

  // ---------------------------------------------------------------------------
  // Construction/Destruction

public:

  /// Constructor
  ImageEdgeDistance(const char * = "", double = 1.0);

  /// Copy constructor
  ImageEdgeDistance(const ImageEdgeDistance &);

  /// Assignment operator
  ImageEdgeDistance &operator =(const ImageEdgeDistance &);

  /// Destructor
  virtual ~ImageEdgeDistance();

  // ---------------------------------------------------------------------------
  // Configuration

protected:

  /// Set parameter value from string
  virtual bool SetWithoutPrefix(const char *, const char *);

public:

  // Import other overloads
  using SurfaceForce::Parameter;

  /// Get parameter name/value pairs
  virtual ParameterList Parameter() const;

  // ---------------------------------------------------------------------------
  // Initialization

  /// Initialize external force once input and parameters have been set
  virtual void Initialize();

  // ---------------------------------------------------------------------------
  // Evaluation

public:

  /// Update moving input points and internal state of force term
  virtual void Update(bool = true);

protected:

  /// Evaluate external force term
  virtual double Evaluate();

  /// Evaluate external force
  virtual void EvaluateGradient(double *, double, double);

};

////////////////////////////////////////////////////////////////////////////////
// Enum <-> string conversion
////////////////////////////////////////////////////////////////////////////////

template <> bool FromString(const char *, enum ImageEdgeDistance::EdgeType &);
template <> string ToString(const enum ImageEdgeDistance::EdgeType &, int, char, bool);


} // namespace mirtk

#endif // MIRKT_ImageEdgeDistance_H
