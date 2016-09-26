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

#include "mirtk/ImageEdgeDistance.h"

#include "mirtk/Math.h"
#include "mirtk/Parallel.h"
#include "mirtk/Profiling.h"
#include "mirtk/DataStatistics.h"
#include "mirtk/MeshSmoothing.h"
#include "mirtk/MedianMeshFilter.h"
#include "mirtk/EdgeConnectivity.h"
#include "mirtk/FastCubicBSplineInterpolateImageFunction.h"

#include "mirtk/PointSetIO.h"
#include "mirtk/PointSetUtils.h"

#include "vtkPointData.h"

#include <algorithm>


namespace mirtk {


// Register energy term with object factory during static initialization
mirtkAutoRegisterEnergyTermMacro(ImageEdgeDistance);


// =============================================================================
// Auxiliary functions
// =============================================================================

namespace ImageEdgeDistanceUtils {


// Type of discrete intensity image
typedef GenericImage<double> DiscreteImage;

// Type of interpolated image
typedef GenericFastCubicBSplineInterpolateImageFunction<DiscreteImage> ContinuousImage;

// -----------------------------------------------------------------------------
/// Compute distance to closest image edge
struct ComputeDistances
{
  vtkPoints       *_Points;
  vtkDataArray    *_Normals;
  ContinuousImage *_Image;
  vtkDataArray    *_ImageGradient;
  vtkDataArray    *_Distances;
  vtkDataArray    *_Magnitude;
  double           _Padding;
  double           _MinIntensity;
  double           _MaxIntensity;
  double           _MaxDistance;
  double           _StepLength;

  enum ImageEdgeDistance::EdgeType _EdgeType;

  // ---------------------------------------------------------------------------
  inline void SampleIntensity(Array<double> &f, Point p, const Vector3 &dp) const
  {
    p -= static_cast<double>((f.size() - 1) / 2) * dp;
    for (size_t i = 0; i < f.size(); ++i, p += dp) {
      f[i] = _Image->Evaluate(p.x, p.y, p.z);
    }
  }

  // ---------------------------------------------------------------------------
  inline void SampleGradient(Array<double> &g, Point p, const Vector3 &dp) const
  {
    Matrix jac(1, 3);
    Vector3 n = dp;
    n.Normalize();
    p -= static_cast<double>((g.size() - 1) / 2) * dp;
    for (size_t i = 0; i < g.size(); ++i, p += dp) {
      _Image->Jacobian3D(jac, p.x, p.y, p.z);
      g[i] = n.x * jac(0, 0) + n.y * jac(0, 1) + n.z * jac(0, 2);
    }
  }

  // ---------------------------------------------------------------------------
  inline int ClosestMinimum(const Array<double> &g) const
  {
    const int k  = static_cast<int>(g.size());
    const int i0 = (k - 1) / 2;

    int i1 = i0, i2 = i0;
    while (i1 < k - 2 && g[i1] > g[i1+1]) ++i1;
    while (i2 > 1     && g[i2] > g[i2-1]) --i2;

    return (g[i2] < g[i1] ? i2 : i1);
  }

  // ---------------------------------------------------------------------------
  inline int ClosestMaximum(const Array<double> &g) const
  {
    const int k  = static_cast<int>(g.size());
    const int i0 = (k - 1) / 2;

    int i1 = i0, i2 = i0;
    while (i1 < k - 2 && g[i1] < g[i1+1]) ++i1;
    while (i2 > 1     && g[i2] < g[i2-1]) --i2;

    return (g[i2] > g[i1] ? i2 : i1);
  }

  // ---------------------------------------------------------------------------
  inline int StrongestMinimum(const Array<double> &g) const
  {
    const int k  = static_cast<int>(g.size());
    const int i0 = (k - 1) / 2;

    int i = i0;
    for (int j = i0 + 1; j < k; ++j) {
      if (g[j] < g[i]) i = j;
    }
    for (int j = i0 - 1; j >= 0; --j) {
      if (g[j] < g[i]) i = j;
    }

    return i;
  }

  // ---------------------------------------------------------------------------
  inline int StrongestMaximum(const Array<double> &g) const
  {
    const int k  = static_cast<int>(g.size());
    const int i0 = (k - 1) / 2;

    int i = i0;
    for (int j = i0 + 1; j < k; ++j) {
      if (g[j] > g[i]) i = j;
    }
    for (int j = i0 - 1; j >= 0; --j) {
      if (g[j] > g[i]) i = j;
    }

    return i;
  }

  // ---------------------------------------------------------------------------
  void operator ()(const blocked_range<int> &ptIds) const
  {
    const int r = ifloor(_MaxDistance / _StepLength);
    const int k = 2 * r + 1;

    Array<double> f(k), g(k);
    int     i, j, j1, j2;
    Point   p;
    Vector3 n;

    for (int ptId = ptIds.begin(); ptId != ptIds.end(); ++ptId) {
      // Get point position and scaled normal
      _Points ->GetPoint(ptId, p);
      _Normals->GetTuple(ptId, n), n *= _StepLength;
      // Transform point/vector to image space
      _Image->WorldToImage(p);
      _Image->WorldToImage(n);
      // Sample image gradient along cast ray
      SampleGradient(g, p, n);
      // Find edge in normal direction
      switch (_EdgeType) {
        case ImageEdgeDistance::Extremum: {
          if      (g[r] < 0.) j = ClosestMinimum(g);
          else if (g[r] > 0.) j = ClosestMaximum(g);
          else                j = r;
        } break;
        case ImageEdgeDistance::ClosestMinimum: {
          j = ClosestMinimum(g);
        } break;
        case ImageEdgeDistance::ClosestMaximum: {
          j = ClosestMaximum(g);
        } break;
        case ImageEdgeDistance::ClosestExtremum: {
          j1 = ClosestMinimum(g);
          j2 = ClosestMaximum(g);
          j  = (abs(j1 - r) < abs(j2 - r) ? j1 : j2);
        } break;
        case ImageEdgeDistance::StrongestMinimum: {
          j = StrongestMinimum(g);
        } break;
        case ImageEdgeDistance::StrongestMaximum: {
          j = StrongestMaximum(g);
        } break;
        case ImageEdgeDistance::StrongestExtremum: {
          j1 = StrongestMinimum(g);
          j2 = StrongestMaximum(g);
          j  = (abs(g[j1]) > abs(g[j2]) ? j1 : j2);
        } break;
      }
      // When intensity thresholds set, use it to ignore irrelevant edges
      if (j != r && (!IsNaN(_Padding) || !IsInf(_MinIntensity) || !IsInf(_MaxIntensity))) {
        SampleIntensity(f, p, n);
        if (f[j] < _MinIntensity || f[j] > _MaxIntensity) {
          j = r;
        }
        if (j < r) {
          i = r;
          for (i = r; i > 0; --i) {
            if (f[i] < _Padding) {
              i = 0;
              break;
            }
            if (g[j] * g[i] < 0.) break;
          }
          if (i == 0) j = r;
        } else if (j > r) {
          i = r;
          for (i = r; i < k; ++i) {
            if (f[i] < _Padding) {
              i = k;
              break;
            }
            if (g[j] * g[i] < 0.) break;
          }
          if (i == k) j = r;
        }
      }
      // Set point distance to found edge and edge strength
      _Distances->SetComponent(ptId, 0, static_cast<double>(j - r) * _StepLength);
      _Magnitude->SetComponent(ptId, 0, IsNaN(g[j]) ? 0. : abs(g[j]));
    }
  }
};

// -----------------------------------------------------------------------------
/// Compute magnitude of image edge force
struct ComputeMagnitude
{
  vtkDataArray *_Status;
  vtkDataArray *_Distances;
  double        _DistanceScale;
  double        _MaxMagnitude;
  vtkDataArray *_Magnitude;

  void operator ()(const blocked_range<int> &ptIds) const
  {
    double d, d2, m1 = 1., m2;
    for (auto ptId = ptIds.begin(); ptId != ptIds.end(); ++ptId) {
      if (_Status && _Status->GetComponent(ptId, 0) == 0.) {
        _Magnitude->SetComponent(ptId, 0, 0.);
      } else {
        // Edge magnitude factor
        m1 = _Magnitude->GetComponent(ptId, 0);
        m1 = SShapedMembershipFunction(m1, 0., _MaxMagnitude);
        // Edge distance factor
        d  = _Distances->GetComponent(ptId, 0);
        d2 = _DistanceScale * d, d2 *= d2;
        m2 = d2 / (1. + d2);
        // Force magnitude
        _Magnitude->SetComponent(ptId, 0, m1 * copysign(m2, d));
      }
    }
  }
};

// -----------------------------------------------------------------------------
/// Compute force term penalty
struct ComputePenalty
{
  vtkDataArray *_Distances;
  double        _Sum;

  ComputePenalty() : _Sum(0.) {}

  ComputePenalty(const ComputePenalty &other, split)
  :
    _Distances(other._Distances), _Sum(0.)
  {}

  void join(const ComputePenalty &other)
  {
    _Sum += other._Sum;
  }

  void operator ()(const blocked_range<int> &ptIds)
  {
    for (auto ptId = ptIds.begin(); ptId != ptIds.end(); ++ptId) {
      _Sum += abs(_Distances->GetComponent(ptId, 0));
    }
  }
};

// -----------------------------------------------------------------------------
/// Compute gradient of force term, i.e., the negative force
struct ComputeGradient
{
  typedef ImageEdgeDistance::GradientType GradientType;

  vtkDataArray *_Normals;
  vtkDataArray *_Magnitude;
  GradientType *_Gradient;

  void operator ()(const blocked_range<int> &ptIds) const
  {
    double m, n[3];
    for (auto ptId = ptIds.begin(); ptId != ptIds.end(); ++ptId) {
      _Normals->GetTuple(ptId, n);
      m = _Magnitude->GetComponent(ptId, 0);
      _Gradient[ptId] = -m * GradientType(n);
    }
  }
};


} // namespace ImageEdgeDistanceUtils
using namespace ImageEdgeDistanceUtils;

// =============================================================================
// Enum <-> string conversion
// =============================================================================

// -----------------------------------------------------------------------------
template <>
bool FromString(const char *str, enum ImageEdgeDistance::EdgeType &value)
{
  const string lstr = ToLower(str);
  if (lstr == "extremum") {
    value = ImageEdgeDistance::Extremum;
  } else if (lstr == "closestminimum" || lstr == "closest minimum" ||
             lstr == "localminimum"   || lstr == "local minimum"   ||
             lstr == "minimum" || lstr == "min") {
    value = ImageEdgeDistance::ClosestMinimum;
  } else if (lstr == "closestmaximum" || lstr == "closest maximum" ||
             lstr == "localmaximum"   || lstr == "local maximum"   ||
             lstr == "maximum" || lstr == "max") {
    value = ImageEdgeDistance::ClosestMaximum;
  } else if (lstr == "closestextremum" || lstr == "closest extremum") {
    value = ImageEdgeDistance::ClosestExtremum;
  } else if (lstr == "strongestminimum" || lstr == "strongest minimum") {
    value = ImageEdgeDistance::StrongestMinimum;
  } else if (lstr == "strongestmaximum" || lstr == "strongest maximum") {
    value = ImageEdgeDistance::StrongestMaximum;
  } else if (lstr == "strongestextremum" || lstr == "strongest extremum") {
    value = ImageEdgeDistance::StrongestExtremum;
  } else {
    return false;
  }
  return true;
}

// -----------------------------------------------------------------------------
template <>
string ToString(const enum ImageEdgeDistance::EdgeType &value, int w, char c, bool left)
{
  const char *str;
  switch (value) {
    case ImageEdgeDistance::Extremum:          { str = "Extremum"; } break;
    case ImageEdgeDistance::ClosestMinimum:    { str = "ClosestMinimum"; } break;
    case ImageEdgeDistance::ClosestMaximum:    { str = "ClosestMaximum"; } break;
    case ImageEdgeDistance::ClosestExtremum:   { str = "ClosestExtremum"; } break;
    case ImageEdgeDistance::StrongestMinimum:  { str = "StrongestMinimum"; } break;
    case ImageEdgeDistance::StrongestMaximum:  { str = "StrongestMaximum"; } break;
    case ImageEdgeDistance::StrongestExtremum: { str = "StrongestExtremum"; } break;
  }
  return ToString(str, w, c, left);
}

// =============================================================================
// Construction/Destruction
// =============================================================================

// -----------------------------------------------------------------------------
void ImageEdgeDistance::CopyAttributes(const ImageEdgeDistance &other)
{
  _EdgeType           = other._EdgeType;
  _MinIntensity       = other._MinIntensity;
  _MaxDistance        = other._MaxDistance;
  _MedianFilterRadius = other._MedianFilterRadius;
  _DistanceSmoothing  = other._DistanceSmoothing;
  _MagnitudeSmoothing = other._MagnitudeSmoothing;
}

// -----------------------------------------------------------------------------
ImageEdgeDistance::ImageEdgeDistance(const char *name, double weight)
:
  SurfaceForce(name, weight),
  _EdgeType(Extremum),
  _Padding(NaN),
  _MinIntensity(-inf),
  _MaxIntensity(inf),
  _MaxDistance(0.),
  _MedianFilterRadius(0),
  _DistanceSmoothing(0),
  _MagnitudeSmoothing(2)
{
  _ParameterPrefix.push_back("Image edge distance ");
  _ParameterPrefix.push_back("Intensity edge distance ");
  _ParameterPrefix.push_back("Edge distance ");
}

// -----------------------------------------------------------------------------
ImageEdgeDistance::ImageEdgeDistance(const ImageEdgeDistance &other)
:
  SurfaceForce(other)
{
  CopyAttributes(other);
}

// -----------------------------------------------------------------------------
ImageEdgeDistance &ImageEdgeDistance::operator =(const ImageEdgeDistance &other)
{
  if (this != &other) {
    SurfaceForce::operator =(other);
    CopyAttributes(other);
  }
  return *this;
}

// -----------------------------------------------------------------------------
ImageEdgeDistance::~ImageEdgeDistance()
{
}

// =============================================================================
// Configuration
// =============================================================================

// -----------------------------------------------------------------------------
bool ImageEdgeDistance::SetWithoutPrefix(const char *param, const char *value)
{
  if (strcmp(param, "Type") == 0 || strcmp(param, "Mode") == 0) {
    return FromString(value, _EdgeType);
  }
  if (strcmp(param, "Maximum") == 0 || strcmp(param, "Maximum distance") == 0) {
    return FromString(value, _MaxDistance);
  }
  if (strcmp(param, "Intensity threshold") == 0 || strcmp(param, "Padding") == 0) {
    return FromString(value, _Padding);
  }
  if (strcmp(param, "Lower intensity threshold") == 0  || strcmp(param, "Lower threshold") == 0 || strcmp(param, "Minimum intensity") == 0 || strcmp(param, "Intensity threshold") == 0) {
    return FromString(value, _MinIntensity);
  }
  if (strcmp(param, "Upper intensity threshold") == 0 || strcmp(param, "Upper intensity") == 0 || strcmp(param, "Maximum intensity") == 0) {
    return FromString(value, _MaxIntensity);
  }
  if (strcmp(param, "Median filtering") == 0 || strcmp(param, "Median filter radius") == 0) {
    return FromString(value, _MedianFilterRadius);
  }
  if (strcmp(param, "Smoothing iterations")          == 0 ||
      strcmp(param, "Distance smoothing")            == 0 ||
      strcmp(param, "Distance smoothing iterations") == 0) {
    return FromString(value, _DistanceSmoothing);
  }
  if (strcmp(param, "Magnitude smoothing")            == 0 ||
      strcmp(param, "Magnitude smoothing iterations") == 0) {
    return FromString(value, _MagnitudeSmoothing);
  }
  return SurfaceForce::SetWithoutPrefix(param, value);
}

// -----------------------------------------------------------------------------
ParameterList ImageEdgeDistance::Parameter() const
{
  ParameterList params = SurfaceForce::Parameter();
  InsertWithPrefix(params, "Type",                 _EdgeType);
  InsertWithPrefix(params, "Maximum",              _MaxDistance);
  InsertWithPrefix(params, "Intensity threshold",  _Padding);
  InsertWithPrefix(params, "Lower intensity",      _MinIntensity);
  InsertWithPrefix(params, "Upper intensity",      _MaxIntensity);
  InsertWithPrefix(params, "Median filter radius", _MedianFilterRadius);
  InsertWithPrefix(params, "Smoothing iterations", _DistanceSmoothing);
  InsertWithPrefix(params, "Magnitude smoothing",  _MagnitudeSmoothing);
  return params;
}

// =============================================================================
// Initialization
// =============================================================================

// -----------------------------------------------------------------------------
void ImageEdgeDistance::Initialize()
{
  // Initialize base class
  SurfaceForce::Initialize();
  if (_NumberOfPoints == 0) return;

  // Image resolution, i.e., length of voxel diagonal
  const double res = sqrt(pow(_Image->XSize(), 2) +
                          pow(_Image->YSize(), 2) +
                          pow(_Image->ZSize(), 2));

  // Parameters for ray casting to sample image intensities near surface
  _StepLength = .25 * res;
  if (_MaxDistance <= 0.) _MaxDistance = 4. * res;

  // Add point data arrays
  AddPointData("Distance");
  AddPointData("Magnitude");
}

// =============================================================================
// Evaluation
// =============================================================================

// -----------------------------------------------------------------------------
void ImageEdgeDistance::Update(bool gradient)
{
  // Update base class
  SurfaceForce::Update(gradient);

  vtkPolyData  * const surface   = DeformedSurface();
  vtkDataArray * const distances = PointData("Distance");
  vtkDataArray * const magnitude = PointData("Magnitude");
  vtkDataArray * const status    = Status();

  if (distances->GetMTime() >= surface->GetMTime()) return;

  // Compute distance to closest image edge
  ContinuousImage image;
  image.Input(_Image);
  image.Initialize();

  MIRTK_START_TIMING();
  ComputeDistances eval;
  eval._Points       = Points();
  eval._Normals      = Normals();
  eval._Image        = &image;
  eval._Distances    = distances;
  eval._Magnitude    = magnitude;
  eval._Padding      = _Padding;
  eval._MinIntensity = _MinIntensity;
  eval._MaxIntensity = _MaxIntensity;
  eval._MaxDistance  = _MaxDistance;
  eval._StepLength   = _StepLength;
  eval._EdgeType     = _EdgeType;
  parallel_for(blocked_range<int>(0, _NumberOfPoints), eval);
  MIRTK_DEBUG_TIMING(5, "computing edge distances");

  // Smooth measurements
  if (_MedianFilterRadius > 0) {
    MIRTK_RESET_TIMING();
    MedianMeshFilter median;
    median.Input(surface);
    median.Connectivity(_MedianFilterRadius);
    median.DataArray(distances);
    median.Run();
    distances->DeepCopy(median.Output()->GetPointData()->GetArray(distances->GetName()));
    MIRTK_DEBUG_TIMING(5, "edge distance median filtering");
  }
  if (_DistanceSmoothing > 0) {
    MIRTK_RESET_TIMING();
    MeshSmoothing smoother;
    smoother.Input(surface);
    smoother.SmoothPointsOff();
    smoother.SmoothArray(distances->GetName());
    smoother.Weighting(MeshSmoothing::Gaussian);
    smoother.NumberOfIterations(_DistanceSmoothing);
    smoother.Run();
    distances->DeepCopy(smoother.Output()->GetPointData()->GetArray(distances->GetName()));
    MIRTK_DEBUG_TIMING(5, "edge distance smoothing");
  }
  if (_MagnitudeSmoothing > 0) {
    MIRTK_RESET_TIMING();
    MeshSmoothing smoother;
    smoother.Input(surface);
    smoother.SmoothPointsOff();
    smoother.SmoothArray(magnitude->GetName());
    smoother.Weighting(MeshSmoothing::Combinatorial);
    smoother.NumberOfIterations(_MagnitudeSmoothing);
    smoother.Run();
    magnitude->DeepCopy(smoother.Output()->GetPointData()->GetArray(magnitude->GetName()));
    MIRTK_DEBUG_TIMING(5, "edge magnitude smoothing");
  }

  // Make force magnitude proportional to both edge distance and strength
  using mirtk::data::statistic::Mean;
  using mirtk::data::statistic::AbsPercentile;

  MIRTK_RESET_TIMING();
  bool * const mask = new bool[_NumberOfPoints];
  for (int ptId = 0; ptId < _NumberOfPoints; ++ptId) {
    mask[ptId] = (status->GetComponent(ptId, 0) != 0.);
  }
  const auto dmax = AbsPercentile::Calculate(95, distances, mask);
  const auto mavg = Mean::Calculate(magnitude, mask);
  delete[] mask;
  MIRTK_DEBUG_TIMING(5, "calculating edge distance statistics");

  if (dmax > 0. && mavg > 0.) {
    MIRTK_RESET_TIMING();
    ComputeMagnitude eval;
    eval._Status        = status;
    eval._Distances     = distances;
    eval._DistanceScale = 1. / max(.1, dmax);
    eval._MaxMagnitude  = mavg;
    eval._Magnitude     = magnitude;
    parallel_for(blocked_range<int>(0, _NumberOfPoints), eval);
    MIRTK_DEBUG_TIMING(5, "computing edge force magnitude");
  } else {
    magnitude->FillComponent(0, 0.);
  }

  distances->Modified();
  magnitude->Modified();
}

// -----------------------------------------------------------------------------
double ImageEdgeDistance::Evaluate()
{
  if (_NumberOfPoints == 0) return 0.;
  ComputePenalty eval;
  eval._Distances = PointData("Distance");
  parallel_reduce(blocked_range<int>(0, _NumberOfPoints), eval);
  return eval._Sum / _NumberOfPoints;
}

// -----------------------------------------------------------------------------
void ImageEdgeDistance::EvaluateGradient(double *gradient, double step, double weight)
{
  if (_NumberOfPoints == 0) return;

  memset(_Gradient, 0, _NumberOfPoints * sizeof(GradientType));

  ComputeGradient eval;
  eval._Normals   = Normals();
  eval._Magnitude = PointData("Magnitude");
  eval._Gradient  = _Gradient;
  parallel_for(blocked_range<int>(0, _NumberOfPoints), eval);

  SurfaceForce::EvaluateGradient(gradient, step, weight / _NumberOfPoints);
}


} // namespace mirtk
