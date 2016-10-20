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
#include "mirtk/FastCubicBSplineInterpolateImageFunction.h"

#include "mirtk/PointSetIO.h"
#include "mirtk/PointSetUtils.h"

#include "vtkPointData.h"
#include "vtkImageData.h"
#include "vtkImageStencilData.h"
#include "vtkImageStencilIterator.h"


namespace mirtk {


// Register energy term with object factory during static initialization
mirtkAutoRegisterEnergyTermMacro(ImageEdgeDistance);

// =============================================================================
// Debugging -- output for creating figures of intensity profiles
// =============================================================================
#define BUILD_WITH_DEBUG_CODE 0

#if BUILD_WITH_DEBUG_CODE
  const double dbg_dist = 2.;

  // dHCP CC00050XX01
  // ----------------
  // * outside wrong CSF->BG edge
  //const Point dbg_voxel(64, 90, 61);
  //const Point dbg_voxel(16, 128, 86);
  // * outside within CSF next to opposite WM->GM edge
  //const Point dbg_voxel(36, 64, 136);
  // * already correct
  //const Point dbg_voxel(80, 90, 66);
  // * strong dark GM next to bright CSF at boundary
  //const Point dbg_voxel(146, 76, 163);


  // dHCP CC00051XX02
  // ----------------

  // * WM->dGM->cWM transition
  //const Point dbg_voxel(60, 59, 85);
  //const Point dbg_voxel(62, 65, 85);
  // * already correct, near BG
  //const Point dbg_voxel(94, 42, 73);
  //const Point dbg_voxel(94, 43, 70);

  // dHCP CC00055XX06
  // ----------------

  // * already correct, missing after refinement
  //const Point dbg_voxel(77, 47, 74);
  //const Point dbg_voxel(76, 47, 74);
  //const Point dbg_voxel(117, 98, 52);
  const Point dbg_voxel(108, 54, 155);

  // * slightly inside, missing after refinement
  //const Point dbg_voxel(105, 89, 71);

  // * WM->cGM->BG misclassified as WM->dGM->cGM transition
  //const Point dbg_voxel(69, 41, 57);
  //const Point dbg_voxel(108, 77, 85); // not very clear

  // * correctly applied WM->dGM->cGM correction
  //const Point dbg_voxel(119, 39, 82);

#endif

// =============================================================================
// Auxiliary functions
// =============================================================================

namespace ImageEdgeDistanceUtils {


// Type of discrete intensity image
typedef GenericImage<double> DiscreteImage;

// Type of local intensity statistics image
typedef ImageEdgeDistance::LocalStatsImage LocalStatsImage;

// Type of interpolated image
typedef GenericFastCubicBSplineInterpolateImageFunction<DiscreteImage> ContinuousImage;

// -----------------------------------------------------------------------------
/// Compute intersection of two normal distributions
double IntersectionOfNormalDistributions(double mean1, double var1, double mean2, double var2)
{
  if (fequal(mean1, mean2, 1e-9)) {
    return .5 * (mean1 + mean2);
  }

  const double a = .5/var2 - .5/var1;
  const double b = mean1/var1 - mean2/var2;
  const double c = .5 * (mean2*mean2/var2 - mean1*mean1/var1) + log(sqrt(var2/var1));
  const double d = sqrt(b*b - 4. * a * c);

  const double x1 = -2. * c / (b - d);
  const double x2 = -2. * c / (b + d);

  if (mean1 > mean2) swap(mean1, mean2);
  return (mean1 <= x1 && x1 <= mean2 ? x1 : x2);
}

// -----------------------------------------------------------------------------
/// Compute global intensity statistics
struct ComputeGlobalStatistics : public VoxelReduction
{
private:

  int    _Num;
  double _Sum;
  double _Sum2;

public:

  ComputeGlobalStatistics() : _Num(0), _Sum(0.), _Sum2(0.) {}

  void split(const ComputeGlobalStatistics &)
  {
    _Num = 0;
    _Sum = _Sum2 = 0.;
  }

  void join(const ComputeGlobalStatistics &other)
  {
    _Num  += other._Num;
    _Sum  += other._Sum;
    _Sum2 += other._Sum2;
  }

  template <class TIn, class TMask>
  void operator()(int, int, int, int, const TIn *in, const TMask *mask)
  {
    if (*mask != 0) {
      _Num  += 1;
      _Sum  += (*in);
      _Sum2 += (*in) * (*in);
    }
  }

  double Mean() const
  {
    return (_Num == 0 ? 0. : _Sum / _Num);
  }

  double Variance() const
  {
    const double mean = Mean();
    return (_Num == 0 ? 0. : (_Sum2 / _Num) - mean * mean);
  }
};

// -----------------------------------------------------------------------------
/// Compute local intensity statistics
class ComputeLocalStatistics : public VoxelFunction
{
  int    _Radius;
  double _GlobalMean;
  double _GlobalVariance;
  double _MaxMeanValue;
  int    _MinNumberOfSamples;

public:

  ComputeLocalStatistics(const ImageAttributes &attr, int width, double global_mean = 0., double global_variance = 0.)
  :
    _Radius(width / 2),
    _GlobalMean(global_mean),
    _GlobalVariance(global_variance),
    _MaxMeanValue(global_mean + 3. * sqrt(global_variance))
  {
    int max_nsamples = 1;
    if (attr._x > 1) max_nsamples *= width;
    if (attr._y > 1) max_nsamples *= width;
    if (attr._z > 1) max_nsamples *= width;
    _MinNumberOfSamples = max(1, 5 * max_nsamples / 100);
  }

  template <class TIn, class TMask, class TOut>
  void operator ()(int ci, int cj, int ck, int cl, const TIn *in, const TMask *mask, TOut *mean, TOut *var)
  {
    int    num = 0;
    double sum = 0., sum2 = 0., v;

    const int nx = _Domain->_x;
    const int ny = _Domain->_y;
    const int nz = _Domain->_z;

    const int i1 = max(0, ci - _Radius), i2 = min(ci + _Radius, nx - 1);
    const int j1 = max(0, cj - _Radius), j2 = min(cj + _Radius, ny - 1);
    const int k1 = max(0, ck - _Radius), k2 = min(ck + _Radius, nz - 1);

    const int xstride = 1;
    const int ystride =  nx - (i2 - i1 + 1);
    const int zstride = (ny - (j2 - j1 + 1)) * nx;
    const int offset  = _Domain->LatticeToIndex(i1, j1, k1) - _Domain->LatticeToIndex(ci, cj, ck);

    in -= offset, mask -= offset;
    for (int k = k1; k <= k2; ++k, in += zstride, mask += zstride)
    for (int j = j1; j <= j2; ++j, in += ystride, mask += ystride)
    for (int i = i1; i <= i2; ++i, in += xstride, mask += xstride) {
      if (*mask != 0) {
        v = static_cast<double>(*in);
        sum += v, sum2 += v * v, ++num;
      }
    }
    if (num >= _MinNumberOfSamples) {
      sum /= num, sum2 /= num;
      *mean = sum;
      *var  = sum2 - sum * sum;
      // Upper limit of variance is the global intensity variance such
      // that subcortical regions with dGM and CSF intensities included
      // in white matter segmentation cause no troubles in identifying WM;
      // similarly, the mean is limited to the global mean plus 5 * sigma
      //
      // Note: The following conditions are always false when global values
      //       are not available, i.e., NaN.
      if (*var > _GlobalVariance) {
        *var  = _GlobalVariance;
      }
      if (*mean > _MaxMeanValue) {
        *mean = _MaxMeanValue;
      }
    } else {
      *mean = _GlobalMean;
      *var  = _GlobalVariance;
    }
  }
};

// -----------------------------------------------------------------------------
/// Compute absolute difference of intensities around the mean
struct ComputeMeanAbsoluteDifference : public VoxelReduction
{
private:

  double _Mean;
  int    _Num;
  double _Sum;

public:

  ComputeMeanAbsoluteDifference(double mean) : _Mean(mean), _Num(0), _Sum(0.) {}

  void split(const ComputeMeanAbsoluteDifference &)
  {
    _Num = 0;
    _Sum = 0.;
  }

  void join(const ComputeMeanAbsoluteDifference &other)
  {
    _Num += other._Num;
    _Sum += other._Sum;
  }

  template <class TIn, class TMask>
  void operator()(int, int, int, int, const TIn *in, const TMask *mask)
  {
    if (*mask != 0) {
      _Num += 1;
      _Sum += abs(*in - _Mean);
    }
  }

  double Value() const
  {
    return (_Num == 0 ? 0. : _Sum / _Num);
  }
};

// -----------------------------------------------------------------------------
/// Compute bounding box for WM->dGM->cGM correction
///
/// The region in which to perform this correction is posterior the corpus
/// callosum towards the mid-plane of the cerebrum, nearby the posterior
/// lateral ventricles. Given a spatial normalization of the brain image
/// by the subdivide-brain-image tool, where the image axes are in RAS orientation
/// and the image center is the center of the brain, this function determines
/// a suitable bounding box for the correction. A ventricle mask/distance map
/// further helps to limit the correction to the left/right given the bounding
/// box of the lateral ventricles. The interior distance to the cortex as
/// produced by the subdivide-brain-image -output-inner-cortical-distance
/// option is further used to disable the correction near the boundary of
/// the cortical hull and limit it to gyri deep within the brain volume.
Array<int> ComputeCorticalDeepGreyMatterBoundingBox(const ImageAttributes &attr, const RealImage *dvent)
{
  Array<int> bounds(6);
  // Left/Right bounds
  if (dvent) {
    bounds[0] = bounds[1] = attr._x / 2;
    for (int k = 0; k < attr._z; ++k)
    for (int j = 0; j < attr._y; ++j) {
      for (int i = bounds[0] - 1; i >= 0; --i) {
        if (dvent->Get(i, j, k) < 0.) {
          bounds[0] = i;
        }
      }
      for (int i = bounds[1] + 1; i < attr._x; ++i) {
        if (dvent->Get(i, j, k) < 0.) {
          bounds[1] = i;
        }
      }
    }
    const int margin = ifloor(4. / attr._dx) + 1;
    bounds[0] = max(0,         bounds[0] - margin);
    bounds[1] = min(attr._x-1, bounds[1] + margin);
  } else {
    bounds[0] = 0;
    bounds[1] = attr._x - 1;
  }
  // Posterior/Anterior bounds
  bounds[2] = 0;
  bounds[3] = attr._y / 2;
  // Inferior/Superior bounds -- inner cortical distance used separately
  bounds[4] = 0;
  bounds[5] = attr._z - 1;
  return bounds;
}

// -----------------------------------------------------------------------------
/// Compute distance to closest image edge
struct ComputeDistances
{
  vtkPoints    *_Points;
  vtkDataArray *_Status;
  vtkDataArray *_Normals;

  vtkDataArray *_ImageGradient;
  vtkDataArray *_Distances;
  vtkDataArray *_Magnitude;

  const ContinuousImage *_Image;
  const BinaryImage     *_SurfaceMask;
  const RealImage       *_CorticalHullDistance;
  const RealImage       *_VentriclesDistance;
  const RealImage       *_CerebellumDistance;
  const LocalStatsImage *_LocalWhiteMatterMean;
  const LocalStatsImage *_LocalWhiteMatterVariance;
  const LocalStatsImage *_LocalGreyMatterMean;
  const LocalStatsImage *_LocalGreyMatterVariance;

  double _Padding;
  double _MinIntensity;
  double _MaxIntensity;
  double _MinGradient;
  double _MaxGradient;
  double _MaxDistance;
  double _StepLength;
  int    _NumberOfSamples;
  int    _MaxBackgroundSnapDepth;
  double _MinCorticalHullDistance;
  double _GlobalWhiteMatterMean;
  double _GlobalWhiteMatterSigma;
  double _GlobalWhiteMatterVariance;
  double _GlobalGreyMatterMean;
  double _GlobalGreyMatterSigma;
  double _GlobalGreyMatterVariance;
  double _GlobalWhiteMatterThreshold;
  const int *_CorticalDeepGreyMatterBoundingBox;

  #if BUILD_WITH_DEBUG_CODE
    Matrix *_IntensitySamples;
    Matrix *_GradientSamples;
  #endif

  /// Enumeration of different image edge forces
  enum ImageEdgeDistance::EdgeType _EdgeType;

  // ---------------------------------------------------------------------------
  /// Structure used to store information of an extremum of the intensity profile
  struct Extremum
  {
    int    idx;  ///< Index of normal ray sample corresponding to this extremum
    bool   min;  ///< Whether this extremum is a minimum (GM) or maximum (WM)
    double mean; ///< Local intensity mean
    double std;  ///< Local intensity standard deviation
    double var;  ///< Local intensity variance
    double prb;  ///< Probability that this minimum/maximum belongs to GM/WM

    Extremum(int i = -1, bool is_min = false)
    :
      idx(i), min(is_min), mean(NaN), var(NaN), prb(0.)
    {}

    inline operator bool() const { return idx >= 0; }
    inline operator int() const { return idx; }
    inline operator size_t() const { return static_cast<size_t>(idx); }
  };

  /// Type of 3D voxel index
  typedef Vector3D<int> Voxel;

  /// Sequence of function minima/maxima
  typedef Array<Extremum> Extrema;

  // ---------------------------------------------------------------------------
  /// Image coordinates of i-th point of the ray in dp centered at p
  inline Point RayPoint(const Point &p, const Vector3 &dp, int i, int k) const
  {
    return p + static_cast<double>(i - k/2) * dp;
  }

  // ---------------------------------------------------------------------------
  /// World coordinates of i-th point of the ray in dp centered at p
  inline Point RayWorld(const Point &p, const Vector3 &dp, int i, int k) const
  {
    Point q = RayPoint(p, dp, i, k);
    _Image->ImageToWorld(q);
    return q;
  }

  // ---------------------------------------------------------------------------
  /// Indices of voxel nearest to the i-th point of the ray in dp centered at p
  inline Voxel RayVoxel(const Point &p, const Vector3 &dp, int i, int k) const
  {
    const Point q = RayPoint(p, dp, i, k);
    return Voxel(iround(q.x), iround(q.y), iround(q.z));
  }

  // ---------------------------------------------------------------------------
  /// Unnormalized Gaussian function value
  inline double GaussianWeight(double value, double mean, double var) const
  {
    value -= mean;
    return exp(-.5 * value * value / var);
  }

  // ---------------------------------------------------------------------------
  /// Evaluate directional image derivative along ray in dp centered at p
  inline void SampleGradient(Array<double> &g, int k, const Point &p, const Vector3 &dp) const
  {
    const int i0 = k/2;
    int i;

    Point q;
    Voxel v;
    Matrix jac(1, 3);
    Vector3 n = dp;
    n.Normalize();

    q = p;
    for (i = i0; i <= k; ++i, q += dp) {
      v.x = iround(q.x), v.y = iround(q.y), v.z = iround(q.z);
      if (!_Image->Input()->IsInsideForeground(v.x, v.y, v.z)) break;
      _Image->Jacobian3D(jac, q.x, q.y, q.z);
      g[i] = n.x * jac(0, 0) + n.y * jac(0, 1) + n.z * jac(0, 2);
    }
    while (i <= k) g[i++] = NaN;

    q = p, q -= dp;
    for (i = i0 - 1; i >= 0; --i, q -= dp) {
      v.x = iround(q.x), v.y = iround(q.y), v.z = iround(q.z);
      if (!_Image->Input()->IsInsideForeground(v.x, v.y, v.z)) break;
      _Image->Jacobian3D(jac, q.x, q.y, q.z);
      g[i] = n.x * jac(0, 0) + n.y * jac(0, 1) + n.z * jac(0, 2);
    }
    while (i >= 0) g[i--] = NaN;
  }

  // ---------------------------------------------------------------------------
  /// Evaluate image function along ray in dp centered at p
  ///
  /// When a mask of the current surface inside region is given, the function
  /// values are set to NaN as soon as the ray goes from inside/outside the
  /// surface to outside/inside of it. Given f and g, it can be determined if
  /// a function value is NaN because of either the image foreground mask or
  /// the surface mask. The latter is used to prevent the force of causing
  /// self-intersections by finding the wrong image edges.
  ///
  /// \param[in] g Directional derivative values which are previously set to NaN
  ///              once the ray left the image foreground region. Used to avoid
  ///              re-evaluation of whether a point is in foreground or not.
  inline void SampleIntensity(Array<double> &f, const Array<double> &g, int k, const Point &p, const Vector3 &dp) const
  {
    const int i0 = k/2;
    int i;

    Point q;
    Voxel v;

    i = i0, q = p;
    if (_SurfaceMask) {
      while (i <= k && !IsNaN(g[i])) {
        f[i] = _Image->Evaluate(q.x, q.y, q.z);
        v.x = iround(q.x), v.y = iround(q.y), v.z = iround(q.z);
        if (_SurfaceMask->Get(v.x, v.y, v.z) == 0) {
          ++i, q += dp;
          break;
        }
        ++i, q += dp;
      }
    }
    while (i <= k && !IsNaN(g[i])) {
      v.x = iround(q.x), v.y = iround(q.y), v.z = iround(q.z);
      if (_SurfaceMask && _SurfaceMask->Get(v.x, v.y, v.z) != 0) break;
      f[i] = _Image->Evaluate(q.x, q.y, q.z);
      ++i, q += dp;
    }
    while (i <= k) f[i++] = NaN;

    i = i0, q = p;
    if (_SurfaceMask) {
      while (i >= 0 && !IsNaN(g[i])) {
        if (i != i0) f[i] = _Image->Evaluate(q.x, q.y, q.z);
        v.x = iround(q.x), v.y = iround(q.y), v.z = iround(q.z);
        if (_SurfaceMask && _SurfaceMask->Get(v.x, v.y, v.z) != 0) {
          --i, q -= dp;
          break;
        }
        --i, q -= dp;
      }
    }
    while (i >= 0 && !IsNaN(g[i])) {
      v.x = iround(q.x), v.y = iround(q.y), v.z = iround(q.z);
      if (_SurfaceMask && _SurfaceMask->Get(v.x, v.y, v.z) == 0) break;
      f[i] = _Image->Evaluate(q.x, q.y, q.z);
      if (_VentriclesDistance) {
        const double d = _VentriclesDistance->Get(v.x, v.y, v.z);
        if (d < _StepLength || (d < 2. && f[i] > _GlobalWhiteMatterMean)) {
          if (f[i] > _GlobalWhiteMatterMean && i0 - i > iceil(1. / _StepLength)) {
            --i;
          }
          --i;
          break;
        }
      }
      --i, q -= dp;
    }
    while (i >= 0) f[i--] = NaN;
  }

  // ---------------------------------------------------------------------------
  /// Evaluate image function at a single ray point
  inline double SampleIntensity(Point p, const Vector3 &dp, int i, int k) const
  {
    p += static_cast<double>(i - k/2) * dp;
    const Voxel v(iround(p.x), iround(p.y), iround(p.z));
    if (_Image->Input()->IsInsideForeground(v.x, v.y, v.z)) {
      return _Image->Evaluate(p.x, p.y, p.z);
    } else {
      return NaN;
    }
  }

  // ---------------------------------------------------------------------------
  /// Get minimum value within interval
  inline double MinimumValue(const Array<double> &v, int i = 0, int j = -1) const
  {
    if (j < 0) {
      j = static_cast<int>(v.size()) - 1;
    } else if (i > j) {
      swap(i, j);
    }
    double vmin = v[i];
    while (++i <= j) {
      if (v[i] < vmin) vmin = v[i];
    }
    return vmin;
  }

  // ---------------------------------------------------------------------------
  /// Get maximum value within interval
  inline double MaximumValue(const Array<double> &v, int i = 0, int j = -1) const
  {
    if (j < 0) {
      j = static_cast<int>(v.size()) - 1;
    } else if (i > j) {
      swap(i, j);
    }
    double vmax = v[i];
    while (++i <= j) {
      if (v[i] > vmax) vmax = v[i];
    }
    return vmax;
  }

  // ---------------------------------------------------------------------------
  /// Get maximum absolute value within interval
  inline double MaximumAbsValue(const Array<double> &v, int i = 0, int j = -1) const
  {
    if (j < 0) {
      j = static_cast<int>(v.size() - 1);
    } else if (i > j) {
      swap(i, j);
    }
    double vmax = abs(v[i]);
    while (++i <= j) {
      if (abs(v[i]) > vmax) vmax = abs(v[i]);
    }
    return vmax;
  }

  // ---------------------------------------------------------------------------
  inline int ClosestMinimum(const Array<double> &g) const
  {
    const int k  = static_cast<int>(g.size());
    const int i0 = (k - 1) / 2;

    auto i1 = i0;
    while (i1 < k - 1 && IsNaN(g[i1]))    ++i1;
    while (i1 < k - 1 && g[i1] > g[i1+1]) ++i1;

    auto i2 = i0;
    while (i2 > 0 && IsNaN(g[i2]))    --i2;
    while (i2 > 0 && g[i2] > g[i2-1]) --i2;

    return (g[i2] > g[i1] ? i2 : i1);
  }

  // ---------------------------------------------------------------------------
  inline int ClosestMaximum(const Array<double> &g) const
  {
    const int k  = static_cast<int>(g.size());
    const int i0 = (k - 1) / 2;

    auto i1 = i0;
    while (i1 < k - 1 && IsNaN(g[i1]))    ++i1;
    while (i1 < k - 1 && g[i1] < g[i1+1]) ++i1;

    auto i2 = i0;
    while (i2 > 0 && IsNaN(g[i2]))    --i2;
    while (i2 > 0 && g[i2] < g[i2-1]) --i2;

    return (g[i2] > g[i1] ? i2 : i1);
  }

  // ---------------------------------------------------------------------------
  inline int StrongestMinimum(const Array<double> &g) const
  {
    const int k  = static_cast<int>(g.size());
    const int i0 = (k - 1) / 2;

    auto i1 = i0;
    while (i1 < k - 1 && IsNaN(g[i1])) ++i1;
    for (auto i = i1 + 1; i < k; ++i) {
      if (g[i] < g[i1]) i1 = i;
    }

    auto i2 = i0;
    while (i2 > 0 && IsNaN(g[i2])) --i2;
    for (auto i = i2 - 1; i >= 0; --i) {
      if (g[i] < g[i2]) i2 = i;
    }

    return (g[i2] < g[i1] ? i2 : i1);
  }

  // ---------------------------------------------------------------------------
  inline int StrongestMaximum(const Array<double> &g) const
  {
    const int k  = static_cast<int>(g.size());
    const int i0 = (k - 1) / 2;

    auto i1 = i0;
    while (i1 < k - 1 && IsNaN(g[i1])) ++i1;
    for (auto i = i1 + 1; i < k; ++i) {
      if (g[i] > g[i1]) i1 = i;
    }

    auto i2 = i0;
    while (i2 > 0 && IsNaN(g[i2])) --i2;
    for (auto i = i2 - 1; i >= 0; --i) {
      if (g[i] < g[i2]) i2 = i;
    }

    return (g[i2] > g[i1] ? i2 : i1);
  }

  // ---------------------------------------------------------------------------
  /// Find index of previous extremum, including whether it is a minimum or maximum
  inline Extremum PrevExtremum(const Extremum &current, const Array<double> &v, int k) const
  {
    Extremum next;
    if (current.idx > 0) {
      int j = current.idx - 1;
      if (!IsNaN(v[j])) {
        while (j > 0 && v[j] == v[j-1]) --j;
        if (j > 0) {
          next.min = (v[current.idx] > v[j]);
          if (next.min) {
            while (j > 0 && v[j] >= v[j-1]) --j;
          } else {
            while (j > 0 && v[j] <= v[j-1]) --j;
          }
          next.idx = j;
        }
      }
    }
    return next;
  }

  // ---------------------------------------------------------------------------
  /// Find index of next extremum, including whether it is a minimum or maximum
  inline Extremum NextExtremum(const Extremum &current, const Array<double> &v, int k) const
  {
    Extremum next;
    if (current.idx < k) {
      int j = current.idx + 1;
      if (!IsNaN(v[j])) {
        while (j < k && v[j] == v[j+1]) ++j;
        if (j <= k) {
          next.min = (v[current.idx] > v[j]);
          if (next.min) {
            while (j < k && v[j] >= v[j+1]) ++j;
          } else {
            while (j < k && v[j] <= v[j+1]) ++j;
          }
          next.idx = j;
        }
      }
    }
    return next;
  }

  // ---------------------------------------------------------------------------
  /// Get indices of alternating sequence of minima and maxima
  inline void FindExtrema(Extrema &extrema, const Array<double> &v, int k) const
  {
    extrema.clear();
    Extremum begin(k/2);
    if (!IsNaN(v[begin.idx])) {
      while (begin.idx > 0 && !IsNaN(v[begin.idx-1])) --begin.idx;
    }
    Extremum current = NextExtremum(begin, v, k);
    if (current) {
      begin.min = !current.min;
      extrema.push_back(begin);
      do {
        extrema.push_back(current);
      } while ((current = NextExtremum(current, v, k)));
    }
  }

  // ---------------------------------------------------------------------------
  /// Get position of central extremum
  inline Extrema::iterator CentralExtremum(Extrema &extrema, int k, bool min = false) const
  {
    const int i0 = k/2;
    if (extrema.size() < 2) return extrema.end();
    Extrema::iterator m = extrema.begin();
    while (m != extrema.end() && m->idx < i0) ++m;
    if (m == extrema.end()) --m;
    if (min) {
      if (m->min) {
        //if (distance(extrema.begin(), m) > 1) m -= 2;
      } else {
        if (m == extrema.begin()) ++m;
        else                      --m;
      }
    }
    return m;
  }

  // ---------------------------------------------------------------------------
  /// Remove irrelevant extrema
  inline void CleanExtrema(Extrema &extrema, const Array<double> &f, const Array<double> &g, int k) const
  {
    Extrema::iterator l, r;
    if (extrema.size() < 2) {
      extrema.clear();
      return;
    }
    const double gm_thres = _GlobalGreyMatterMean - 3. * _GlobalGreyMatterSigma;
    // Insert stationary inflection points and interim points in cases
    // where the minimum is below Mean(GM) - 3 StDev(GM), i.e., likely
    // a transition to background. If even below the _MinIntensity,
    // this minimum will be removed afterwards. An actual minimum at
    // the interim point where the curvature changes rapidly is needed
    // to still be able to identify a WM/GM edge near dark background.
    const double min_diff = 2. * _GlobalWhiteMatterSigma;
    const double max_diff = 5. * _GlobalWhiteMatterSigma;
    Extremum prv, mid, nxt;
    r = extrema.end() - 1;
    while (r != extrema.begin()) {
      l = r - 1;
      const double diff = abs(f[l->idx] - f[r->idx]);
      if (diff > min_diff) {
        int prv_mid_idx = r->idx;
        nxt = PrevExtremum(*r,  g, k);
        if (l->idx < nxt.idx) {
          mid = PrevExtremum(nxt, g, k);
          if (l->idx < mid.idx) {
            prv = PrevExtremum(mid, g, k);
            while (l->idx < prv.idx) {
              // When edge is downhill, require mid to be maximum of derivative g
              // When edge is uphill,   require mid to be minimum of derivative g
              if (mid.min == l->min) {
                bool insert = false;
                // Require mid point to be a nearly stationary point
                if (abs(g[mid.idx]) <= _MinGradient && diff <= max_diff) {
                  insert = (abs(g[mid.idx] - g[prv.idx]) >= _MinGradient) &&
                           (abs(g[mid.idx] - g[nxt.idx]) >= _MinGradient);
                // If minimum is below GM threshold or intensity difference
                // between extrema is too big, insert interim point even if
                // there is only a subtle change of curvature
                } else {
                  const bool bg = (r->min && f[r->idx] < gm_thres) ||
                                  (l->min && f[l->idx] < gm_thres);
                  if (bg || diff > max_diff) {
                    const double min_g = .5 * _MinGradient;
                    const bool   left  = (abs(g[mid.idx] - g[prv.idx]) >= min_g);
                    const bool   right = (abs(g[mid.idx] - g[nxt.idx]) >= min_g);
                    insert = left || right;
                    if (insert && bg && (!left || !right)) {
                      if (mid.min) {
                        while (mid.idx > prv.idx && f[mid.idx] > _GlobalGreyMatterMean) --mid.idx;
                      } else {
                        while (mid.idx < prv_mid_idx && f[mid.idx] > _GlobalGreyMatterMean) ++mid.idx;
                      }
                    }
                  }
                }
                if (insert) {
                  mid.min = !r->min;
                  r = extrema.insert(r, mid);
                  mid.min = !mid.min;
                  r = extrema.insert(r, mid);
                  l = r - 1;
                }
                prv_mid_idx = mid.idx;
              }
              nxt = mid, mid = prv, prv = PrevExtremum(mid, g, k);
            }
          }
        }
      }
      --r;
    }
    // Remove extrema outside the surface which are cut off by BG or CSF
    r = CentralExtremum(extrema, k);
    while (r != extrema.end()) {
      if (f[r->idx] < _MinIntensity || f[r->idx] > _MaxIntensity) {
        if (r == extrema.begin()) {
          extrema.clear();
        } else {
          const int idx = r->idx;
          extrema.erase(r, extrema.end());
          // Insert a substitute point within GM for cases where the WM->GM->BG
          // edge would otherwise get lost by removing the BG minimum; however,
          // avoid creating an artificial WM->GM edge next to a very weak WM->GM
          // edge caused by partial volume where the cortex is very close to
          // the brain mask BG
          if (f[idx] < _MinIntensity) {
            const int i = extrema.back().idx;
            if (!extrema.back().min || abs(g[i]) >= _MaxGradient) {
              int j = idx - 1;
              while (i < j) {
                if (_MinIntensity <= f[j] && f[j] <= _MaxIntensity) {
                  if ((abs(g[j]) <= _MaxGradient && f[j] > gm_thres) || f[j-1] > _GlobalGreyMatterMean) {
                    if (abs(f[i] - f[j]) >= _MinGradient && abs(g[j]) < 1.5 * _GlobalGreyMatterSigma) {
                      extrema.push_back(Extremum(j, f[i] >= f[j]));
                    }
                    break;
                  }
                }
                --j;
              }
              if (i == j) {
                j = idx - 1;
                while (i < j) {
                  if (_MinIntensity <= f[j] && f[j] <= _MaxIntensity && abs(g[j] - g[j-1]) <= .1 * _MinGradient) {
                    if (abs(f[i] - f[j]) >= _MinGradient && abs(g[j]) < 1.5 * _GlobalGreyMatterSigma) {
                      extrema.push_back(Extremum(j, f[i] >= f[j]));
                    }
                    break;
                  }
                  --j;
                }
              }
            }
          }
        }
        break;
      }
      ++r;
    }
    // Trim extrema at ends caused by end-of-ray
    if (extrema.size() > 2) {
      // Remove last value if it does not significantly differ from last extremum
      r = extrema.end() - 1;
      if (r->idx == k || IsNaN(g[r->idx+1])) {
        l = r - 1;
        if (abs(f[l->idx] - f[r->idx]) <= _MinGradient) {
          extrema.erase(r);
        }
      }
      // Remove first value if it does not significantly differ from first extremum
      l = extrema.begin();
      if (l->idx == 0 || IsNaN(g[l->idx-1])) {
        r = l + 1;
        if (abs(f[l->idx] - f[r->idx]) <= _MinGradient) {
          extrema.erase(l);
        }
      }
    }
    // Remove intermediate extrema if too close
    // Don't do this in conjunction with the insertion of inflection points...
    #if 0
      double d, m;
      Extrema::iterator i;
      while (extrema.size() > 2) {
        l = extrema.begin() + 1;
        r = l + 1;
        m = _MinGradient;
        while (r != extrema.end() - 1) {
          d = abs(f[l->idx] - f[r->idx]);
          if (d < m) {
            i = l;
            m = d;
          }
          l = r++;
        }
        if (m < _MinGradient) {
          extrema.erase(i, i + 2);
        } else break;
      }
    #endif
    if (extrema.size() < 2) {
      extrema.clear();
      return;
    }
  }

  // ---------------------------------------------------------------------------
  /// Get either global or local normal distribution parameters
  inline void GetIntensityStatistics(const Point &p, const Vector3 &dp, int i, int k,
                                     double &mean, double &std, double &var,
                                     const double &global_mean,
                                     const double &global_std,
                                     const double &global_var,
                                     const LocalStatsImage *local_mean = nullptr,
                                     const LocalStatsImage *local_var  = nullptr) const
  {
    if (local_mean != nullptr && local_var != nullptr) {
      const Voxel v = RayVoxel(p, dp, i, k);
      mean = local_mean->Get(v.x, v.y, v.z);
      var  = local_var ->Get(v.x, v.y, v.z);
      std  = sqrt(var);
    } else {
      mean = global_mean;
      std  = global_std;
      var  = global_var;
    }
  }

  // ---------------------------------------------------------------------------
  /// Get either global or local normal distribution parameters of WM intensities
  inline void GetWhiteMatterStatistics(const Point &p, const Vector3 &dp,
                                       int i, int k, double &mean, double &std, double &var) const
  {
    GetIntensityStatistics(p, dp, i, k, mean, std, var,
        _GlobalWhiteMatterMean, _GlobalWhiteMatterSigma, _GlobalWhiteMatterVariance,
        _LocalWhiteMatterMean,  _LocalWhiteMatterVariance);
  }

  // ---------------------------------------------------------------------------
  /// Get either global or local normal distribution parameters of GM intensities
  inline void GetGreyMatterStatistics(const Point &p, const Vector3 &dp,
                                      int i, int k, double &mean, double &std, double &var) const
  {
    GetIntensityStatistics(p, dp, i, k, mean, std, var,
        _GlobalGreyMatterMean, _GlobalGreyMatterSigma, _GlobalGreyMatterVariance,
        _LocalGreyMatterMean,  _LocalGreyMatterVariance);
  }

  // ---------------------------------------------------------------------------
  /// Evaluate tissue probabilities and return position of central minimum
  inline void EvalExtremum(Extremum &extremum,
                           const Point &p, const Vector3 &dp,
                           const Array<double> &f, int k) const
  {
    double lower, upper, limit;
    const auto &value  = f[extremum.idx];
    if (_MinIntensity <= value && value <= _MaxIntensity) {
      if (extremum.min) {
        GetGreyMatterStatistics(p, dp, extremum.idx, k, extremum.mean, extremum.std, extremum.var);
        //limit = _GlobalWhiteMatterThreshold; //extremum.mean + 3. * extremum.std;
        limit = extremum.mean + 3. * extremum.std;
        upper = extremum.mean + .5 * extremum.std;
        lower = extremum.mean - 3. * extremum.std;
        // If minimum which looks like background is right next to clearly CSF,
        // still consider it as GM as long as it is not below the _MinIntensity
        // (which is GM mean minus 5 times the GM standard deviation).
        Extremum ext;
        if (((ext = NextExtremum(extremum, f, k)) && f[ext.idx] > _MaxIntensity) ||
            ((ext = PrevExtremum(extremum, f, k)) && f[ext.idx] > _MaxIntensity)) {
          lower = _MinIntensity;
        }
        if (value > upper) {
          extremum.prb = 1. - SShapedMembershipFunction(value, upper, limit);
        } else if (value < lower) {
          extremum.prb = SShapedMembershipFunction(value, _MinIntensity, lower);
        } else {
          extremum.prb = 1.;
        }
      } else {
        extremum.prb = 1.;
        GetWhiteMatterStatistics(p, dp, extremum.idx, k, extremum.mean, extremum.std, extremum.var);
        upper = extremum.mean + 1.5 * extremum.std; //+ 3. * extremum.std;
        lower = _GlobalWhiteMatterThreshold;//extremum.mean - 1. * extremum.std;
        limit = extremum.mean - 3. * extremum.std;
        if (_VentriclesDistance != nullptr) {
          const Voxel v = RayVoxel(p, dp, extremum.idx, k);
          if (_VentriclesDistance->Get(v.x, v.y, v.z) < .5) {
            extremum.prb = 0.;
            upper = lower = NaN;
          } else if (_VentriclesDistance->Get(v.x, v.y, v.z) < 2.) {
            upper = extremum.mean - 1. * extremum.std;
            lower = extremum.mean - 3. * extremum.std;
            limit = extremum.mean - 5. * extremum.std;
          }
        }
        if (value > upper) {
          extremum.prb = 1. - SShapedMembershipFunction(value, upper, _MaxIntensity);
        } else if (value < lower) {
          extremum.prb = SShapedMembershipFunction(value, limit, lower);
        }
      }
    } else {
      extremum.prb = 0.;
    }
  }

  // ---------------------------------------------------------------------------
  /// Evaluate tissue probabilities and return position of central minimum
  inline void EvalExtrema(Extrema &extrema,
                          const Point &p, const Vector3 &dp,
                          const Array<double> &f, int k) const
  {
    for (auto &&extremum : extrema) {
      EvalExtremum(extremum, p, dp, f, k);
    }
  }

  // ---------------------------------------------------------------------------
  /// Check if edge may belong to white surface boundary
  inline bool IsNeonatalWhiteSurfaceEdge(const Extrema::iterator &i, const Extrema::iterator &j,
                                         const Array<double> &f, const Array<double> &g, int k) const
  {
    if (i->idx < j->idx && !i->min && j->min && i->prb > .5 && j->prb > .5) {
      const double slope = -MinimumValue(g, i->idx, j->idx);
      if (slope > _MinGradient) return true;
    }
    return false;
  }

  // ---------------------------------------------------------------------------
  /// Check if found white surface boundary is opposite to the desired
  /// boundary, i.e., belongs to a neighboring gyrus than this one
  inline bool IsOtherNeonatalWhiteSurfaceEdge(const Extrema &extrema, const Point &p, const Vector3 &dp,
                                              const Extrema::iterator &i, const Extrema::iterator &j,
                                              const Array<double> &f, const Array<double> &g, int k) const
  {
    const int idx = (j == extrema.end() ? NextExtremum(*i, f, k) : *j).idx + 1;
    if (idx != -1 && idx <= k && IsNaN(f[idx]) && !IsNaN(g[idx])) {
      if (_CerebellumDistance != nullptr) {
        const Voxel v = RayVoxel(p, dp, idx, k);
        if (_CerebellumDistance->Get(v.x, v.y, v.z) < .5) return false;
      }
      return true;
    }
    return false;
  }

  // ---------------------------------------------------------------------------
  /// Find image edge of neonatal white surface given the two edge extrema
  inline int FindNeonatalWhiteSurface(Extrema::iterator i, Extrema::iterator j,
                                      const Array<double> &f, const Array<double> &g, int k) const
  {
    int edge;
    // If minimum is not an actual minimum but interim point where intensity
    // falls rapidly below GM mean towards background, move inwards to
    // previous maximum in gradient function and then further inwards below
    // the preceeding minimum of the gradient function, i.e., the first
    // downhill part of the WM->GM->BG transition
    if (g[j->idx] < -_MinGradient) {
      edge = j->idx - 1;
      while (edge > i->idx && g[edge] < g[edge-1]) --edge;
      while (edge > i->idx && g[edge] > g[edge-1]) --edge;
      if (edge == i->idx) edge = -1;
    } else {
      edge = -1;
      // 1. Find edge stronger than _MaxGradient closest to GM minimum
      double min_gradient   = -_MinGradient;
      int    strongest_edge = -1;
      Extremum ext(j->idx);
      while ((ext = PrevExtremum(ext, g, k)).idx > i->idx) {
        if (ext.min) {
          if (g[ext.idx] < -_MaxGradient) {
            edge = ext.idx;
            break;
          }
          if (g[ext.idx] < min_gradient) {
            strongest_edge = ext.idx;
            min_gradient   = 1.5 * g[ext.idx];
          }
        }
      }
      // 2. If no such edge found, use strongest edge
      if (edge == -1) edge = strongest_edge;
    }
    // 3. Last resort is middle point between WM max and GM min
    if (edge == -1) edge = (i->idx + j->idx) / 2;
    // 4. Move outwards if edge is steep and intensity is yet above threshold
    if (g[edge] < 1.5 * _GlobalWhiteMatterSigma && f[j->idx] < _GlobalGreyMatterMean) {
      int idx = edge;
      while (idx < j->idx && f[idx] > _GlobalWhiteMatterThreshold) ++idx;
      if (idx > edge) {
        edge = idx++;
        if (_GlobalWhiteMatterThreshold - f[edge] > f[idx] - _GlobalWhiteMatterThreshold) {
          edge = idx;
        }
      }
    }
    return edge;
  }

  // ---------------------------------------------------------------------------
  /// Find image edge of neonatal pial surface given the two edge extrema
  inline int FindNeonatalPialSurface(const Extrema::iterator &i,
                                     const Extrema::iterator &j,
                                     const Array<double> &g, int k,
                                     double min_gradient) const
  {
    int    idx = -1;
    double min = min_gradient;
    Extremum prv = Extremum(i->idx, true);
    Extremum nxt = prv;
    while (nxt.idx < j->idx) {
      nxt = NextExtremum(prv, g, k);
      if (nxt.idx == -1) break;
      if (prv.min && !nxt.min && g[nxt.idx] > min) {
        idx = nxt.idx;
        min = g[idx] + _MinGradient;
        if (g[idx] > _GlobalGreyMatterSigma) break;
      }
      prv = nxt;
    }
    return idx;
  }

  // ---------------------------------------------------------------------------
  /// Find image edge of WM/cGM boundary in T2-weighted MRI of neonatal brain
  ///
  /// The initial surface for the deformation process is the white surface
  /// obtained by deforming a sphere/convex hull towards the white matter
  /// tissue segmentation mask. The surface thus is close to the target boundary
  /// and should only be refined using this force.
  inline int NeonatalWhiteSurface(int ptId, const Point &p, const Vector3 &dp,
                                  const Array<double> &f, const Array<double> &g,
                                  Extrema &extrema, Extrema::iterator &i,
                                  Extrema::iterator &j, bool dbg = false) const
  {
    const int k  = static_cast<int>(g.size()) - 1;
    const int i0 = k / 2;

    // Find relevant extrema of intensity profile
    FindExtrema(extrema, f, k);
    #if BUILD_WITH_DEBUG_CODE
      if (dbg) {
        cout << "\n\ti=[";
        for (i = extrema.begin(); i != extrema.end(); ++i) {
          if (i != extrema.begin()) cout << ", ";
          cout << i->idx;
        }
        cout << "];";
      }
    #endif
    CleanExtrema(extrema, f, g, k);
    #if BUILD_WITH_DEBUG_CODE
      if (dbg) {
        cout << "\n\tj=[";
        for (i = extrema.begin(); i != extrema.end(); ++i) {
          if (i != extrema.begin()) cout << ", ";
          cout << i->idx;
        }
        cout << "];";
      }
    #endif

    // Evaluate tissue probabilities and initialize search using next minimum inside
    Extrema::iterator m = CentralExtremum(extrema, k, true);
    if (m == extrema.end()) {
      i = j = extrema.end();
      return i0;
    }

    // Avoid mistakes when right next to ventricles
    if (_VentriclesDistance != nullptr) {
      Voxel v(iround(p.x), iround(p.y), iround(p.z));
      const double ventricles_distance = _VentriclesDistance->Get(v.x, v.y, v.z);
      // When inside a ventricle, move outwards until the surface no longer
      // intersects the ventricles
      if (ventricles_distance < .1) {
        #if BUILD_WITH_DEBUG_CODE
          if (dbg) cout << "\n\tinside ventricles, move outwards" << endl;
        #endif
        int idx;
        for (idx = i0; idx < k && !IsNaN(g[idx]); ++idx) {
          v = RayVoxel(p, dp, idx, k);
          if (_VentriclesDistance->Get(v.x, v.y, v.z) > .1) break;
        }
        if (IsNaN(g[idx])) return i0;
        return idx;
      // When right next to a ventricle, the following assumptions of valid
      // image edges may be incorrect; thus better stay and rely on the segmentation.
      } else if (ventricles_distance < 1.) {
        #if BUILD_WITH_DEBUG_CODE
          if (dbg) cout << "\n\tnext to ventricles, don't move" << endl;
        #endif
        return i0;
      }
    }

    // Whether correction of found WM->dGM edge near the ventricles is allowed
    // given that the intensity profile matches a WM->dGM->cGM transition
    bool allow_deep_matter_correction = (_CorticalHullDistance != nullptr);

    // Find probable WM/cGM edge inside/outside the surface
    EvalExtrema(extrema, p, dp, f, k);
    #if BUILD_WITH_DEBUG_CODE
      if (dbg) {
        cout << "\n\t";
        for (i = extrema.begin(); i != extrema.end(); ++i) {
          if (i != extrema.begin()) cout << ", ";
          cout << "Pr(" << (i->min ? "GM" : "WM") << "|" << f[i->idx] << ") = " << i->prb;
        }
        cout << "};";
      }
    #endif
    Extrema::iterator pos1 = extrema.end();
    Extrema::iterator pos2 = extrema.end();
    double prb1 = 0., prb2 = 0., prb, slope;
    int    nop1 = 0,  nop2 = 0;
    if (m != extrema.begin()) {
      i = m;
      do {
        j = i--;
        if (i->idx < j->idx) {
          if (i->min) {
            slope = MaximumValue(g, i->idx, j->idx);
            if (slope > _MinGradient) {
              if (i->prb > .8 && j->prb > .8) {
                if (i->idx >= i0) {
                  ++nop2;
                } else {
                  ++nop1;
                }
              }
              if (pos1 != extrema.end()) break;
            }
          } else if (IsNeonatalWhiteSurfaceEdge(i, j, f, g, k)) {
            prb = i->prb * j->prb;
            if (prb > prb1) {
              pos1 = i;
              prb1 = prb;
            }
          }
        }
      } while (i != extrema.begin());
    }
    j = m + 1;
    while (j != extrema.end()) {
      i = j - 1;
      if (i->idx < j->idx) {
        if (i->min) {
          if (i->idx >= i0) {
            slope = MaximumValue(g, i->idx, j->idx);
            if (slope > _MinGradient) {
              if (i->prb > .8 && j->prb > .8) {
                #if BUILD_WITH_DEBUG_CODE
                  if (dbg) {
                    cout << "\n\topposite WM/GM edge [" << i->idx << ", " << j->idx << "] encountered, look no further outwards";
                  }
                #endif
                break;
              }
              if (slope > 1.5 * _GlobalWhiteMatterSigma) {
                #if BUILD_WITH_DEBUG_CODE
                  if (dbg) {
                    cout << "\n\tstrong opposite edge [" << i->idx << ", " << j->idx << "] encountered, look no further outwards";
                  }
                #endif
                break;
              }
            }
          }
        } else if (IsNeonatalWhiteSurfaceEdge(i, j, f, g, k)) {
          if (IsOtherNeonatalWhiteSurfaceEdge(extrema, p, dp, j, j + 1, f, g, k)) {
            // Close to the superior of corpus callosum and the lateral ventricles,
            // there are small sulci where the cortex begins, keep this edge even
            // when close to another
            if (_VentriclesDistance && f[j->idx] > _GlobalGreyMatterMean - 1.5 * _GlobalGreyMatterSigma) {
              const Voxel v = RayVoxel(p, dp, i->idx, k);
              if (_VentriclesDistance->Get(v.x, v.y, v.z) < 5.) {
                pos2 = i;
                prb2 = i->prb * j->prb;
              }
            }
            if (pos2 == extrema.end()) {
              #if BUILD_WITH_DEBUG_CODE
                if (dbg) {
                  cout << "\n\tedge [" << i->idx << ", " << j->idx << "] is opposite to another WM/cGM edge";
                }
              #endif
              // When there is a stationary inflection point and the second
              // downhill part of the edge (i, j) is next to a neighboring gyrus,
              // choose the downhill part corresponding to the previous minimum
              // in the gradient function as white surface boundary edge.
              Extremum ext = PrevExtremum(Extremum(j->idx), g, k);
              if (ext.idx > i->idx && ext.min) {
                Extremum mid = PrevExtremum(ext, g, k);
                if (mid.idx > i->idx && abs(g[mid.idx]) < _MinGradient) {
                  ext = PrevExtremum(mid, g, k);
                  if (ext.idx > i->idx) {
                    EvalExtremum(mid, p, dp, f, k);
                    j = extrema.insert(j, mid); // need alternating sequence of min/max
                    mid.min = true;
                    j = extrema.insert(j, mid);
                    pos2 = i;
                    prb2 = i->prb * j->prb;
                  }
                }
              }
              // Otherwise, choose the previous outside edge instead which may
              // have only a slight GM minimum above the GM mean due to partial
              // volume with nearby CSF and was therefore skipped before.
              if (pos2 == extrema.end() && distance(extrema.begin(), i) > 1) {
                i -= 2, j = i + 1;
                prb = i->prb * j->prb;
                if (pos1 != i && prb > .1) {
                  pos2 = i;
                  prb2 = prb;
                }
              }
            }
            break;
          } else {
            prb = i->prb * j->prb;
            #if BUILD_WITH_DEBUG_CODE
              if (dbg) {
                cout << "\n\tPr([" << i->idx << ", " << j->idx << "]) = " << prb;
              }
            #endif
            if (prb > prb2) {
              pos2 = i;
              prb2 = prb;
            }
          }
        } else {
          #if BUILD_WITH_DEBUG_CODE
            if (dbg) {
              cout << "\n\tskip edge [" << i->idx << ", " << j->idx << "], not a WM/cGM edge";
            }
          #endif
        }
      }
      ++j;
    }
    #if BUILD_WITH_DEBUG_CODE
      if (dbg) {
        cout << "\n\tmid = " << m->idx;
        if (prb1 > 0.) {
          cout << "; inwards: edge = [" << pos1->idx << ", " << (pos1+1)->idx << "], prb = " << prb1 << ", nop = " << nop1;
        } else {
          cout << "; inwards: edge = None, nop = " << nop1;
        }
        if (prb2 > 0.) {
          cout << "; outwards: edge = [" << pos2->idx << ", " << (pos2+1)->idx << "], prb = " << prb2 << ", nop = " << nop2;
        } else {
          cout << "; outwards: edge = None";
        }
      }
    #endif

    // Choose pair of maximum and minimum within which image edge occurs
    i = extrema.end();
    if (pos1 != extrema.end() && pos2 != extrema.end()) {
      // If the outwards edge is opposite to bright CSF, choose it even if
      // its "probability" score is lower than the inwards edge as long as
      // it is high enough and nop2 is zero
      if (nop1 == 0 && prb2 > .5) {
        j = pos2 + 1;
        if (MaximumValue(g, pos2->idx, j->idx) < -_GlobalGreyMatterSigma) {
          Extremum ext = NextExtremum(*j, f, k);
          if (ext.idx != -1 && f[ext.idx] > _MaxIntensity) i = pos2;
        }
      }
      if (i == extrema.end()) {
        #if 0
        if (nop1 == nop2) {
          i = (prb1 >= prb2 ? pos1 : pos2);
        } else {
          i = (nop1 < nop2 ? pos1 : pos2);
        }
        #else
        if (nop1 == 0) {
          i = (prb1 >= prb2 ? pos1 : pos2);
        } else {
          i = pos1;
        }
        #endif
      }
    } else if (pos1 != extrema.end()) {
      i = pos1;
    } else if (pos2 != extrema.end() && nop2 == 0) {
      i = pos2;
    }
    j = i + 1;
    if (i == extrema.end() || j == extrema.end() || i->min) {
      #if BUILD_WITH_DEBUG_CODE
        if (dbg) {
          cout << "\n\tno suitable edge found";
        }
      #endif
      i = j = extrema.end();
      return i0;
    }

    // Identify strongest (negative) image gradient between these extrema;
    // prefer image edges closer to GM minimum over those near WM maximum;
    // require maximum in gradient to the right of this minimum to avoid
    // finding a strong "minimum" in the gradient near foreground boundary
    int edge = FindNeonatalWhiteSurface(i, j, f, g, k);
    #if BUILD_WITH_DEBUG_CODE
      if (dbg) {
        cout << "\n\tWM/cGM edge index = " << edge;
      }
    #endif
    const Voxel v = RayVoxel(p, dp, edge, k);
    if (_CorticalDeepGreyMatterBoundingBox) {
      const int * const &bounds = _CorticalDeepGreyMatterBoundingBox;
      if (v.x < bounds[0] || v.x > bounds[1] ||
          v.y < bounds[2] || v.y > bounds[3] ||
          v.z < bounds[4] || v.z > bounds[5]) {
        allow_deep_matter_correction = false;
        #if BUILD_WITH_DEBUG_CODE
          if (dbg) {
            cout << "\n\tedge is not within WM->dGM->cGM bounding box";
          }
        #endif
      }
    }
    if (allow_deep_matter_correction &&
        MinimumValue(g, i->idx, j->idx) < -1.5 * _GlobalWhiteMatterSigma) {
      #if BUILD_WITH_DEBUG_CODE
        if (dbg) {
          cout << "\n\tedge is strong, skip WM->dGM->cGM correction";
        }
      #endif
      allow_deep_matter_correction = false;
    }
    if (allow_deep_matter_correction) {
      // If the found edge is followed by another decrease of intensity
      // towards a relatively dark cortical GM where the start of this
      // following edge is within darker WM or deep GM, use this next
      // edge instead which is further outside. But only when this edge
      // was not previously identified as near to a neighboring gyrus.
      const auto a = j + 1, b = j + 2;
      if (a != extrema.end() && b != extrema.end()) {
        if (abs(g[b->idx]) < _MaxGradient) {
          if (IsOtherNeonatalWhiteSurfaceEdge(extrema, p, dp, b, b + 1, f, g, k)) {
            #if BUILD_WITH_DEBUG_CODE
              if (dbg) {
                cout << "\n\tfollowing edge is opposite to another WM/cGM edge";
              }
            #endif
          } else {
            // Use approximate distance maps to outer cortical surface
            // (boundary of union of GM and WM tissue segmentation) and distance
            // to ventricles to decide when to perform this correction
            double cortex_distance = _CorticalHullDistance->Get(v.x, v.y, v.z);
            #if BUILD_WITH_DEBUG_CODE
              if (dbg) {
                cout << "\n\tf[i=" << i->idx << "]=" << f[i->idx] << ", f[j=" << j->idx << "]=" << f[j->idx]
                     << ", f[a=" << a->idx << "]=" << f[a->idx] << ", f[b=" << b->idx << "]=" << f[b->idx]
                     << ", cortex distance = " << cortex_distance;
              }
            #endif
            if (cortex_distance > 5.) {
              double vents_distance = NaN;
              if (_VentriclesDistance != nullptr) {
                vents_distance = _VentriclesDistance->Get(v.x, v.y, v.z);
              }
              #if BUILD_WITH_DEBUG_CODE
                if (dbg) {
                  cout << ", ventricles distance = " << vents_distance;
                }
              #endif
              // Default parameters
              double min_max_f_delta = max(1.5 * _GlobalGreyMatterSigma, .5 * (f[i->idx] - f[j->idx]));
              double max_max_f_delta = 0.5 * _GlobalWhiteMatterSigma;
              double min_min_f_delta = 0.5 * _GlobalGreyMatterSigma;
              double min_max_g_limit = 2.0 * _MaxGradient;
              double min_g_value     = -_GlobalGreyMatterSigma;
              // ...relax parameters nearer the ventricles
              if (vents_distance < 5.) {
                min_max_f_delta = 3.0 * _GlobalGreyMatterSigma;
                min_g_value    *= .5;
              }
              #if 0 // too relaxed for dHCP CC00050XX01
                if (vents_distance < 2.5) {
                  min_max_f_delta = inf;
                  max_max_f_delta = 0.;
                  min_min_f_delta = -.5 * _GlobalGreyMatterSigma;
                }
              #endif
              // Do not perform such correction if the intensity profile is
              // high -> low -> less high -> lower -> low --> even lower,
              // i.e., when we picked already the "lower -> low" interval
              if (distance(extrema.begin(), i) > 1) {
                const auto c = i - 2;
                const auto d = i - 1;
                if (f[i->idx] < _GlobalGreyMatterMean + 1.5 * _GlobalGreyMatterSigma
                    // 1. Difference between GM minimum and next maximum is low
                    && f[i->idx] - f[d->idx] < min_max_f_delta
                    // 2. Next maximum is below WM maximum
                    && f[c->idx] - f[i->idx] > max_max_f_delta
                    // 3. Next minimum is lower than this GM minimum
                    && f[d->idx] - f[j->idx] > min_min_f_delta
                    // 4. Edge from GM minimum to next maximum is not too strong
                    && MaximumValue(g, d->idx, i->idx) < min_max_g_limit) {
                  allow_deep_matter_correction = false;
                  #if BUILD_WITH_DEBUG_CODE
                    if (dbg) {
                      cout << "\n\tis probably already dGM->cGM edge of WM->dGM->cGM transition (case 1)";
                    }
                  #endif
                } else if (f[c->idx] > _GlobalWhiteMatterMean + 1.5 * _GlobalWhiteMatterSigma
                           && f[i->idx] < _GlobalWhiteMatterMean + .5 * _GlobalWhiteMatterSigma
                           && f[i->idx] > _GlobalWhiteMatterThreshold) {
                  allow_deep_matter_correction = false;
                  #if BUILD_WITH_DEBUG_CODE
                    if (dbg) {
                      cout << "\n\tis probably already dGM->cGM edge of WM->dGM->cGM transition (case 2)";
                    }
                  #endif
                }
              }
              if (allow_deep_matter_correction) {
                // Need to distinguish WM->GM->BG from WM->dGM->cGM transition
                // Use subdivide-brain-image -output-inner-cortical-distance to
                // generate cortical depth map.
                if (f[b->idx] < _GlobalGreyMatterMean - 3. * _GlobalGreyMatterSigma && (cortex_distance < 5. || vents_distance > 20.)) {
                  #if BUILD_WITH_DEBUG_CODE
                    if (dbg) {
                      cout << "\n\tis possibly WM->GM->BG transition, keep first found edge";
                    }
                  #endif
                } else {
                  #if BUILD_WITH_DEBUG_CODE
                    if (dbg) {
                      cout << "\n\tupper min_max_f_delta = " << min_max_f_delta << " (difference = " << f[a->idx] - f[j->idx] << ")"
                           <<   ", lower max_max_f_delta = " << max_max_f_delta << " (difference = " << f[i->idx] - f[a->idx] << ")"
                           <<   ", lower min_min_f_delta = " << min_min_f_delta << " (difference = " << f[j->idx] - f[b->idx] << ")"
                           <<   ", upper min_max_g_limit = " << min_max_g_limit << " (max. grad. = " << MaximumValue(g, j->idx, a->idx) << ")";
                    }
                  #endif
                  if (// 1. Difference between GM minimum and next maximum is low
                      f[a->idx] - f[j->idx] < min_max_f_delta
                      // 2. Next maximum is below WM maximum
                      && f[i->idx] - f[a->idx] > max_max_f_delta
                      // 3. Next minimum is lower than this GM minimum
                      && f[j->idx] - f[b->idx] > min_min_f_delta
                      // 4. Edge from GM minimum to next maximum is not too strong
                      && MaximumValue(g, j->idx, a->idx) < min_max_g_limit) {
                    // Following edge is relatively strong or at least stronger than the previous one
                    const int idx = FindNeonatalWhiteSurface(a, b, f, g, k);
                    if (g[idx] < min_g_value || g[idx] < g[edge] - _MinGradient || MaximumValue(g, edge, idx) < 0.) {
                      i = a, j = b;
                      edge = idx;
                      #if BUILD_WITH_DEBUG_CODE
                        if (dbg) {
                          cout << "\n\tnew WM/cGM edge index = " << edge;
                        }
                      #endif
                    } else {
                      #if BUILD_WITH_DEBUG_CODE
                        if (dbg) {
                          cout << "\n\tnew WM/cGM edge not strong enough";
                        }
                      #endif
                    }
                  }
                }
              }
            }
          }
        } else {
          #if BUILD_WITH_DEBUG_CODE
            if (dbg) {
              cout << "\n\tno WM->dGM->cGM correction due to high gradient at next minimum";
            }
          #endif
        }
      } else {
        #if BUILD_WITH_DEBUG_CODE
          if (dbg) {
            cout << "\n\tless than two extrema left outwards";
          }
        #endif
      }
    } else {
      #if BUILD_WITH_DEBUG_CODE
        if (dbg) {
          cout << "\n\tWM->dGM->cGM correction disabled";
        }
      #endif
    }
    return edge;
  }

  // ---------------------------------------------------------------------------
  /// Find image edge of cGM/CSF boundary in T2-weighted MRI of neonatal brain
  inline int NeonatalPialSurface(const Point &p, const Vector3 &dp,
                                 const Array<double> &f, const Array<double> &g,
                                 Extrema &extrema, Extrema::iterator &i,
                                 Extrema::iterator &j, bool dbg = false) const
  {
    const int k = static_cast<int>(f.size()) - 1;

    int edge = -1;
    NeonatalWhiteSurface(0, p, dp, f, g, extrema, i, j, dbg);
    if (i != extrema.end() && j != extrema.end() && j->min) {
      i = j, j = i + 1;
      if (j != extrema.end()) {
        edge = FindNeonatalPialSurface(i, j, g, k, _MinGradient);
      }
    }
    if (edge == -1) {
      edge = ClosestMaximum(g);
    }
    #if BUILD_WITH_DEBUG_CODE
      if (dbg) {
        cout << "\n\tGM/CSF edge index = " << edge;
      }
    #endif

    return edge;
  }

  // ---------------------------------------------------------------------------
  void operator ()(const blocked_range<int> &ptIds) const
  {
    const int k = _NumberOfSamples - 1;
    const int r = k / 2;

    double  value;
    int     i, j, j1, j2;
    Point   p;
    Vector3 n;

    bool dbg = false;
    const bool cortex = (_EdgeType == ImageEdgeDistance::NeonatalWhiteSurface ||
                         _EdgeType == ImageEdgeDistance::NeonatalPialSurface);

    Array<double> g(k+1), f;
    Extrema extrema;
    Extrema::iterator a, b;
    if (cortex) {
      f.resize(g.size());
      extrema.reserve(10);
    }

    for (int ptId = ptIds.begin(); ptId != ptIds.end(); ++ptId) {
      if (_Status && _Status->GetComponent(ptId, 0) == 0.) {
        _Distances->SetComponent(ptId, 0, 0.);
        _Magnitude->SetComponent(ptId, 0, 0.);
        continue;
      }
      // Get point position and scaled normal
      _Points ->GetPoint(ptId, p);
      _Normals->GetTuple(ptId, n), n *= _StepLength;
      // Transform point/vector to image space
      _Image->WorldToImage(p);
      _Image->WorldToImage(n);
      // Sample image gradient/intensities along ray
      SampleGradient(g, k, p, n);
      #if BUILD_WITH_DEBUG_CODE
        if (_GradientSamples) {
          for (int i = 0; i < _NumberOfSamples; ++i) {
            _GradientSamples->Put(ptId, i, g[i]);
          }
        }
      #endif
      if (cortex) {
        SampleIntensity(f, g, k, p, n);
        #if BUILD_WITH_DEBUG_CODE
          if (_IntensitySamples) {
            for (int i = 0; i < _NumberOfSamples; ++i) {
              _IntensitySamples->Put(ptId, i, f[i]);
            }
          }
        #endif
      }
      // Choose points for which to print the values and extrema indices
      // for visualization and analysis in MATLAB, for example
      // - Copy and paste output into MATLAB to define f, g, i, and j
      // - Plot with the following MATLAB code:
      //     x=1:size(f,2);
      //     yyaxis left, plot(x, f), hold on
      //     plot(x(i+1), f(i+1), 'r*')
      //     plot(x(j+1), f(j+1), 'g+')
      //     yyaxis right, plot(x, g), hold off
      #if BUILD_WITH_DEBUG_CODE
        dbg = (p.Distance(dbg_voxel) < dbg_dist);
        if (dbg) {
          cout << "\nPoint " << ptId << ":\n\tf=[";
          for (size_t i = 0; i < f.size(); ++i) {
            if (i > 0) cout << ", ";
            cout << f[i];
          }
          cout << "];\n\tg=[";
          for (size_t i = 0; i < g.size(); ++i) {
            if (i > 0) cout << ", ";
            cout << g[i];
          }
          cout << "];";
        }
      #endif
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
        case ImageEdgeDistance::NeonatalWhiteSurface: {
          j = NeonatalWhiteSurface(ptId, p, n, f, g, extrema, a, b, dbg);
        } break;
        case ImageEdgeDistance::NeonatalPialSurface: {
          j = NeonatalPialSurface(p, n, f, g, extrema, a, b, dbg);
        } break;
      }
      // When intensity thresholds set, use them to ignore irrelevant edges
      if (_EdgeType != ImageEdgeDistance::NeonatalWhiteSurface &&
          _EdgeType != ImageEdgeDistance::NeonatalPialSurface) {
        if (j != r && (!IsInf(_MinIntensity) || !IsInf(_MaxIntensity))) {
          value = SampleIntensity(p, n, j, k);
          if (value < _MinIntensity || value > _MaxIntensity) {
            j = r;
          }
        }
        if (j != r && !IsInf(_Padding)) {
          if (j < r) {
            for (i = r; i > 0; --i) {
              if (f[i] < _Padding) {
                i = 0;
                break;
              }
              if (g[j] * g[i] < 0.) break;
            }
            if (i == 0) j = r;
          } else if (j > r) {
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
      }
      // Set point distance to found edge and edge strength
      _Distances->SetComponent(ptId, 0, static_cast<double>(j - r) * _StepLength);
      _Magnitude->SetComponent(ptId, 0, IsNaN(g[j]) ? 0. : abs(g[j]));
      #if BUILD_WITH_DEBUG_CODE
        if (dbg) cout << endl;
      #endif
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
        //m1 = _Magnitude->GetComponent(ptId, 0);
        //m1 = SShapedMembershipFunction(m1, 0., _MaxMagnitude);
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
  } else if (lstr == "neonatal white surface" || lstr == "neonatal white" ||
             lstr == "neonatal t2-w wm/cgm"   || lstr == "neonatal t2-w cgm/wm") {
    value = ImageEdgeDistance::NeonatalWhiteSurface;
  } else if (lstr == "neonatal pial surface" || lstr == "neonatal pial" ||
             lstr == "neonatal t2-w cgm/csf" || lstr == "neonatal t2-w csf/cgm") {
    value = ImageEdgeDistance::NeonatalPialSurface;
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
    case ImageEdgeDistance::Extremum:             { str = "Extremum"; } break;
    case ImageEdgeDistance::ClosestMinimum:       { str = "ClosestMinimum"; } break;
    case ImageEdgeDistance::ClosestMaximum:       { str = "ClosestMaximum"; } break;
    case ImageEdgeDistance::ClosestExtremum:      { str = "ClosestExtremum"; } break;
    case ImageEdgeDistance::StrongestMinimum:     { str = "StrongestMinimum"; } break;
    case ImageEdgeDistance::StrongestMaximum:     { str = "StrongestMaximum"; } break;
    case ImageEdgeDistance::StrongestExtremum:    { str = "StrongestExtremum"; } break;
    case ImageEdgeDistance::NeonatalWhiteSurface: { str = "Neonatal T2-w WM/cGM"; } break;
    case ImageEdgeDistance::NeonatalPialSurface:  { str = "Neonatal T2-w cGM/CSF"; } break;
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
  _Padding            = other._Padding;
  _MinIntensity       = other._MinIntensity;
  _MaxIntensity       = other._MaxIntensity;
  _MinGradient        = other._MinGradient;
  _MaxDistance        = other._MaxDistance;
  _MedianFilterRadius = other._MedianFilterRadius;
  _DistanceSmoothing  = other._DistanceSmoothing;
  _MagnitudeSmoothing = other._MagnitudeSmoothing;
  _StepLength         = other._StepLength;

  _WhiteMatterMask           = other._WhiteMatterMask;
  _GreyMatterMask            = other._GreyMatterMask;
  _CorticalHullDistance      = other._CorticalHullDistance;
  _VentriclesDistance        = other._VentriclesDistance;
  _CerebellumDistance        = other._CerebellumDistance;
  _WhiteMatterWindowWidth    = other._WhiteMatterWindowWidth;
  _GreyMatterWindowWidth     = other._GreyMatterWindowWidth;
  _GlobalWhiteMatterMean     = other._GlobalWhiteMatterMean;
  _GlobalWhiteMatterVariance = other._GlobalWhiteMatterVariance;
  _GlobalGreyMatterMean      = other._GlobalGreyMatterMean;
  _GlobalGreyMatterVariance  = other._GlobalGreyMatterVariance;
  _LocalWhiteMatterMean      = other._LocalWhiteMatterMean;
  _LocalWhiteMatterVariance  = other._LocalWhiteMatterVariance;
  _LocalGreyMatterMean       = other._LocalGreyMatterMean;
  _LocalGreyMatterVariance   = other._LocalGreyMatterVariance;
}

// -----------------------------------------------------------------------------
ImageEdgeDistance::ImageEdgeDistance(const char *name, double weight)
:
  SurfaceForce(name, weight),
  _EdgeType(Extremum),
  _Padding(-inf),
  _MinIntensity(-inf),
  _MaxIntensity(+inf),
  _MinGradient(NaN),
  _MaxGradient(NaN),
  _MaxDistance(0.),
  _MedianFilterRadius(0),
  _DistanceSmoothing(0),
  _MagnitudeSmoothing(2),
  _StepLength(1.),
  _WhiteMatterMask(nullptr),
  _GreyMatterMask(nullptr),
  _CorticalHullDistance(nullptr),
  _VentriclesDistance(nullptr),
  _CerebellumDistance(nullptr),
  _WhiteMatterWindowWidth(0),
  _GreyMatterWindowWidth(0),
  _GlobalWhiteMatterMean(NaN),
  _GlobalWhiteMatterVariance(NaN),
  _GlobalGreyMatterMean(NaN),
  _GlobalGreyMatterVariance(NaN)
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
  if (strcmp(param, "Minimum gradient") == 0 || strcmp(param, "Minimum gradient magnitude") == 0) {
    return FromString(value, _MinGradient);
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
  if (strcmp(param, "Local white matter window width") == 0) {
    return FromString(value, _WhiteMatterWindowWidth);
  }
  if (strcmp(param, "Local white matter window radius") == 0) {
    int radius;
    if (!FromString(value, radius)) return false;
    _WhiteMatterWindowWidth = 2 * radius + 1;
    return true;
  }
  if (strcmp(param, "Local grey matter window width") == 0) {
    return FromString(value, _GreyMatterWindowWidth);
  }
  if (strcmp(param, "Local grey matter window radius") == 0) {
    int radius;
    if (!FromString(value, radius)) return false;
    _GreyMatterWindowWidth = 2 * radius + 1;
    return true;
  }
  if (strcmp(param, "Local window width") == 0) {
    int width;
    if (!FromString(value, width)) return false;
    _WhiteMatterWindowWidth = _GreyMatterWindowWidth = width;
    return false;
  }
  if (strcmp(param, "Local window radius") == 0) {
    int radius;
    if (!FromString(value, radius)) return false;
    _WhiteMatterWindowWidth = _GreyMatterWindowWidth = 2 * radius + 1;
    return true;
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
  InsertWithPrefix(params, "Minimum gradient magnitude", _MinGradient);
  InsertWithPrefix(params, "Median filter radius", _MedianFilterRadius);
  InsertWithPrefix(params, "Smoothing iterations", _DistanceSmoothing);
  InsertWithPrefix(params, "Magnitude smoothing",  _MagnitudeSmoothing);
  InsertWithPrefix(params, "Local white matter window width", _WhiteMatterWindowWidth);
  InsertWithPrefix(params, "Local grey matter window width", _GreyMatterWindowWidth);
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

  #if BUILD_WITH_DEBUG_CODE
    cout << "\n" << NameOfClass() << "::" << __FUNCTION__ << ":";
  #endif

  // Image resolution, i.e., length of voxel diagonal
  const double res = sqrt(pow(_Image->XSize(), 2) +
                          pow(_Image->YSize(), 2) +
                          pow(_Image->ZSize(), 2));

  // Parameters for ray casting to sample image intensities near surface
  _StepLength = .25 * res;
  if (_MaxDistance <= 0.) _MaxDistance = 4. * res;
  #if BUILD_WITH_DEBUG_CODE
    cout << "\n\tstep length = " << _StepLength << ", max distance = " << _MaxDistance;
  #endif

  // Add point data arrays
  AddPointData("Distance");
  AddPointData("Magnitude");

  // Calculate image intensity statistics
  _LocalWhiteMatterMean.Clear();
  _LocalWhiteMatterVariance.Clear();
  _LocalGreyMatterMean.Clear();
  _LocalGreyMatterVariance.Clear();
  if (_EdgeType == NeonatalWhiteSurface || _EdgeType == NeonatalPialSurface) {
    ImageAttributes attr = _Image->Attributes(); attr._dt = 0.;
    if (_WhiteMatterMask) {
      if (!_WhiteMatterMask->HasSpatialAttributesOf(_Image)) {
        Throw(ERR_RuntimeError, __FUNCTION__, "Attributes of white matter mask differ from those of the intensity image!");
      }
      ComputeGlobalStatistics global;
      ParallelForEachVoxel(attr, _Image, _WhiteMatterMask, global);
      _GlobalWhiteMatterMean     = global.Mean();
      _GlobalWhiteMatterVariance = global.Variance();
      if (_WhiteMatterWindowWidth > 0) {
        _LocalWhiteMatterMean.Initialize(attr);
        _LocalWhiteMatterVariance.Initialize(attr);
        ComputeLocalStatistics local(attr, _WhiteMatterWindowWidth, _GlobalWhiteMatterMean, _GlobalWhiteMatterVariance);
        ParallelForEachVoxel(attr, _Image, _WhiteMatterMask, &_LocalWhiteMatterMean, &_LocalWhiteMatterVariance, local);
      }
      if (IsNaN(_MinGradient)) {
        ComputeMeanAbsoluteDifference mad(_GlobalWhiteMatterMean);
        ParallelForEachVoxel(attr, _Image, _WhiteMatterMask, mad);
        _MinGradient = .25 * mad.Value();
      }
    }
    if (_GreyMatterMask) {
      if (!_GreyMatterMask->HasSpatialAttributesOf(_Image)) {
        Throw(ERR_RuntimeError, __FUNCTION__, "Attributes of grey matter mask differ from those of the intensity image!");
      }
      ComputeGlobalStatistics global;
      ParallelForEachVoxel(attr, _Image, _GreyMatterMask, global);
      _GlobalGreyMatterMean     = global.Mean();
      _GlobalGreyMatterVariance = global.Variance();
      if (_GreyMatterWindowWidth > 0) {
        _LocalGreyMatterMean.Initialize(attr);
        _LocalGreyMatterVariance.Initialize(attr);
        ComputeLocalStatistics local(attr, _GreyMatterWindowWidth, _GlobalGreyMatterMean, _GlobalGreyMatterVariance);
        ParallelForEachVoxel(attr, _Image, _GreyMatterMask, &_LocalGreyMatterMean, &_LocalGreyMatterVariance, local);
      }
    }
    if (IsInf(_MinIntensity)) {
      _MinIntensity = _GlobalGreyMatterMean  - 5. * sqrt(_GlobalGreyMatterVariance);
    }
    if (IsInf(_MaxIntensity)) {
      _MaxIntensity = _GlobalWhiteMatterMean + 5. * sqrt(_GlobalWhiteMatterVariance);
    }
    _CorticalDeepGreyMatterBoundingBox = ComputeCorticalDeepGreyMatterBoundingBox(attr, _VentriclesDistance);
    #if BUILD_WITH_DEBUG_CODE
      cout << "\n\tintensity range = [" << _MinIntensity << ", " << _MaxIntensity << "]";
      if (_WhiteMatterMask) {
        cout << "\n\tWM mean = " << _GlobalWhiteMatterMean;
        cout << ", WM sigma = " << sqrt(_GlobalWhiteMatterVariance);
      }
      if (_GreyMatterMask) {
        cout << "\n\tGM mean = " << _GlobalGreyMatterMean;
        cout << ", GM sigma = " << sqrt(_GlobalGreyMatterVariance);
      }
      if (_WhiteMatterMask && _GreyMatterMask) {
        const double thres = IntersectionOfNormalDistributions(
                                _GlobalWhiteMatterMean, _GlobalWhiteMatterVariance,
                                _GlobalGreyMatterMean,  _GlobalGreyMatterVariance);
        cout << "\n\tWM/GM threshold = " << thres;
      }
      cout << ", min. gradient = " << _MinGradient;
      cout << "\n\tWM->dGM->cGM bounding box = [";
      for (int i = 0; i < 6; ++i) {
        if (i > 0) {
          if (i % 2 == 0) cout << "] x [";
          else            cout << ", ";
        }
        cout << _CorticalDeepGreyMatterBoundingBox[i];
      }
      cout << "]";
    #endif
  }
  if (IsNaN(_MinGradient)) {
    _MinGradient = 0.;
  }
  if (IsNaN(_MaxGradient)) {
    _MaxGradient = 4. * _MinGradient;
  }
  #if BUILD_WITH_DEBUG_CODE
    cout << "\n" << endl;
  #endif
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
  const int radius   = ifloor(_MaxDistance / _StepLength);
  const int nsamples = 2 * radius + 1;

  const double wm_threshold = IntersectionOfNormalDistributions(
                                  _GlobalWhiteMatterMean, _GlobalWhiteMatterVariance,
                                  _GlobalGreyMatterMean,  _GlobalGreyMatterVariance);

  ContinuousImage image;
  image.Input(_Image);
  image.DefaultValue(NaN);
  image.Initialize();

  MIRTK_START_TIMING();
  ComputeDistances eval;
  eval._Points          = Points();
  eval._Status          = InitialStatus();
  eval._Normals         = Normals();
  eval._Image           = &image;
  eval._Distances       = distances;
  eval._Magnitude       = magnitude;
  eval._Padding         = _Padding;
  eval._MinIntensity    = _MinIntensity;
  eval._MaxIntensity    = _MaxIntensity;
  eval._MinGradient     = _MinGradient;
  eval._MaxGradient     = _MaxGradient;
  eval._MaxDistance     = _MaxDistance;
  eval._StepLength      = _StepLength;
  eval._NumberOfSamples = nsamples;
  eval._EdgeType        = _EdgeType;

  eval._CorticalHullDistance = _CorticalHullDistance;
  eval._VentriclesDistance   = _VentriclesDistance;
  eval._CerebellumDistance   = _CerebellumDistance;
  eval._CorticalDeepGreyMatterBoundingBox = _CorticalDeepGreyMatterBoundingBox.data();

  BinaryImage surface_mask;
  if (_EdgeType == NeonatalWhiteSurface) {
    surface_mask.Initialize(_Image->Attributes(), 1);
    vtkSmartPointer<vtkPointSet> pointset = WorldToImage(surface, &surface_mask);
    vtkSmartPointer<vtkPolyData> polydata = vtkPolyData::SafeDownCast(pointset);
    vtkSmartPointer<vtkImageData>        vtkmask = NewVtkMask(_Image->X(), _Image->Y(), _Image->Z());
    vtkSmartPointer<vtkImageStencilData> stencil = ImageStencil(vtkmask, polydata);
    ImageStencilToMask(stencil, vtkmask);
    surface_mask.CopyFrom(reinterpret_cast<BinaryPixel *>(vtkmask->GetScalarPointer()));
    eval._SurfaceMask = &surface_mask;
  } else {
    eval._SurfaceMask = nullptr;
  }

  if (_EdgeType == NeonatalWhiteSurface) {
    eval._MaxBackgroundSnapDepth = max(1, iceil(1. / _StepLength));
  } else {
    eval._MaxBackgroundSnapDepth = 0.;
  }

  eval._GlobalWhiteMatterMean      = _GlobalWhiteMatterMean;
  eval._GlobalWhiteMatterSigma     = sqrt(_GlobalWhiteMatterVariance);
  eval._GlobalWhiteMatterVariance  = _GlobalWhiteMatterVariance;
  eval._GlobalGreyMatterMean       = _GlobalGreyMatterMean;
  eval._GlobalGreyMatterSigma      = sqrt(_GlobalGreyMatterVariance);
  eval._GlobalGreyMatterVariance   = _GlobalGreyMatterVariance;
  eval._GlobalWhiteMatterThreshold = wm_threshold;
  eval._LocalWhiteMatterMean       = (_LocalWhiteMatterMean    .IsEmpty() ? nullptr : &_LocalWhiteMatterMean);
  eval._LocalWhiteMatterVariance   = (_LocalWhiteMatterVariance.IsEmpty() ? nullptr : &_LocalWhiteMatterVariance);
  eval._LocalGreyMatterMean        = (_LocalGreyMatterMean     .IsEmpty() ? nullptr : &_LocalGreyMatterMean);
  eval._LocalGreyMatterVariance    = (_LocalGreyMatterVariance .IsEmpty() ? nullptr : &_LocalGreyMatterVariance);

  #if BUILD_WITH_DEBUG_CODE
    static int ncalls = 0;
    #if 0
      ++ncalls;
    #endif
    Matrix f, g;
    if (ncalls == 1) {
      f.Initialize(_NumberOfPoints, nsamples);
      g.Initialize(_NumberOfPoints, nsamples);
      eval._IntensitySamples = &f;
      eval._GradientSamples  = &g;
    } else {
      eval._IntensitySamples = nullptr;
      eval._GradientSamples  = nullptr;
    }
    if (dbg_dist >= 0.) {
      eval(blocked_range<int>(0, _NumberOfPoints));
    } else {
      parallel_for(blocked_range<int>(0, _NumberOfPoints), eval);
    }
    if (f.Rows() > 0) {
      #if MIRTK_Numerics_WITH_MATLAB
        f.WriteMAT("debug_image_edge_distance_f.mat", "f");
      #else
        f.Write("debug_image_edge_distance_f.bin");
      #endif
    }
    if (g.Rows() > 0) {
      #if MIRTK_Numerics_WITH_MATLAB
        g.WriteMAT("debug_image_edge_distance_g.mat", "g");
      #else
        g.Write("debug_image_edge_distance_g.bin");
      #endif
    }
  #else
    parallel_for(blocked_range<int>(0, _NumberOfPoints), eval);
  #endif
  MIRTK_DEBUG_TIMING(5, "computing edge distances");

  // Smooth measurements
  if (_MedianFilterRadius > 0) {
    MIRTK_RESET_TIMING();
    MedianMeshFilter median;
    median.Input(surface);
    median.EdgeTable(SharedEdgeTable());
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
    smoother.EdgeTable(SharedEdgeTable());
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
    smoother.EdgeTable(SharedEdgeTable());
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
