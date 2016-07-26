/*
 * Medical Image Registration ToolKit (MIRTK)
 *
 * Copyright 2013-2016 Imperial College London
 * Copyright 2013-2016 Andreas Schuh
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

#include "mirtk/Common.h"
#include "mirtk/Options.h"

#include "mirtk/IOConfig.h"
#include "mirtk/NumericsConfig.h"
#include "mirtk/DeformableConfig.h"
#include "mirtk/TransformationConfig.h"

#include "mirtk/PointSetIO.h"
#include "mirtk/PointSetUtils.h"
#include "mirtk/Transformation.h"
#include "mirtk/BSplineFreeFormTransformation3D.h"
#include "mirtk/BSplineFreeFormTransformationSV.h"

// Deformable surface model / parameterization
#include "mirtk/DeformableSurfaceModel.h"
#include "mirtk/DeformableSurfaceLogger.h"
#include "mirtk/DeformableSurfaceDebugger.h"

// Optimization method
#include "mirtk/LocalOptimizer.h"
#include "mirtk/EulerMethod.h"
#include "mirtk/EulerMethodWithMomentum.h"
#include "mirtk/GradientDescent.h"
#include "mirtk/InexactLineSearch.h"
#include "mirtk/BrentLineSearch.h"

// Stopping criteria
#include "mirtk/MinActiveStoppingCriterion.h"
#include "mirtk/InflationStoppingCriterion.h"

// External forces
#include "mirtk/BalloonForce.h"
#include "mirtk/ImageEdgeForce.h"
#include "mirtk/ImplicitSurfaceDistance.h"
#include "mirtk/ImplicitSurfaceSpringForce.h"

// Internal forces
#include "mirtk/SpringForce.h"
#include "mirtk/InflationForce.h"
#include "mirtk/CurvatureConstraint.h"
#include "mirtk/GaussCurvatureConstraint.h"
#include "mirtk/MeanCurvatureConstraint.h"
#include "mirtk/QuadraticCurvatureConstraint.h"
#include "mirtk/MetricDistortion.h"
#include "mirtk/StretchingForce.h"
#include "mirtk/RepulsiveForce.h"
#include "mirtk/NonSelfIntersectionConstraint.h"

// Transformation constraints
#include "mirtk/SmoothnessConstraint.h"

// VTK
#include "vtkPointData.h"
#include "vtkCellData.h"
#include "vtkDataArray.h"
#include "vtkFloatArray.h"
#include "vtkGenericCell.h"
#include "vtkCellTreeLocator.h"
#include "vtkSortDataArray.h"
#include "vtkPolyDataNormals.h"


using namespace mirtk;


// =============================================================================
// Help
// =============================================================================

// -----------------------------------------------------------------------------
void PrintHelp(const char *name)
{
  DeformableSurfaceModel model; // with default parameters
  cout << endl;
  cout << "Usage: " << name << " <input> <output> [options]" << endl;
  cout << endl;
  cout << "Description:" << endl;
  cout << "  Iteratively minimizes a deformable surface model energy functional. The gradient of" << endl;
  cout << "  the energy terms are the internal and external forces of the deformable surface model." << endl;
  cout << endl;
  cout << "Input options:" << endl;
  cout << "  -initial <file>" << endl;
  cout << "      Point set used to initialize the deformed output mesh. Usually the output of a" << endl;
  cout << "      previous optimization with possibly saved node status (see :option:`-save-status`)." << endl;
  cout << "      (default: input)" << endl;
  cout << "  -dof <type> [<dx> [<dy> <dz>]]" << endl;
  cout << "      Optimize spatial transformation of named <type> to deform the mesh points." << endl;
  cout << "      The optional <dx>, <dy>, and <dz> arguments specify the control point spacing" << endl;
  cout << "      of free-form deformation (FFD) transformations. Common transformation types are:" << endl;
  cout << "      - ``FFD``:   Cubic B-spline FFD." << endl;
  cout << "      - ``SVFFD``: Stationary velocity (SV) cubic B-spline FFD." << endl;
  cout << "  -image <file>" << endl;
  cout << "      Intensity image on which external forces are based. (default: none)" << endl;
  cout << "  -distance-image, -dmap <file>" << endl;
  cout << "      Euclidean distance image on which implicit surface forces are based. (default: none)" << endl;
  cout << "  -distance-offset, -dmap-offset <value>" << endl;
  cout << "      Implicit surface isovalue of :option:`-distance-image`. (default: 0)" << endl;
  cout << "  -mask <file>" << endl;
  cout << "      Mask defining region in which external forces are non-zero. (default: none)" << endl;
  cout << "  -padding <value>" << endl;
  cout << "      Padding/Background value of input :option:`-image`. (default: none)" << endl;
  cout << "  -inflate-brain" << endl;
  cout << "      Set default parameters of cortical surface inflation process equivalent" << endl;
  cout << "      to FreeSurfer's mris_inflate command." << endl;
  cout << endl;
  cout << "Optimization options:" << endl;
  cout << "  -optimizer <name>" << endl;
  cout << "      Optimization method used to minimize the energy of the deformable surface model:" << endl;
  cout << "      - ``EulerMethod``:              Forward Euler integration (default)" << endl;
  cout << "      - ``EulerMethodWithDamping``:   Forward Euler integration with momentum." << endl;
  cout << "      - ``EulerMethodWithMomentum``:  Forward Euler integration with momentum." << endl;
  cout << "      - ``GradientDescent``:          Gradient descent optimizer." << endl;
  cout << "      - ``ConjugateGradientDescent``: Conjugate gradient descent." << endl;
  cout << "  -line-search, -linesearch <name>" << endl;
  cout << "      Line search method used by gradient descent optimizers:" << endl;
  cout << "      - ``Adaptive``: Line search with adaptive step length. (default)" << endl;
  cout << "      - ``Brent``: Brent's line search method." << endl;
  cout << "  -damping <value>" << endl;
  cout << "      Damping ratio used by Euler method with momentum modelling the effect" << endl;
  cout << "      of dissipation of kinetic energy." << endl;
  cout << "  -momentum <value>" << endl;
  cout << "      Momentum of Euler method with momentum, i.e., :math:`1 - damping` (see :option:`-damping`)" << endl;
  cout << "  -mass <value>" << endl;
  cout << "      Node mass used by Euler methods with momentum. (default: 1)" << endl;
  cout << "  -levels <max> | <min> <max>" << endl;
  cout << "      Perform optimization on starting at level <max> until level <min> (> 0)." << endl;
  cout << "      When only the <max> level argument is given, the <min> level is set to 1." << endl;
  cout << "      On each level, the node forces are averaged :math:`2^{level-1}` times which" << endl;
  cout << "      is similar to computing the forces on a coarser mesh. See :option:`-force-averaging`. (default: 0 0)" << endl;
  cout << "  -force-averaging <n>..." << endl;
  cout << "      Number of force averaging steps. (default: 0)" << endl;
  cout << "  -distance-averaging <n>..." << endl;
  cout << "      Number of :option:`-distance` force averaging steps. (default: 0)" << endl;
  cout << "  -steps, -max-steps, -iterations, -max-iterations <n>..." << endl;
  cout << "      Maximum number of iterations. (default: 100)" << endl;
  cout << "  -step, -dt <value>..." << endl;
  cout << "      Length of integration/gradient steps. (default: 1)" << endl;
  cout << "  -max-dx, -maxdx, -dx <value>..." << endl;
  cout << "      Maximum displacement of a node at each iteration. By default, the node displacements" << endl;
  cout << "      are normalized by the maximum node displacement. When this option is used, the node" << endl;
  cout << "      displacements are clamped to the specified maximum length instead. (default: :option:`-step`)" << endl;
  cout << "  -remesh <n>" << endl;
  cout << "      Remesh surface mesh every n-th iteration. (default: " << model.RemeshInterval() << ")" << endl;
  cout << "  -remesh-adaptively" << endl;
  cout << "      Remesh surface mesh using an adaptive edge length interval based on local curvature" << endl;
  cout << "      of the deformed surface mesh or input implicit surface (:option:`-distance-image`)." << endl;
  cout << "  -[no]triangle-inversion" << endl;
  cout << "      Whether to allow inversion of pair of triangles during surface remeshing. (default: on)" << endl;
  cout << "  -min-edgelength <value>..." << endl;
  cout << "      Minimum edge length used for local adaptive remeshing. (default: " << model.MinEdgeLength() << ")" << endl;
  cout << "  -max-edgelength <value>..." << endl;
  cout << "      Maximum edge length used for local adaptive remeshing. (default: " << model.MaxEdgeLength() << ")" << endl;
  cout << "  -min-angle <degrees>..." << endl;
  cout << "      Minimum angle between edge node normals for an edge be excluded from collapsing during" << endl;
  cout << "      iterative :option:`-remesh` operations. (default: " << model.MinFeatureAngle() << ")" << endl;
  cout << "  -max-angle <degrees>..." << endl;
  cout << "      Maximum angle between edge node normals for an edge be excluded from splitting during" << endl;
  cout << "      iterative :option:`-remesh` operations. (default: " << model.MaxFeatureAngle() << ")" << endl;
  cout << "  -lowpass <n>" << endl;
  cout << "      Low-pass filter surface mesh every n-th iteration. (default: " << model.LowPassInterval() << ")" << endl;
  cout << "  -lowpass-iterations <n>" << endl;
  cout << "      Number of :option:`-lowpass` filter iterations. (default: " << model.LowPassIterations() << ")" << endl;
  cout << "  -lowpass-band <band>" << endl;
  cout << "      Low-pass filtering band argument, usually in the range [0, 2]. (default: " << model.LowPassBand() << ")" << endl;
  cout << "  -nointersection" << endl;
  cout << "      Hard non-self-intersection constraint for surface meshes. (default: off)" << endl;
  cout << "  -mind, -min-distance <value>" << endl;
  cout << "      Minimum distance to other triangles in front of a given triangle." << endl;
  cout << "  -minw, -min-width <value>" << endl;
  cout << "      Minimum distance to other triangles in the back of a given triangle." << endl;
  cout << "  -max-collision-angle <degrees>" << endl;
  cout << "      Maximum angle between vector connecting centers of nearby triangles and the face normal" << endl;
  cout << "      of the reference triangle for a collision to be detected. When the triangles are within" << endl;
  cout << "      the same flat neighborhood of the surface mesh, this angle will be close to 90 degrees." << endl;
  cout << "      This parameter reduces false collision detection between neighboring triangles. (default: " << model.MaxCollisionAngle() << ")" << endl;
  cout << "  -fast-collision-test\n";
  cout << "      Use fast approximate triangle-triangle collision test based on distance of their centers only. (default: off)" << endl;
  cout << "  -reset-status" << endl;
  cout << "      Set status of all mesh nodes to active again after each level (see :option:`-levels`). (default: off)" << endl;
  cout << endl;
  cout << "Deformable model options:" << endl;
  cout << "  -neighborhood <n>" << endl;
  cout << "      Size of node neighborhoods used by internal force terms that consider more" << endl;
  cout << "      than only the adjacent nodes, but also up to n-connected nodes. (default: " << model.NeighborhoodRadius() << ")" << endl;
  cout << "  -distance <w>" << endl;
  cout << "      Weight of implicit surface distance. (default: 0)" << endl;
  cout << "  -distance-spring, -dspring <w>" << endl;
  cout << "      Weight of implicit surface spring force. (default: 0)" << endl;
  cout << "  -distance-measure <name>" << endl;
  cout << "      Implicit surface distance measure used by :option:`-distance` and :option:`-distance-spring`):" << endl;
  cout << "      - ``minimum``: Minimum surface distance (see :option:`-distance-image`, default)" << endl;
  cout << "      - ``normal``:  Estimate distance by casting rays along normal direction." << endl;
  cout << "  -balloon-inflation, -balloon <w>" << endl;
  cout << "      Weight of inflation force based on local intensity statistics. (default: 0)" << endl;
  cout << "  -balloon-deflation <w>" << endl;
  cout << "      Weight of deflation force based on local intensity statistics. (default: 0)" << endl;
  cout << "  -balloon-min <intensity>" << endl;
  cout << "      Global lower intensity threshold for :option:`-balloon-inflation` or :option:`-balloon-deflation`. (default: -inf)" << endl;
  cout << "  -balloon-max <intensity>" << endl;
  cout << "      Global lower intensity threshold for :option:`-balloon-inflation` or :option:`-balloon-deflation`. (default: +inf)" << endl;
  cout << "  -balloon-range <min> <max>" << endl;
  cout << "      Global intensity thresholds for :option:`-balloon-inflation` or :option:`-balloon-deflation`. (default: [-inf +inf])" << endl;
  cout << "  -balloon-radius <r>" << endl;
  cout << "      Radius for local intensity statistics of :option:`-balloon-inflation` or :option:`-balloon-deflation`. (default: 7 times voxel size)" << endl;
  cout << "  -balloon-sigma <sigma>" << endl;
  cout << "      Local intensity standard deviation scaling factor of :option:`-balloon-inflation` or :option:`-balloon-deflation`. (default: 5)" << endl;
  cout << "  -balloon-mask <file>" << endl;
  cout << "      Image mask used for local intensity statistics for :option:`-balloon-inflation` or :option:`-balloon-deflation`." << endl;
  cout << "      (default: interior of deformed surface)" << endl;
  cout << "  -edges <w>" << endl;
  cout << "      Weight of image edge force. (default: 0)" << endl;
  cout << "  -inflation <w>" << endl;
  cout << "      Weight of surface inflation force used for cortical surface inflation. (default: 0)" << endl;
  cout << "  -bending-energy <w>" << endl;
  cout << "      Weight of bending energy of :option:`-dof` transformation. (default: 0)" << endl;
  cout << "  -spring <w>" << endl;
  cout << "      Weight of internal spring force. (default: 0)" << endl;
  cout << "  -normal-spring, -nspring <w>" << endl;
  cout << "      Weight of internal spring force in normal direction. (default: 0)" << endl;
  cout << "  -tangential-spring, -tspring <w>" << endl;
  cout << "      Weight of internal spring force in tangent plane. (default: 0)" << endl;
  cout << "  -normalized-spring <w>" << endl;
  cout << "      Weight of internal spring force normalized w.r.t. force in normal direction. (default: 0)" << endl;
  cout << "  -curvature <w>" << endl;
  cout << "      Weight of surface curvature. (default: 0)" << endl;
  cout << "  -quadratic-curvature, -qcurvature <w>" << endl;
  cout << "      Weight of surface curvature estimated by quadratic fit of node neighbor" << endl;
  cout << "      to tangent plane distance. (default: 0)" << endl;
  cout << "  -gauss-curvature, -gcurvature <w>" << endl;
  cout << "      Weight of Gauss curvature constraint. (default: 0)" << endl;
  cout << "  -distortion <w>" << endl;
  cout << "      Weight of metric distortion." << endl;
  cout << "  -stretching <w>" << endl;
  cout << "      Weight of spring force based on difference of neighbor distance compared to" << endl;
  cout << "      initial distance. (default: 0)" << endl;
  cout << "  -repulsion <w> [<radius>]" << endl;
  cout << "      Weight of node repulsion force. (default: 0 0)" << endl;
  cout << "  -collision <w>" << endl;
  cout << "      Weight of triangle repulsion force." << endl;
  cout << endl;
  cout << "Stopping criterion options:" << endl;
  cout << "  -extrinsic-energy" << endl;
  cout << "      Consider only sum of external energy terms as total energy value of deformable model functional." << endl;
  cout << "      Internal forces still contribute to the gradient of the functional, but are excluded from the" << endl;
  cout << "      energy function value (see :option:`-epsilon` and :option:`-min-energy`). (default: off)" << endl;
  cout << "  -epsilon <value>" << endl;
  cout << "      Minimum change of deformable surface energy convergence criterion." << endl;
  cout << "  -delta <value>" << endl;
  cout << "      Minimum maximum node displacement or :option:`-dof` parameter value." << endl;
  cout << "  -min-energy <value>" << endl;
  cout << "      Target deformable surface energy value. (default: 0)" << endl;
  cout << "  -min-active <ratio>" << endl;
  cout << "      Minimum ratio of active nodes in [0, 1]. (default: 0)" << endl;
  cout << "  -inflation-error <threshold>" << endl;
  cout << "      Threshold of surface inflation RMS measure. (default: off)" << endl;
  cout << endl;
  cout << "Output options:" << endl;
  cout << "  -track [<name>]" << endl;
  cout << "      Record sum of node displacements along normal direction. The integrated" << endl;
  cout << "      displacements are stored in the point data array named \"NormalDisplacement\"" << endl;
  cout << "      by default or with the specified <name>. (default: off)" << endl;
  cout << "  -track-zero-mean [<name>]" << endl;
  cout << "      Same as :option:`-track`, but subtract mean from tracked values." << endl;
  cout << "      This option is implicit when :option:`-inflate-brain` is given to inflate a cortical" << endl;
  cout << "      brain surface. The default output array name is \"SulcalDepth\". Otherwise, the default" << endl;
  cout << "      point data array name is \"NormalDisplacementZeroMean\". (default: off)" << endl;
  cout << "  -track-zero-median [<name>]" << endl;
  cout << "      Same as :option:`-track`, but subtract median from tracked values." << endl;
  cout << "      The default point data array name is \"NormalDisplacementZeroMedian\". (default: off)" << endl;
  cout << "  -track-unit-variance [<name>]" << endl;
  cout << "      Same as :option:`-track`, but divide tracked values by their standard deviation." << endl;
  cout << "      The default point data array name is \"NormalDisplacementUnitVariance\". (default: off)" << endl;
  cout << "  -track-zvalues [<name>]" << endl;
  cout << "      Same as :option:`-track`, but subtract mean from tracked values and divide by standard deviation." << endl;
  cout << "      The resulting values are the Z-score normalized standard scores of the tracked displacements." << endl;
  cout << "      The default point data array name is \"NormalDisplacementZValues\". (default: off)" << endl;
  cout << "  -track-zero-median-zvalues [<name>]" << endl;
  cout << "      Same as :option:`-track`, but subtract median from tracked values and divide by standard deviation." << endl;
  cout << "      It can be used with :option:`-inflate-brain` to obtain a normalized curvature measure." << endl;
  cout << "      The default point data array name is \"NormalDisplacementZeroMedianZValues\". (default: off)" << endl;
  cout << "  -track-without-momentum" << endl;
  cout << "      When tracking the total displacement of a node in normal direction using the EulerMethodWithMomentum" << endl;
  cout << "      :option:`-optimizer` as used in particular by :option:`-inflate-brain`, exclude the momentum from the." << endl;
  cout << "      tracked displacement. This is idential to FreeSurfer's mrisTrackTotalDisplacement used for the curvature" << endl;
  cout << "      output of mris_inflate. The correct curvature value is, however, obtained by including the momentum" << endl;
  cout << "      component as it integrates the actual amount by which each node is displaced during the Euler steps." << endl;
  cout << "      Not using this option corresponds to the mrisTrackTotalDisplacementNew function. (default: off)" << endl;
  cout << "  -notrack" << endl;
  cout << "      Do not track node displacements along normal direction." << endl;
  cout << "  -center-output" << endl;
  cout << "      Center output mesh such that center is at origin. (default: off)" << endl;
  cout << "  -match-area" << endl;
  cout << "      Scale output mesh by ratio of input and output surface area. (default: off)" << endl;
  cout << "  -match-sampling" << endl;
  cout << "      Resample output mesh at corresponding positions of input mesh." << endl;
  cout << "      This option is only useful in conjunction with :option:`-remesh`. (default: off)" << endl;
  cout << "  -save-status" << endl;
  cout << "      Save node status (active/passive) to output file. (default: off)" << endl;
  cout << "  -ascii | -nobinary" << endl;
  cout << "      Write legacy VTK in ASCII format. (default: off)" << endl;
  cout << "  -binary | -noascii" << endl;
  cout << "      Write legacy VTK in binary format. (default: on)" << endl;
  cout << "  -[no]compress" << endl;
  cout << "      Write XML VTK file with or without compression. (default: on)" << endl;
  cout << "  -debug-prefix <prefix>" << endl;
  cout << "      File name prefix for :option:`-debug` output. (default: deform_mesh\\_)" << endl;
  cout << "  -debug-interval <n>" << endl;
  cout << "      Write :option:`-debug` output every n-th iteration. (default: 10)" << endl;
  cout << "  -[no]level-prefix" << endl;
  cout << "      Write :option:`-debug` output without level prefix in file names. (default: on)" << endl;
  cout << endl;
  cout << "Advanced options:" << endl;
  cout << "  -par <name> <value>" << endl;
  cout << "      Advanced option to set surface model or optimizer parameter." << endl;
  PrintCommonOptions(cout);
  cout << endl;
}

// =============================================================================
// Auxiliaries
// =============================================================================

// -----------------------------------------------------------------------------
/// Get parameter for current level
///
/// Some parameters can be set per level. The number of levels is defined by
/// the maximum number of values for each parameter. When the user did not
/// specify a parameter for each level, the first value is used for the initial
/// levels for which a parameter value is missing.
template <class T>
T ParameterValue(int level, int nlevels, const Array<T> &values, T default_value)
{
  mirtkAssert(nlevels > 0,                   "at least one level");
  mirtkAssert(0 <= level && level < nlevels, "valid level index");
  if (values.empty()) return default_value;
  const int nvalues = static_cast<int>(values.size());
  while (level >= nvalues) --level;
  return values[level];
}

// -----------------------------------------------------------------------------
/// Initialize array of total energy gradient averaging steps
Array<int> GradientAveraging(int min_level, int max_level)
{
  Array<int> navgs;
  navgs.reserve(max_level - min_level + 1);
  for (int level = max_level; level >= min_level; --level) {
    navgs.push_back(level > 1 ? static_cast<int>(pow(2, level - 2)) : 0);
  }
  return navgs;
}

// -----------------------------------------------------------------------------
/// Subract mean of tracked normal displacements
void DemeanValues(vtkDataArray *values, bool use_median = false)
{
  double mu;

  // Compute mean (or median)
  if (use_median) {
    vtkSmartPointer<vtkDataArray> sorted;
    sorted.TakeReference(values->NewInstance());
    sorted->DeepCopy(values);
    vtkSortDataArray::Sort(sorted);
    mu = sorted->GetComponent(sorted->GetNumberOfTuples() / 2, 0);
  } else {
    mu = .0;
    for (vtkIdType id = 0; id < values->GetNumberOfTuples(); ++id) {
      mu += values->GetComponent(id, 0);
    }
    mu /= values->GetNumberOfTuples();
  }

  // Subtract mean from curvature measures
  for (vtkIdType id = 0; id < values->GetNumberOfTuples(); ++id) {
    values->SetComponent(id, 0, values->GetComponent(id, 0) - mu);
  }
}

// -----------------------------------------------------------------------------
/// Normalize variance of tracked normal displacements
void NormalizeVariance(vtkDataArray *values)
{
  // Compute mean
  double mu = .0;
  for (vtkIdType id = 0; id < values->GetNumberOfTuples(); ++id) {
    mu += values->GetComponent(id, 0);
  }
  mu /= values->GetNumberOfTuples();

  // Compute variance
  double var = .0;
  for (vtkIdType id = 0; id < values->GetNumberOfTuples(); ++id) {
    var += pow(values->GetComponent(id, 0) - mu, 2);
  }
  var /= values->GetNumberOfTuples();

  // Normalize variance
  double sigma = (var == .0 ? 1.0 : sqrt(var));
  for (vtkIdType id = 0; id < values->GetNumberOfTuples(); ++id) {
    values->SetComponent(id, 0, mu + (values->GetComponent(id, 0) - mu) / sigma);
  }
}

// -----------------------------------------------------------------------------
/// Normalize tracked normal displacements
void NormalizeValues(vtkDataArray *values, bool use_median = false)
{
  double mu, var;

  // Compute mean (or median)
  if (use_median) {
    vtkSmartPointer<vtkDataArray> sorted;
    sorted.TakeReference(values->NewInstance());
    sorted->DeepCopy(values);
    vtkSortDataArray::Sort(sorted);
    mu = sorted->GetComponent(sorted->GetNumberOfTuples() / 2, 0);
  } else {
    mu = .0;
    for (vtkIdType id = 0; id < values->GetNumberOfTuples(); ++id) {
      mu += values->GetComponent(id, 0);
    }
    mu /= values->GetNumberOfTuples();
  }

  // Compute variance
  var = .0;
  for (vtkIdType id = 0; id < values->GetNumberOfTuples(); ++id) {
    var += pow(values->GetComponent(id, 0) - mu, 2);
  }
  var /= values->GetNumberOfTuples();

  // Z-score normalize curvature measures
  double sigma = (var == .0 ? 1.0 : sqrt(var));
  for (vtkIdType id = 0; id < values->GetNumberOfTuples(); ++id) {
    values->SetComponent(id, 0, (values->GetComponent(id, 0) - mu) / sigma);
  }
}

// -----------------------------------------------------------------------------
/// Resample surface mesh at corresponding positions of the initial points
///
/// \todo Resample also output point data.
vtkSmartPointer<vtkPointSet>
ResampleAtInitialPoints(vtkSmartPointer<vtkPointSet> input, vtkSmartPointer<vtkPointSet> output)
{
  double    p[3], x[3], pcoords[3], dist2, *weights;
  vtkIdType cellId;
  int       subId;

  vtkDataArray *initial_position = output->GetPointData()->GetArray("InitialPoints");
  if (!initial_position) {
    Warning("Cannot resample surface mesh at points corresponding to points of input mesh:"
            " deformed mesh has not point data array named \"InitialPoints\".");
    return output;
  }

  vtkSmartPointer<vtkPoints> initial_points = vtkSmartPointer<vtkPoints>::New();
  initial_points->SetNumberOfPoints(output->GetNumberOfPoints());
  for (vtkIdType ptId = 0; ptId < output->GetNumberOfPoints(); ++ptId) {
    initial_points->SetPoint(ptId, initial_position->GetTuple(ptId));
  }

  vtkSmartPointer<vtkPointSet> initial;
  initial.TakeReference(output->NewInstance());
  initial->ShallowCopy(output);
  initial->SetPoints(initial_points);

  vtkSmartPointer<vtkAbstractCellLocator> locator;
  locator = vtkSmartPointer<vtkCellTreeLocator>::New();
  locator->SetDataSet(initial);
  locator->BuildLocator();

  vtkSmartPointer<vtkGenericCell> cell = vtkSmartPointer<vtkGenericCell>::New();
  weights = new double[output->GetMaxCellSize()];

  vtkSmartPointer<vtkPoints> resampled_points = vtkSmartPointer<vtkPoints>::New();
  resampled_points->SetNumberOfPoints(input->GetNumberOfPoints());

  for (vtkIdType ptId = 0; ptId < input->GetNumberOfPoints(); ++ptId) {
    input->GetPoint(ptId, p);
    locator->FindClosestPoint(p, x, cell, cellId, subId, dist2);
    cell->EvaluatePosition(x, NULL, subId, pcoords, dist2, weights);
    p[0] = p[1] = p[2];
    for (vtkIdType i = 0; i < cell->GetNumberOfPoints(); ++i) {
      output->GetPoint(cell->GetPointId(i), x);
      p[0] += weights[i] * x[0];
      p[1] += weights[i] * x[1];
      p[2] += weights[i] * x[2];
    }
    resampled_points->SetPoint(ptId, p);
  }

  delete[] weights;

  vtkSmartPointer<vtkPointSet> resampled;
  resampled.TakeReference(input->NewInstance());
  resampled->ShallowCopy(input);
  resampled->SetPoints(resampled_points);
  return resampled;
}

// =============================================================================
// Main
// =============================================================================

// -----------------------------------------------------------------------------
/// Helper macro to parse multiple parameter arguments
#define PARSE_ARGUMENTS(T, name) \
  do { \
    T value; \
    PARSE_ARGUMENT(value); \
    (name).push_back(value); \
  } while (HAS_ARGUMENT); \
  nlevels = max(nlevels, static_cast<int>((name).size()))

// -----------------------------------------------------------------------------
int main(int argc, char *argv[])
{
  FileOption output_fopt = FO_Default;

  verbose = 1; // default verbosity level
  EXPECTS_POSARGS(2);

  // Initialize libraries / object factories
  InitializeIOLibrary();
  InitializeNumericsLibrary();
  InitializeDeformableLibrary();
  InitializeTransformationLibrary();

  // Deformable surface model and default optimizer
  UniquePtr<Transformation> dof;
  DeformableSurfaceModel    model;
  DeformableSurfaceLogger   logger;
  DeformableSurfaceDebugger debugger(&model);
  UniquePtr<LocalOptimizer> optimizer(new EulerMethod(&model));
  ParameterList             params;

  // Read input point set
  vtkSmartPointer<vtkPointSet> input = ReadPointSet(POSARG(1), output_fopt);
  vtkPointData * const inputPD = input->GetPointData();
  vtkCellData  * const inputCD = input->GetCellData();
  ImageAttributes domain = PointSetDomain(input);
  model.Input(input);

  // External forces (inactive if weight == 0)
  BalloonForce               balloon ("Balloon force", .0);
  ImageEdgeForce             edges   ("Edge force",    .0);
  ImplicitSurfaceDistance    distance("Distance",      .0);
  ImplicitSurfaceSpringForce dspring ("Dist. spring",  .0);

  // Internal forces (inactive if weight == 0)
  SpringForce                   spring    ("Bending",        .0);
  InflationForce                normspring("Bending",        .0);
  InflationForce                inflation ("Inflation",      .0);
  CurvatureConstraint           curvature ("Curvature",      .0);
  GaussCurvatureConstraint      gcurvature("Gauss curv.",    .0);
  MeanCurvatureConstraint       mcurvature("Mean curv.",     .0);
  QuadraticCurvatureConstraint  qcurvature("Quad. curv.",    .0);
  MetricDistortion              distortion("Distortion",     .0);
  StretchingForce               stretching("Stretching",     .0);
  RepulsiveForce                repulsion ("Repulsion",      .0);
  NonSelfIntersectionConstraint collision ("Collision",      .0);
  SmoothnessConstraint          dofbending("Bending energy", .0);

  spring.NormalWeight(.0);
  spring.TangentialWeight(.0);

  repulsion.Radius(.0);

  // Stopping criteria (disabled by default)
  MinActiveStoppingCriterion min_active(&model);
  InflationStoppingCriterion inflation_error(&model);

  min_active.Threshold(.0);
  inflation_error.Threshold(NaN);

  // Optional arguments
  const char *image_name        = nullptr;
  const char *balloon_mask_name = nullptr;
  const char *dmap_name         = nullptr;
  const char *dmag_name         = nullptr;
  double      dmap_offset       = .0;
  const char *mask_name         = nullptr;
  const char *track_name        = nullptr; // track normal movement of nodes
  bool        track_zero_mean   = false;   // subtract mean from tracked normal movements
  bool        track_unit_var    = false;   // Z-score normalize tracked normal movements
  bool        track_use_median  = false;   // use median instead of mean for normalization
  const char *initial_name      = NULL;
  const char *debug_prefix      = "deform_mesh_";
  double      padding           = NaN;
  bool        level_prefix      = true;
  bool        reset_status      = false;
  bool        center_output     = false;
  bool        match_area        = false;
  bool        match_sampling    = false;
  bool        signed_gradient   = false; // average gradient vectors with positive dot product
  bool        save_status       = false;
  bool        inflate_brain     = false; // mimick mris_inflate
  int         nlevels           = 1;     // no. of levels

  Array<int>    navgs;           // no. of total gradient averaging steps
  Array<int>    distance_navgs;  // no. of distance gradient averaging steps
  Array<int>    balloon_navgs;   // no. of balloon force gradient averaging steps
  Array<int>    nsteps;          // maximum no. of integration steps
  Array<double> max_dt;          // maximum integration step length
  Array<double> max_dx;          // maximum node displacement at each integration step
  Array<double> min_edge_length; // minimum average edge length
  Array<double> max_edge_length; // maximum average edge length
  Array<double> min_edge_angle;  // minimum angle between edge end points to allow melting
  Array<double> max_edge_angle;  // maximum angle between edge end points before enforcing subdivision

  bool   barg;
  int    iarg;
  double farg;

  for (ALL_OPTIONS) {
    // Input
    if (OPTION("-image")) {
      image_name = ARGUMENT;
    }
    else if (OPTION("-dmap") || OPTION("-distance-map") || OPTION("-distance-image") || OPTION("-implicit-surface")) {
      dmap_name = ARGUMENT;
    }
    else if (OPTION("-dmap-offset") || OPTION("-distance-offset") || OPTION("-implicit-surface-offset")) {
      PARSE_ARGUMENT(dmap_offset);
    }
    else if (OPTION("-dmag") || OPTION("-distance-magnitude")) {
      dmag_name = ARGUMENT;
    }
    else if (OPTION("-mask")) {
      mask_name = ARGUMENT;
    }
    else if (OPTION("-initial")) {
      initial_name = ARGUMENT;
    }
    else if (OPTION("-padding")) {
      PARSE_ARGUMENT(padding);
    }
    // Presets
    else if (OPTION("-inflate-brain")) { // cf. FreeSurfer's mris_inflate
      inflate_brain = true;
      nlevels = 6;
      navgs = GradientAveraging(1, 6);
      nsteps.resize(1);
      nsteps[0] = 10;
      inflation.Weight(.5);   //  1 / 2 b/c InflationForce   gradient weight incl. factor 2
      distortion.Weight(.05); // .1 / 2 b/c MetricDistortion gradient weight incl. factor 2
      inflation_error.Threshold(.015);
      max_dt.resize(1);
      max_dt[0] = .9;
      model.NeighborhoodRadius(2);
      UniquePtr<EulerMethodWithMomentum> euler(new EulerMethodWithMomentum());
      euler->Momentum(.9);
      euler->NormalizeStepLength(false);
      euler->MaximumDisplacement(1.0);
      optimizer.reset(euler.release());
      center_output    = true;
      match_area       = true;
      signed_gradient  = false;
      track_name       = "SulcalDepth";
      track_zero_mean  = true;
      track_unit_var   = false;
      track_use_median = false;
    }
    // Optimization method
    else if (OPTION("-optimizer") || OPTION("-optimiser")) {
      OptimizationMethod m;
      PARSE_ARGUMENT(m);
      optimizer.reset(LocalOptimizer::New(m, &model));
    }
    else if (OPTION("-line-search") || OPTION("-linesearch")) {
      Insert(params, "Line search strategy", ARGUMENT);
    }
    else if (OPTION("-dof")) {
      string arg = ARGUMENT;
      double dx = 1.0, dy = 1.0, dz = 1.0;
      if (HAS_ARGUMENT) {
        PARSE_ARGUMENT(dx);
        if (HAS_ARGUMENT) {
          PARSE_ARGUMENT(dy);
          PARSE_ARGUMENT(dz);
        } else {
          dy = dz = dx;
        }
      }
      string larg = ToLower(arg);
      if (larg == "none") {
        dof.reset(NULL);
      } else if (larg == "ffd") {
        dof.reset(new BSplineFreeFormTransformation3D(domain, dx, dy, dz));
      } else if (larg == "svffd") {
        dof.reset(new BSplineFreeFormTransformationSV(domain, dx, dy, dz));
      } else {
        TransformationType type = Transformation::TypeOfClass(arg.c_str());
        if (type == TRANSFORMATION_UNKNOWN) {
          FatalError("Invalid -dof transformation type argument: " << arg);
        }
        dof.reset(Transformation::New(type));
      }
      model.Transformation(dof.get());
    }
    else if (OPTION("-levels")) {
      int min_level, max_level;
      PARSE_ARGUMENT(min_level);
      if (HAS_ARGUMENT) {
        PARSE_ARGUMENT(max_level);
      } else {
        max_level = min_level;
        min_level = 1;
      }
      if (min_level < 1 || max_level < 1) {
        FatalError("Invalid -levels argument");
      }
      navgs   = GradientAveraging(min_level, max_level);
      nlevels = max(nlevels, static_cast<int>(navgs.size()));
    }
    else if (OPTION("-force-averaging")) {
      PARSE_ARGUMENTS(int, navgs);
    }
    else if (OPTION("-max-steps")      || OPTION("-steps")      ||
             OPTION("-max-iterations") || OPTION("-iterations") ||
             OPTION("-max-iter")       || OPTION("-iter")) {
      PARSE_ARGUMENTS(int, nsteps);
    }
    else if (OPTION("-step") || OPTION("-dt") || OPTION("-h")) {
      PARSE_ARGUMENTS(double, max_dt);
    }
    else if (OPTION("-max-dx") || OPTION("-maxdx") || OPTION("-maxd") || OPTION("-dx") || OPTION("-d")) {
      PARSE_ARGUMENTS(double, max_dx);
    }
    else if (OPTION("-normalize-forces") || OPTION("-normalise-forces")) {
      Insert(params, "Normalize gradient vectors", true);
    }
    else if (OPTION("-damping"))   Insert(params, "Deformable surface damping", ARGUMENT);
    else if (OPTION("-momentum"))  Insert(params, "Deformable surface momentum", ARGUMENT);
    else if (OPTION("-mass"))      Insert(params, "Deformable surface mass", ARGUMENT);
    else if (OPTION("-epsilon"))   Insert(params, "Epsilon", ARGUMENT);
    else if (OPTION("-delta"))     Insert(params, "Delta", ARGUMENT);
    else if (OPTION("-min-energy") || OPTION("-minenergy")) {
      Insert(params, "Target energy function value", ARGUMENT);
    }
    else if (OPTION("-min-active") || OPTION("-minactive")) {
      PARSE_ARGUMENT(farg);
      min_active.Threshold(farg);
    }
    else if (OPTION("-extrinsic-energy")) {
      model.MinimizeExtrinsicEnergy(true);
    }
    else if (OPTION("-reset-status")) reset_status = true;
    // Force terms
    else if (OPTION("-neighborhood") || OPTION("-neighbourhood")) {
      model.NeighborhoodRadius(atoi(ARGUMENT));
    }
    else if (OPTION("-distance")) {
      PARSE_ARGUMENT(farg);
      distance.Weight(farg);
      signed_gradient = true;
    }
    else if (OPTION("-distance-averaging")) {
      PARSE_ARGUMENTS(int, distance_navgs);
    }
    else if (OPTION("-distance-measure")) {
      ImplicitSurfaceDistance::DistanceMeasureType measure;
      PARSE_ARGUMENT(measure);
      distance.DistanceMeasure(measure);
    }
    else if (OPTION("-balloon-inflation") || OPTION("-balloon")) {
      PARSE_ARGUMENT(farg);
      balloon.Weight(farg);
      balloon.DeflateSurface(false);
    }
    else if (OPTION("-balloon-deflation")) {
      PARSE_ARGUMENT(farg);
      balloon.Weight(farg);
      balloon.DeflateSurface(true);
    }
    else if (OPTION("-balloon-min")) {
      PARSE_ARGUMENT(farg);
      balloon.LowerIntensity(farg);
    }
    else if (OPTION("-balloon-max")) {
      PARSE_ARGUMENT(farg);
      balloon.UpperIntensity(farg);
    }
    else if (OPTION("-balloon-range")) {
      PARSE_ARGUMENT(farg);
      balloon.LowerIntensity(farg);
      PARSE_ARGUMENT(farg);
      balloon.UpperIntensity(farg);
    }
    else if (OPTION("-balloon-radius")) {
      PARSE_ARGUMENT(farg);
      balloon.Radius(farg);
    }
    else if (OPTION("-balloon-sigma")) {
      PARSE_ARGUMENT(farg);
      balloon.LowerIntensitySigma(farg);
      balloon.UpperIntensitySigma(farg);
    }
    else if (OPTION("-balloon-lower-sigma")) {
      PARSE_ARGUMENT(farg);
      balloon.LowerIntensitySigma(farg);
    }
    else if (OPTION("-balloon-upper-sigma")) {
      PARSE_ARGUMENT(farg);
      balloon.UpperIntensitySigma(farg);
    }
    else if (OPTION("-balloon-averaging")) {
      PARSE_ARGUMENTS(int, balloon_navgs);
    }
    else if (OPTION("-balloon-mask")) {
      balloon_mask_name = ARGUMENT;
    }
    else if (OPTION("-edges")) {
      PARSE_ARGUMENT(farg);
      edges.Weight(farg);
    }
    else if (OPTION("-inflation")) {
      PARSE_ARGUMENT(farg);
      inflation.Name("Inflation");
      inflation.Weight(farg);
    }
    else if (OPTION("-bending-energy")) {
      PARSE_ARGUMENT(farg);
      dofbending.Weight(farg);
    }
		else if (OPTION("-spring") || OPTION("-bending")) {
      PARSE_ARGUMENT(farg);
      spring.Weight(farg);
    }
    else if (OPTION("-normal-spring") || OPTION("-nspring")) {
      PARSE_ARGUMENT(farg);
      spring.NormalWeight(farg);
    }
    else if (OPTION("-tangential-spring") || OPTION("-tspring")) {
      PARSE_ARGUMENT(farg);
      spring.TangentialWeight(farg);
    }
    else if (OPTION("-normalized-spring")) {
      PARSE_ARGUMENT(farg);
      normspring.Weight(farg);
    }
    else if (OPTION("-distance-spring") || OPTION("-dspring")) {
      PARSE_ARGUMENT(farg);
      dspring.Weight(farg);
    }
    else if (OPTION("-curvature")) {
      PARSE_ARGUMENT(farg);
      curvature.Weight(farg);
    }
    else if (OPTION("-gauss-curvature") || OPTION("-gaussian-curvature") || OPTION("-gcurvature")) {
      PARSE_ARGUMENT(farg);
      gcurvature.Weight(farg);
    }
    else if (OPTION("-mean-curvature")) {
      PARSE_ARGUMENT(farg);
      mcurvature.Weight(farg);
    }
    else if (OPTION("-quadratic-curvature") || OPTION("-qcurvature")) {
      PARSE_ARGUMENT(farg);
      qcurvature.Weight(farg);
    }
    else if (OPTION("-distortion")) {
      PARSE_ARGUMENT(farg);
      distortion.Weight(farg);
    }
    else if (OPTION("-stretching")) {
      PARSE_ARGUMENT(farg);
      stretching.Weight(farg);
    }
    else if (OPTION("-stretching-rest-length")) {
      const char *arg = ARGUMENT;
      if (strcmp(arg, "avg") == 0) {
        stretching.RestLength(-1.);
        stretching.UseCurrentAverageLength(false);
      } else if (strcmp(arg, "curavg") == 0) {
        stretching.RestLength(-1.);
        stretching.UseCurrentAverageLength(true);
      } else if (FromString(arg, farg)) {
        stretching.RestLength(farg);
        stretching.UseCurrentAverageLength(false);
      } else {
        FatalError("Invalid -stretching-rest-length argument: " << arg);
      }
    }
    else if (OPTION("-repulsion")) {
      PARSE_ARGUMENT(farg);
      repulsion.Weight(farg);
      if (HAS_ARGUMENT) {
        PARSE_ARGUMENT(farg);
        repulsion.Radius(farg);
        if (HAS_ARGUMENT) {
          PARSE_ARGUMENT(farg);
          repulsion.Magnitude(farg);
        } else {
          repulsion.Magnitude(10.);
        }
      }
      else repulsion.Radius(-1.);
    }
    else if (OPTION("-collision")) {
      PARSE_ARGUMENT(farg);
      collision.Weight(farg);
    }
    // Stopping criteria
    else if (OPTION("-inflation-error")) {
      PARSE_ARGUMENT(farg);
      inflation_error.Threshold(farg);
    }
    // Iterative local remeshing
    else if (OPTION("-remesh")) {
      PARSE_ARGUMENT(iarg);
      model.RemeshInterval(iarg);
    }
    else if (OPTION("-remesh-adaptively")) {
      model.RemeshAdaptively(true);
    }
    else if (OPTION("-min-edge-length") || OPTION("-min-edgelength") || OPTION("-minedgelength")) {
      PARSE_ARGUMENTS(double, min_edge_length);
    }
    else if (OPTION("-max-edge-length") || OPTION("-max-edgelength") || OPTION("-maxedgelength")) {
      PARSE_ARGUMENTS(double, max_edge_length);
    }
    else if (OPTION("-min-angle") || OPTION("-minangle")) {
      PARSE_ARGUMENTS(double, min_edge_angle);
    }
    else if (OPTION("-max-angle") || OPTION("-maxangle")) {
      PARSE_ARGUMENTS(double, max_edge_angle);
    }
    else if (OPTION("-triangle-inversion")) {
      model.AllowTriangleInversion(true);
    }
    else if (OPTION("-notriangle-inversion")) {
      model.AllowTriangleInversion(false);
    }
    // Iterative low-pass filtering
    else if (OPTION("-lowpass")) {
      PARSE_ARGUMENT(iarg);
      model.LowPassInterval(iarg);
    }
    else if (OPTION("-lowpass-iterations")) {
      PARSE_ARGUMENT(iarg);
      model.LowPassIterations(iarg);
    }
    else if (OPTION("-lowpass-band")) {
      PARSE_ARGUMENT(farg);
      model.LowPassBand(farg);
    }
    // Non-self-intersection / collision detection
    else if (OPTION("-nointersection")) model.HardNonSelfIntersection(true);
    else if (OPTION("-min-distance") || OPTION("-mindistance") || OPTION("-mind")) {
      PARSE_ARGUMENT(farg);
      model.MinFrontfaceDistance(farg);
    }
    else if (OPTION("-min-width") || OPTION("-minwidth") || OPTION("-minw")) {
      PARSE_ARGUMENT(farg);
      model.MinBackfaceDistance(farg);
    }
    else if (OPTION("-max-collision-angle")) {
      PARSE_ARGUMENT(farg);
      model.MaxCollisionAngle(farg);
    }
    else if (OPTION("-fast-collision-test")) {
      PARSE_ARGUMENT(barg);
      model.FastCollisionTest(barg);
    }
    // Output format
    else if (OPTION("-center-output"))  center_output  = true;
    else if (OPTION("-match-area"))     match_area     = true;
    else if (OPTION("-match-sampling")) match_sampling = true;
    else if (OPTION("-track")) {
      if (HAS_ARGUMENT) track_name = ARGUMENT;
      else              track_name = "NormalDisplacement";
      track_zero_mean  = false;
      track_unit_var   = false;
      track_use_median = false;
    }
    else if (OPTION("-notrack")) track_name = nullptr;
    else if (OPTION("-track-zvalues")) {
      if (HAS_ARGUMENT)     track_name = ARGUMENT;
      else if (!track_name) track_name = "NormalDisplacementZValues";
      track_zero_mean  = true;
      track_unit_var   = true;
      track_use_median = false;
    }
    else if (OPTION("-track-zero-median-zvalues")) {
      if (HAS_ARGUMENT)     track_name = ARGUMENT;
      else if (!track_name) track_name = "NormalDisplacementZeroMedianZValues";
      track_zero_mean  = true;
      track_unit_var   = true;
      track_use_median = true;
    }
    else if (OPTION("-track-zero-mean")) {
      if (HAS_ARGUMENT)     track_name = ARGUMENT;
      else if (!track_name) track_name = "NormalDisplacementZeroMean";
      track_zero_mean  = true;
      track_use_median = false;
    }
    else if (OPTION("-track-zero-median")) {
      if (HAS_ARGUMENT)     track_name = ARGUMENT;
      else if (!track_name) track_name = "NormalDisplacementZeroMedian";
      track_zero_mean  = true;
      track_use_median = true;
    }
    else if (OPTION("-track-unit-variance")) {
      if (HAS_ARGUMENT)     track_name = ARGUMENT;
      else if (!track_name) track_name = "NormalDisplacementUnitVariance";
      track_zero_mean  = false;
      track_use_median = false;
      track_unit_var   = true;
    }
    else if (OPTION("-track-without-momentum")) {
      Insert(params, "Exclude momentum from tracked normal displacement", true);
    }
    else if (OPTION("-save-status")) save_status = true;
    else if (OPTION("-level-prefix") || OPTION("-levelprefix")) {
      level_prefix = true;
    }
    else if (OPTION("-nolevel-prefix") || OPTION("-nolevelprefix")) {
      level_prefix = false;
    }
    // Debugging and other common/advanced options
    else if (OPTION("-par")) {
      const char *name  = ARGUMENT;
      const char *value = ARGUMENT;
      Insert(params, name, value);
    }
    else if (OPTION("-debug-prefix") || OPTION("-debugprefix")) {
      debug_prefix = ARGUMENT;
    }
    else if (OPTION("-debug-interval") || OPTION("-debuginterval")) {
      PARSE_ARGUMENT(iarg);
      debugger.Interval(iarg);
    }
    else HANDLE_POINTSETIO_OPTION(output_fopt);
    else HANDLE_COMMON_OR_UNKNOWN_OPTION();
  }

  if (spring.Weight() == .0) { // no -spring, but -nspring and/or -tspring
    spring.Weight(spring.NormalWeight() + spring.TangentialWeight());
  } else if (spring.NormalWeight() + spring.TangentialWeight() == .0) {
      // no -nspring and -tspring, but -spring
      spring.NormalWeight(.5);
      spring.TangentialWeight(.5);
  }
  if (spring.NormalWeight() + spring.TangentialWeight() <= .0) {
    spring.Weight(.0);
  }
  if ((balloon.Weight() || edges.Weight()) && !image_name) {
    cerr << "Input -image required by external forces!" << endl;
    exit(1);
  }
  if ((distance.Weight() || dspring.Weight()) && !dmap_name) {
    cerr << "Input -dmap required by implicit surface forces!" << endl;
    exit(1);
  }

  // Read input image
  RealImage input_image;
  RegisteredImage image;
  if (image_name) {
    input_image.Read(image_name);
    input_image.PutBackgroundValueAsDouble(padding, true);
    if (mask_name) input_image.PutMask(new BinaryImage(mask_name), true);
    image.InputImage(&input_image);
    image.Initialize(input_image.Attributes());
    image.Update(true, false, false, true);
    image.SelfUpdate(false);
    model.Image(&image);
  }

  // Read implicit surface distance map
  RealImage input_dmap;
  RegisteredImage dmap;
  if (dmap_name) {
    input_dmap.Read(dmap_name);
    if (dmap_offset) input_dmap -= dmap_offset;
    dmap.InputImage(&input_dmap);
    dmap.Initialize(input_dmap.Attributes());
    dmap.Update(true, false, false, true);
    dmap.SelfUpdate(false);
    model.ImplicitSurface(&dmap);
  }

  // Read implicit surface distance force magnitude map
  RealImage input_dmag;
  RegisteredImage dmag;
  if (dmag_name) {
    input_dmag.Read(dmag_name);
    dmag.InputImage(&input_dmag);
    dmag.Initialize(input_dmag.Attributes());
    dmag.Update(true, false, false, true);
    dmag.SelfUpdate(false);
    distance.MagnitudeImage(&dmag);
    distance.InvertMagnitude(false);
    distance.NormalizeMagnitude(false);
  }

  // Read foreground mask of balloon force
  BinaryImage balloon_mask;
  if (balloon_mask_name) {
    balloon_mask.Read(balloon_mask_name);
    balloon.ForegroundMask(&balloon_mask);
  }

  // Add energy terms
  model.Add(&distance,   false);
  model.Add(&balloon,    false);
  model.Add(&edges,      false);
  model.Add(&dspring,    false);
  model.Add(&spring,     false);
  model.Add(&normspring, false);
  model.Add(&inflation,  false);
  model.Add(&curvature,  false);
  model.Add(&gcurvature, false);
  model.Add(&mcurvature, false);
  model.Add(&qcurvature, false);
  model.Add(&distortion, false);
  model.Add(&stretching, false);
  model.Add(&repulsion,  false);
  model.Add(&collision,  false);
  model.Add(&dofbending, false);

  // Add stopping criteria
  if (min_active.Threshold() >= .0) {
    // Can be disabled with -minactive -1, otherwise, even when the stopping
    // criterion will never be fulfilled, it is needed to label nodes as passive
    optimizer->AddStoppingCriterion(&min_active);
  }
  if (!IsNaN(inflation_error.Threshold())) {
    optimizer->AddStoppingCriterion(&inflation_error);
  }

  // Set parameters
  bool ok = true;
  for (ParameterConstIterator it = params.begin(); it != params.end(); ++it) {
    if (!model     .Set(it->first.c_str(), it->second.c_str()) &&
        !optimizer->Set(it->first.c_str(), it->second.c_str())) {
      cerr << "Unused/invalid parameter: " << it->first << " = " << it->second << endl;
      ok = false;
    }
  }
  if (!ok) cout << endl;

  // Rename spring terms (after setting of parameters!)
  if (spring.Weight()) {
    if (spring.NormalWeight() == .0) {
      spring.Name("Tang. spring");
    }
    if (spring.TangentialWeight() == .0) {
      spring.Name("Normal spring");
    }
  }

  // Initialize deformable surface model
  const double repulsion_radius = repulsion.Radius();
  if (repulsion_radius == 0.) repulsion.Radius(1.);

  model.GradientAveraging(0);
  model.AverageSignedGradients(signed_gradient);
  model.AverageGradientMagnitude(false);
  model.Initialize();

  vtkPointSet  *output   = model.Output();
  vtkPointData *outputPD = output->GetPointData();
  vtkCellData  *outputCD = output->GetCellData();

  // Set output points to initial positions from previous execution
  //
  // This is necessary because some energy terms are based on the properties
  // of the original surface mesh, such as the original surface area.
  // Therefore, the input surface mesh must be identical between executions.
  // To continue optimizing a given deformable model, only replace the points
  // of the output by those of the previous output mesh (-initial argument).
  if (initial_name) {
    if (model.Transformation()) {
      cerr << "Error: Option -initial not allowed when optimizing a parametric deformation!" << endl;
      exit(1);
    }
    vtkSmartPointer<vtkPointSet> initial = ReadPointSet(initial_name);
    if (initial->GetNumberOfPoints() != output->GetNumberOfPoints()) {
      cerr << "Error: Point set with initial deformed mesh points has differing number of points" << endl;
      exit(1);
    }
    output->GetPoints()->DeepCopy(initial->GetPoints());
  }

  // Initialize optimizer
  GradientDescent *gd    = dynamic_cast<GradientDescent *>(optimizer.get());
  EulerMethod     *euler = dynamic_cast<EulerMethod     *>(optimizer.get());

  optimizer->Function(&model);
  optimizer->Initialize();

  if (gd) {
    InexactLineSearch *linesearch;
    BrentLineSearch   *brentls;
    linesearch = dynamic_cast<InexactLineSearch *>(gd->LineSearch());
    brentls    = dynamic_cast<BrentLineSearch   *>(gd->LineSearch());
    if (linesearch) {
      if (!Contains(params, "Strict total step length range")) {
        int strict = (gd->LineSearchStrategy() == LS_Brent ? 0 : 2);
        linesearch->StrictStepLengthRange(strict);
      }
      if (!Contains(params, "Maximum streak of rejected steps")) {
        int maxrejected = 1;
        if (gd->LineSearchStrategy() == LS_Brent) maxrejected = -1;
        linesearch->MaxRejectedStreak(maxrejected);
      }
      if (!Contains(params, "Minimum length of steps")) {
        linesearch->MinStepLength(.01);
      }
      if (!Contains(params, "Maximum length of steps")) {
        linesearch->MaxStepLength(1.0);
      }
      if (!Contains(params, "Maximum no. of line search iterations")) {
        linesearch->NumberOfIterations(12);
      }
    }
    if (brentls) {
      if (!Contains(params, "Brent's line search tolerance")) {
        brentls->Tolerance(.1);
      }
    }
  }

  // Add point data array to keep track of node displacment in normal direction
  // (i.e., sulcal depth measure in case of surface -inflation)
  if (track_name) {
    if (euler == NULL || model.Transformation() != NULL || !IsSurfaceMesh(output)) {
      cerr << "Error: Option -track can currently only be used with an Euler method as -optimizer to" << endl;
      cerr << "       directly deform a surface mesh without a parametric transformation (no input -dof)." << endl;
      exit(1);
    }
    vtkSmartPointer<vtkDataArray> track_array;
    track_array = outputPD->GetArray(track_name);
    if (!track_array) {
      track_array = vtkSmartPointer<vtkFloatArray>::New();
      track_array->SetName(track_name);
      track_array->SetNumberOfComponents(1);
      track_array->SetNumberOfTuples(input->GetNumberOfPoints());
      track_array->FillComponent(0, .0);
      outputPD->AddArray(track_array);
    }
    euler->NormalDisplacement(track_array);
  }

  // Deform surface until either local minimum of energy function is reached
  // or the internal and external forces of the model are in equilibrium
  const double distortion_weight = distortion.Weight();

  if (verbose) {
    cout << endl;
    logger.Verbosity(verbose - 1);
    optimizer->AddObserver(logger);
  }
  if (debug) {
    debugger.Prefix(debug_prefix);
    optimizer->AddObserver(debugger);
  }

  for (int level = 0; level < nlevels; ++level) {

    // Set number of integration steps and length of each step
    const auto dt = ParameterValue(level, nlevels, max_dt, 1.);
    optimizer->Set("Maximum length of steps", ToString(dt).c_str());
    optimizer->NumberOfSteps(ParameterValue(level, nlevels, nsteps, 100));

    // Set maximum node displacement at each step
    if (!max_dx.empty()) {
      const auto dx = ParameterValue(level, nlevels, max_dx, 0.);
      optimizer->Set("Normalize length of steps", "No");
      optimizer->Set("Maximum node displacement", ToString(dx).c_str());
    }

    // Set parameters of iterative remeshing step
    model.MinEdgeLength  (ParameterValue(level, nlevels, min_edge_length, 0.));
    model.MaxEdgeLength  (ParameterValue(level, nlevels, max_edge_length, inf));
    model.MinFeatureAngle(ParameterValue(level, nlevels, min_edge_angle,  180.));
    model.MaxFeatureAngle(ParameterValue(level, nlevels, max_edge_angle,  180.));

    // Set radius of repulsion force based on average edge length range
    if (repulsion_radius == 0.) {
      const auto r = model.MinEdgeLength() + .5 * (model.MaxEdgeLength() - model.MinEdgeLength());
      repulsion.Radius(r);
    }

    // Set number of gradient averaging iterations and adjust metric distortion
    // weight for current level (cf. FreeSurfer's MRISinflateBrain function)
    const auto navg = ParameterValue(level, nlevels, navgs, 0);
    if (inflate_brain) {
      distortion.Weight(distortion_weight * sqrt(double(navg)));
      distortion.GradientAveraging(navg);
    } else {
      model.GradientAveraging(navg);
      distance.GradientAveraging(ParameterValue(level, nlevels, distance_navgs, 0));
      balloon .GradientAveraging(ParameterValue(level, nlevels, balloon_navgs,  0));
    }

    // Reset node status
    if (reset_status) {
      vtkDataArray *status = model.Output()->GetPointData()->GetArray("Status");
      if (status) status->FillComponent(0, 1.0);
    }

    // Initialize optimizer
    optimizer->Initialize();

    // Debug/log output
    if (verbose) {
      cout << "Level " << (level + 1) << " out of " << nlevels << "\n";
    }
    if (verbose > 1) {
      cout << "\n";
      PrintParameter(cout, "Maximum no. of steps", optimizer->NumberOfSteps());
      PrintParameter(cout, "Maximum length of steps", dt);
      PrintParameter(cout, "No. of gradient averaging steps", navg);
      if (model.RemeshInterval() > 0) {
        PrintParameter(cout, "Minimum edge length", model.MinEdgeLength());
        PrintParameter(cout, "Maximum edge length", model.MaxEdgeLength());
        PrintParameter(cout, "Minimum edge angle",  model.MinFeatureAngle());
        PrintParameter(cout, "Maximum edge angle",  model.MaxFeatureAngle());
      }
      if (inflate_brain) {
        PrintParameter(cout, "Distortion weight", distortion.Weight());
      }
      if (repulsion.Weight()) {
        PrintParameter(cout, "Repulsion radius", repulsion.Radius());
      }
    }
    cout << endl;
    if (level_prefix) {
      char prefix[64];
      snprintf(prefix, 64, "%slevel_%d_", debug_prefix, level + 1);
      debugger.Prefix(prefix);
      debugger.Iteration(0);
    }

    // Perform optimization at current level
    optimizer->Run();
    if (verbose) cout << endl;
  }

  optimizer->ClearObservers();

  // Remove stopping criteria to avoid their deletion by the optimizer
  optimizer->RemoveStoppingCriterion(&min_active);
  optimizer->RemoveStoppingCriterion(&inflation_error);

  // Get final output mesh
  output   = model.Output();
  outputPD = output->GetPointData();
  outputCD = output->GetCellData();

  // Remove data arrays used by optimizer
  if (!save_status) inputPD->RemoveArray("Status");
  for (int i = 0; i < outputPD->GetNumberOfArrays(); ++i) {
    const char *name = outputPD->GetArrayName(i);
    if (name) {
      if ((!track_name  || strcmp(name, track_name) != 0) &&
          (!save_status || strcmp(name, "Status")   != 0)) {
        if (inputPD->HasArray(name) == 0) {
          outputPD->RemoveArray(name);
          --i;
        }
      }
    }
  }
  for (int i = 0; i < outputCD->GetNumberOfArrays(); ++i) {
    const char *name = outputCD->GetArrayName(i);
    if (name && inputCD->HasArray(name) == 0) {
      outputCD->RemoveArray(name);
      --i;
    }
  }

  // Normalize sulcal depth measure tracked during inflation process
  if (track_name) {
    vtkDataArray *values = outputPD->GetArray(track_name);
    if (track_zero_mean && track_unit_var) NormalizeValues  (values, track_use_median);
    else if (track_zero_mean)              DemeanValues     (values, track_use_median);
    else if (track_unit_var)               NormalizeVariance(values);
  }

  // Resample remeshed output mesh at corresponding initial points
  if (match_sampling && model.RemeshInterval() > 0) {
    output = ResampleAtInitialPoints(input, output);
  }

  // Center output point set
  if (center_output) Center(output);

  // Scale output surface to match input area
  if (match_area) Scale(output, sqrt(Area(input) / Area(output)));

  // Write deformed output surface
  if (!WritePointSet(POSARG(2), output, output_fopt)) {
    FatalError("Failed to write output to file " << output);
  }

  return 0;
}
