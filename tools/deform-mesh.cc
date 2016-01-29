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

#include <mirtkCommon.h>
#include <mirtkOptions.h>

#include <mirtkImageIOConfig.h>
#include <mirtkNumericsConfig.h>
#include <mirtkDeformableConfig.h>
#include <mirtkTransformationConfig.h>

#include <mirtkPointSetUtils.h>
#include <mirtkTransformation.h>
#include <mirtkBSplineFreeFormTransformation3D.h>
#include <mirtkBSplineFreeFormTransformationSV.h>

// Deformable surface model / parameterization
#include <mirtkDeformableSurfaceModel.h>
#include <mirtkDeformableSurfaceLogger.h>
#include <mirtkDeformableSurfaceDebugger.h>

// Optimization method
#include <mirtkLocalOptimizer.h>
#include <mirtkEulerMethod.h>
#include <mirtkEulerMethodWithMomentum.h>
#include <mirtkGradientDescent.h>
#include <mirtkInexactLineSearch.h>
#include <mirtkBrentLineSearch.h>

// Stopping criteria
#include <mirtkMinActiveStoppingCriterion.h>
#include <mirtkInflationStoppingCriterion.h>

// External forces
#include <mirtkBalloonForce.h>
#include <mirtkImageEdgeForce.h>
#include <mirtkImplicitSurfaceDistance.h>
#include <mirtkImplicitSurfaceSpringForce.h>

// Internal forces
#include <mirtkSpringForce.h>
#include <mirtkInflationForce.h>
#include <mirtkCurvatureConstraint.h>
#include <mirtkQuadraticCurvatureConstraint.h>
#include <mirtkMetricDistortion.h>
#include <mirtkStretchingForce.h>
#include <mirtkRepulsiveForce.h>
#include <mirtkNonSelfIntersectionConstraint.h>

// Transformation constraints
#include <mirtkSmoothnessConstraint.h>

// VTK
#include <vtkPointData.h>
#include <vtkCellData.h>
#include <vtkDataArray.h>
#include <vtkFloatArray.h>
#include <vtkGenericCell.h>
#include <vtkCellTreeLocator.h>
#include <vtkSortDataArray.h>
#include <vtkPolyDataNormals.h>


using namespace mirtk;


// =============================================================================
// Default settings
// =============================================================================

const int    nsteps = 100;
const double dt     = 1.0;

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
  cout << "  -dmap <file>" << endl;
  cout << "      Euclidean distance image on which implicit surface forces are based. (default: none)" << endl;
  cout << "  -dmap-offset <value>" << endl;
  cout << "      Implicit surface isovalue in :option:`-dmap` image. (default: 0)" << endl;
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
  cout << "  -linesearch <name>" << endl;
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
  cout << "      is similar to computing the forces on a coarser mesh. (default: 0 0)" << endl;
  cout << "  -steps | -iterations <n>" << endl;
  cout << "      Maximum number of iterations. (default: " << nsteps << ")" << endl;
  cout << "  -step | -dt <value>" << endl;
  cout << "      Length of integration/gradient steps. (default: " << dt << ")" << endl;
  cout << "  -step-magnification <mag>" << endl;
  cout << "      Multiplicative factor for magnification of maximum :option:`-step` length." << endl;
  cout << "      The step length at level n (highest level at n=1) is :math:`dt * mag^{level-1}`. (default: 1)" << endl;
  cout << "  -maxdx <value>" << endl;
  cout << "      Maximum displacement of a node at each iteration. By default, the node displacements" << endl;
  cout << "      are normalized by the maximum node displacement. When this option is used, the node" << endl;
  cout << "      displacements are clamped to the specified maximum length instead. (default: :option:`-step`)" << endl;
  cout << "  -remesh <n>" << endl;
  cout << "      Remesh surface mesh every n-th iteration. (default: " << model.RemeshInterval() << ")" << endl;
  cout << "  -remesh-adaptively" << endl;
  cout << "      Remesh surface mesh using an adaptive edge length interval based on local curvature" << endl;
  cout << "      of the deformed surface mesh or input implicit surface (:option:`-dmap`)." << endl;
  cout << "  -minedgelength <value>" << endl;
  cout << "      Minimum edge length used for local adaptive remeshing. (default: " << model.MinEdgeLength() << ")" << endl;
  cout << "  -maxedgelength <value>" << endl;
  cout << "      Maximum edge length used for local adaptive remeshing. (default: " << model.MaxEdgeLength() << ")" << endl;
  cout << "  -edgelength-magnification <mag>" << endl;
  cout << "      Multiplicative factor for magnification of :option:`-minedgelength` and :option:`-maxedgelength`." << endl;
  cout << "      The edge length at level n (highest level at n=1) is :math:`l * mag^{level-1}`. (default: 1)" << endl;
  cout << "  -minangle <degrees>" << endl;
  cout << "      Minimum angle between edge node normals for an edge be excluded from collapsing during" << endl;
  cout << "      iterative :option:`-remesh` operations. (default: " << model.MinFeatureAngle() << ")" << endl;
  cout << "  -maxangle <degrees>" << endl;
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
  cout << "  -mind | -mindistance <value>" << endl;
  cout << "      Minimum distance to other triangles in front of a given triangle." << endl;
  cout << "  -minw | -minwidth <value>" << endl;
  cout << "      Minimum distance to other triangles in the back of a given triangle." << endl;
  cout << "  -reset-status" << endl;
  cout << "      Set status of all mesh nodes to active again after each level (see :option:`-levels`). (default: off)" << endl;
  cout << endl;
  cout << "Deformable model options:" << endl;
  cout << "  -neighborhood <n>" << endl;
  cout << "      Size of node neighborhoods used by internal force terms that consider more" << endl;
  cout << "      than only the adjacent nodes, but also up to n-connected nodes. (default: " << model.NeighborhoodRadius() << ")" << endl;
  cout << "  -distance <w>" << endl;
  cout << "      Weight of implicit surface distance. (default: 0)" << endl;
  cout << "  -distance-spring | -dspring <w>" << endl;
  cout << "      Weight of implicit surface spring force. (default: 0)" << endl;
  cout << "  -distance-measure <name>" << endl;
  cout << "      Implicit surface distance measure used by :option:`-distance` and :option:`-distance-spring`):" << endl;
  cout << "      - ``minimum``: Minimum surface distance (see :option:`-dmap`, default)" << endl;
  cout << "      - ``normal``:  Estimate distance by casting rays along normal direction." << endl;
  cout << "  -balloon-inflation | -balloon <w>" << endl;
  cout << "      Weight of inflation force based on local intensity statistics. (default: 0)" << endl;
  cout << "  -balloon-deflation <w>" << endl;
  cout << "      Weight of deflation force based on local intensity statistics. (default: 0)" << endl;
  cout << "  -edges <w>" << endl;
  cout << "      Weight of image edge force. (default: 0)" << endl;
  cout << "  -inflation <w>" << endl;
  cout << "      Weight of surface inflation force used for cortical surface inflation. (default: 0)" << endl;
  cout << "  -bending-energy <w>" << endl;
  cout << "      Weight of bending energy of :option:`-dof` transformation. (default: 0)" << endl;
  cout << "  -spring <w>" << endl;
  cout << "      Weight of internal spring force. (default: 0)" << endl;
  cout << "  -normal-spring | -nspring <w>" << endl;
  cout << "      Weight of internal spring force in normal direction. (default: 0)" << endl;
  cout << "  -tangential-spring | -tspring <w>" << endl;
  cout << "      Weight of internal spring force in tangent plane. (default: 0)" << endl;
  cout << "  -normalized-spring <w>" << endl;
  cout << "      Weight of internal spring force normalized w.r.t. force in normal direction. (default: 0)" << endl;
  cout << "  -curvature <w>" << endl;
  cout << "      Weight of surface curvature. (default: 0)" << endl;
  cout << "  -quadratic-curvature | -qcurvature <w>" << endl;
  cout << "      Weight of surface curvature estimated by quadratic fit of node neighbor" << endl;
  cout << "      to tangent plane distance. (default: 0)" << endl;
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
  cout << "      energy function value (see :option:`-epsilon` and :option:`-minenergy`). (default: off)" << endl;
  cout << "  -epsilon <value>" << endl;
  cout << "      Minimum change of deformable surface energy convergence criterion." << endl;
  cout << "  -delta <value>" << endl;
  cout << "      Minimum maximum node displacement or :option:`-dof` parameter value." << endl;
  cout << "  -minenergy <value>" << endl;
  cout << "      Target deformable surface energy value. (default: 0)" << endl;
  cout << "  -minactive <n>" << endl;
  cout << "      Minimum percentage of active nodes. (default: 0)" << endl;
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
  cout << "  -debugprefix <prefix>" << endl;
  cout << "      File name prefix for :option:`-debug` output. (default: deform_mesh\\_)" << endl;
  cout << "  -debuginterval <n>" << endl;
  cout << "      Write :option:`-debug` output every n-th iteration. (default: 10)" << endl;
  cout << "  -[no]levelprefix" << endl;
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
/// Subract mean of tracked normal displacements
void DemeanValues(vtkDataArray *values, bool use_median = false)
{
  double mu;

  // Compute mean (or median)
  if (use_median) {
    vtkSmartPointer<vtkDataArray> sorted;
    sorted = vtkSmartPointer<vtkDataArray>::NewInstance(values);
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
    sorted = vtkSmartPointer<vtkDataArray>::NewInstance(values);
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
  initial = vtkSmartPointer<vtkPointSet>::NewInstance(output);
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
  resampled = vtkSmartPointer<vtkPointSet>::NewInstance(input);
  resampled->ShallowCopy(input);
  resampled->SetPoints(resampled_points);
  return resampled;
}

// =============================================================================
// Main
// =============================================================================

// -----------------------------------------------------------------------------
int main(int argc, char *argv[])
{
  verbose = 1; // default verbosity level
  EXPECTS_POSARGS(2);

  // Initialize libraries / object factories
  InitializeNumericsLibrary();
  InitializeImageIOLibrary();
  InitializeTransformationLibrary();
  InitializeDeformableLibrary();

  // Deformable surface model and default optimizer
  unique_ptr<Transformation> dof;
  DeformableSurfaceModel     model;
  DeformableSurfaceLogger    logger;
  DeformableSurfaceDebugger  debugger(&model);
  unique_ptr<LocalOptimizer> optimizer(new EulerMethod(&model));
  ParameterList              params;

  // Read input point set
  vtkSmartPointer<vtkPointSet> input = ReadPointSet(POSARG(1), NULL, true);
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
  QuadraticCurvatureConstraint  qcurvature("QCurvature",     .0);
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
  inflation_error.Threshold(numeric_limits<double>::quiet_NaN());

  // Optional arguments
  const char *image_name       = NULL;
  const char *dmap_name        = NULL;
  double      dmap_offset      = .0;
  const char *mask_name        = NULL;
  const char *track_name       = NULL;  // track normal movement of nodes
  bool        track_zero_mean  = false; // subtract mean from tracked normal movements
  bool        track_unit_var   = false; // Z-score normalize tracked normal movements
  bool        track_use_median = false; // use median instead of mean for normalization
  const char *initial_name     = NULL;
  const char *debug_prefix     = "deform_mesh_";
  double      padding          = numeric_limits<double>::quiet_NaN();
  bool        level_prefix     = true;
  bool        reset_status     = false;
  bool        ascii            = false;
  bool        compress         = true;
  bool        center_output    = false;
  bool        match_area       = false;
  bool        match_sampling   = false;
  bool        signed_gradient  = false; // average gradient vectors with positive dot product
  bool        save_status      = false;
  int         min_level        = 0;
  int         max_level        = 0;
  int         nsteps1          = nsteps; // iterations at level 1
  int         nstepsN          = nsteps; // iterations at level N
  double      max_step_length           = dt;
  double      step_length_magnification = 1.0;
  double      edge_length_magnification = 1.0;
  bool        inflate_brain  = false; // mimick mris_inflate


  for (ALL_OPTIONS) {
    // Input
    if (OPTION("-image")) {
      image_name = ARGUMENT;
    }
    else if (OPTION("-dmap") || OPTION("-distance-map") || OPTION("-implicit-surface")) {
      dmap_name = ARGUMENT;
    }
    else if (OPTION("-dmap-offset") || OPTION("-distance-offset") || OPTION("-implicit-surface-offset")) {
      dmap_offset = atof(ARGUMENT);
    }
    else if (OPTION("-mask")) {
      mask_name = ARGUMENT;
    }
    else if (OPTION("-initial")) {
      initial_name = ARGUMENT;
    }
    else if (OPTION("-padding")) {
      const char *arg = ARGUMENT;
      if (!FromString(arg, padding)) {
        cerr << "Invalid -padding argument" << endl;
        exit(1);
      }
    }
    // Presets
    else if (OPTION("-inflate-brain")) { // cf. FreeSurfer's mris_inflate
      inflate_brain = true;
      min_level = 1, max_level = 6;
      nsteps1 = nstepsN = 10;
      inflation.Weight(.5);   //  1 / 2 b/c InflationForce   gradient weight incl. factor 2
      distortion.Weight(.05); // .1 / 2 b/c MetricDistortion gradient weight incl. factor 2
      inflation_error.Threshold(.015);
      max_step_length = .9;
      step_length_magnification = 1.0;
      model.NeighborhoodRadius(2);
      unique_ptr<EulerMethodWithMomentum> euler(new EulerMethodWithMomentum());
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
      const char *arg = ARGUMENT;
      OptimizationMethod m;
      if (!FromString(arg, m)) {
        cerr << "Unknown optimization method: " << arg << endl;
        exit(1);
      }
      optimizer.reset(LocalOptimizer::New(m, &model));
    }
    else if (OPTION("-linesearch")) {
      Insert(params, "Line search strategy", ARGUMENT);
    }
    else if (OPTION("-dof")) {
      string arg = ARGUMENT;
      double dx = 1.0, dy = 1.0, dz = 1.0;
      if (HAS_ARGUMENT) {
        if (!FromString(ARGUMENT, dx)) {
          cerr << "Invalid -dof control point spacing argument" << endl;
          exit(1);
        }
        if (HAS_ARGUMENT) {
          if (!FromString(ARGUMENT, dy) || !FromString(ARGUMENT, dz)) {
            cerr << "Invalid -dof control point spacing argument" << endl;
            exit(1);
          }
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
      const int i = atoi(ARGUMENT);
      if (HAS_ARGUMENT) {
        min_level = i;
        max_level = atoi(ARGUMENT);
      } else {
        min_level = 1;
        max_level = i;
      }
      if (min_level < 1 || max_level < 1) {
        cerr << "Error: Invalid -levels argument" << endl;
        exit(1);
      }
    }
    else if (OPTION("-steps") || OPTION("-iterations")) {
      nstepsN = atoi(ARGUMENT);
      if (HAS_ARGUMENT) nsteps1 = atoi(ARGUMENT);
      else              nsteps1 = nstepsN;
    }
    else if (OPTION("-step") || OPTION("-dt") || OPTION("-h")) {
      max_step_length = atof(ARGUMENT);
    }
    else if (OPTION("-step-magnification")) {
      step_length_magnification = atof(ARGUMENT);
    }
    else if (OPTION("-maxdx") || OPTION("-maxd") || OPTION("-dx") || OPTION("-d")) {
      Insert(params, "Normalize length of steps", false);
      Insert(params, "Maximum node displacement", ARGUMENT);
    }
    else if (OPTION("-damping"))   Insert(params, "Deformable surface damping", ARGUMENT);
    else if (OPTION("-momentum"))  Insert(params, "Deformable surface momentum", ARGUMENT);
    else if (OPTION("-mass"))      Insert(params, "Deformable surface mass", ARGUMENT);
    else if (OPTION("-epsilon"))   Insert(params, "Epsilon", ARGUMENT);
    else if (OPTION("-delta"))     Insert(params, "Delta", ARGUMENT);
    else if (OPTION("-minenergy")) Insert(params, "Target energy function value", ARGUMENT);
    else if (OPTION("-minactive")) min_active.Threshold(atof(ARGUMENT));
    else if (OPTION("-extrinsic-energy")) {
      model.MinimizeExtrinsicEnergy(true);
    }
    else if (OPTION("-reset-status")) reset_status = true;
    // Force terms
    else if (OPTION("-neighborhood") || OPTION("-neighbourhood")) {
      model.NeighborhoodRadius(atoi(ARGUMENT));
    }
    else if (OPTION("-distance")) {
      distance.Weight(atof(ARGUMENT));
      signed_gradient = true;
    }
    else if (OPTION("-distance-measure")) {
      ImplicitSurfaceDistance::DistanceMeasureType measure;
      PARSE_ARGUMENT(measure);
      distance.DistanceMeasure(measure);
    }
    else if (OPTION("-balloon-inflation") || OPTION("-balloon")) {
      balloon.Weight(atof(ARGUMENT));
      balloon.DeflateSurface(false);
    }
    else if (OPTION("-balloon-deflation")) {
      balloon.Weight(atof(ARGUMENT));
      balloon.DeflateSurface(true);
    }
    else if (OPTION("-edges")) {
      edges.Weight(atof(ARGUMENT));
    }
    else if (OPTION("-inflation")) {
      inflation.Name("Inflation");
      inflation.Weight(atof(ARGUMENT));
    }
    else if (OPTION("-bending-energy")) {
      dofbending.Weight(atof(ARGUMENT));
    }
		else if (OPTION("-spring") || OPTION("-bending")) {
      spring.Weight(atof(ARGUMENT));
    }
    else if (OPTION("-normal-spring") || OPTION("-nspring")) {
      spring.NormalWeight(atof(ARGUMENT));
    }
    else if (OPTION("-tangential-spring") || OPTION("-tspring")) {
      spring.TangentialWeight(atof(ARGUMENT));
    }
    else if (OPTION("-normalized-spring")) {
      normspring.Weight(atof(ARGUMENT));
    }
    else if (OPTION("-distance-spring") || OPTION("-dspring")) {
      dspring.Weight(atof(ARGUMENT));
    }
    else if (OPTION("-curvature")) {
      curvature.Weight(atof(ARGUMENT));
      curvature.Sigma(HAS_ARGUMENT ? atof(ARGUMENT) : .0);
    }
    else if (OPTION("-quadratic-curvature") || OPTION("-qcurvature")) {
      qcurvature.Weight(atof(ARGUMENT));
    }
    else if (OPTION("-distortion")) {
      distortion.Weight(atof(ARGUMENT));
    }
    else if (OPTION("-stretching")) {
      stretching.Weight(atof(ARGUMENT));
      stretching.RestLength(HAS_ARGUMENT ? atof(ARGUMENT) : -1.0);
    }
    else if (OPTION("-repulsion")) {
      repulsion.Weight(atof(ARGUMENT));
      repulsion.Radius(HAS_ARGUMENT ? atof(ARGUMENT) : 0);
    }
    else if (OPTION("-collision")) collision.Weight(atof(ARGUMENT));
    // Stopping criteria
    else if (OPTION("-inflation-error")) {
      inflation_error.Threshold(atof(ARGUMENT));
    }
    // Iterative local remeshing
    else if (OPTION("-remesh")) model.RemeshInterval(atoi(ARGUMENT));
    else if (OPTION("-remesh-adaptively")) model.RemeshAdaptively(true);
    else if (OPTION("-minedgelength")) model.MinEdgeLength(atof(ARGUMENT));
    else if (OPTION("-maxedgelength")) model.MaxEdgeLength(atof(ARGUMENT));
    else if (OPTION("-edgelength-magnification")) {
      edge_length_magnification = atof(ARGUMENT);
    }
    else if (OPTION("-minangle")) model.MinFeatureAngle(atof(ARGUMENT));
    else if (OPTION("-maxangle")) model.MaxFeatureAngle(atof(ARGUMENT));
    // Iterative low-pass filtering
    else if (OPTION("-lowpass"))            model.LowPassInterval(atoi(ARGUMENT));
    else if (OPTION("-lowpass-iterations")) model.LowPassIterations(atoi(ARGUMENT));
    else if (OPTION("-lowpass-band"))       model.LowPassBand(atof(ARGUMENT));
    // Non-self-intersection / collision detection
    else if (OPTION("-nointersection")) model.HardNonSelfIntersection(true);
    else if (OPTION("-mindistance") || OPTION("-mind")) model.MinFrontfaceDistance(atof(ARGUMENT));
    else if (OPTION("-minwidth")    || OPTION("-minw")) model.MinBackfaceDistance (atof(ARGUMENT));
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
    else if (OPTION("-save-status"))    save_status    = true;
    else if (OPTION("-ascii" ) || OPTION("-nobinary")) ascii = true;
    else if (OPTION("-binary") || OPTION("-noascii" )) ascii = false;
    else if (OPTION("-compress"))   compress = true;
    else if (OPTION("-nocompress")) compress = false;
    else if (OPTION("-levelprefix")) level_prefix = true;
    else if (OPTION("-nolevelprefix")) level_prefix = false;
    // Debugging and other common/advanced options
    else if (OPTION("-par")) {
      const char *name  = ARGUMENT;
      const char *value = ARGUMENT;
      Insert(params, name, value);
    }
    else if (OPTION("-debugprefix")) debug_prefix = ARGUMENT;
    else if (OPTION("-debuginterval")) debugger.Interval(atoi(ARGUMENT));
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

  Insert(params, "Maximum length of steps", max_step_length);
  const double repulsion_radius  = repulsion.Radius();
  const double distortion_weight = distortion.Weight();
  const double min_edge_length   = model.MinEdgeLength();
  const double max_edge_length   = model.MaxEdgeLength();

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

  // Read input distance map
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

  // Add energy terms
  model.Add(&distance,   false);
  model.Add(&balloon,    false);
  model.Add(&edges,      false);
  model.Add(&dspring,    false);
  model.Add(&spring,     false);
  model.Add(&normspring, false);
  model.Add(&inflation,  false);
  model.Add(&curvature,  false);
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
  if (repulsion_radius == .0) repulsion.Radius(1.0);

  model.GradientAveraging(0);
  model.AverageSignedGradients(signed_gradient);
  model.AverageGradientMagnitude(true);
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
  if (verbose) {
    cout << endl;
    logger.Verbosity(verbose - 1);
    optimizer->AddObserver(logger);
  }
  if (debug) {
    debugger.Prefix(debug_prefix);
    optimizer->AddObserver(debugger);
  }

  for (int level = max_level; level >= min_level; --level) {
    // Number of gradient averaging iterations
    const int navgs = (level > 1 ? static_cast<int>(pow(2, level - 2)) : 0);
    // Set number of iterations and length of each step
    double step_length = max_step_length;
    if (level > 1) {
      step_length *= pow(step_length_magnification, level - 1);
    }
    optimizer->Set("Maximum length of steps", ToString(step_length).c_str());
    optimizer->NumberOfSteps(level > min_level ? nstepsN : nsteps1);
    // Set edge length range
    model.MinEdgeLength(min_edge_length * pow(edge_length_magnification, level - 1));
    model.MaxEdgeLength(max_edge_length * pow(edge_length_magnification, level - 1));
    if (repulsion_radius == .0) {
      const double r = model.MinEdgeLength() + .5 * (model.MaxEdgeLength() - model.MinEdgeLength());
      repulsion.Radius(r);
    }
    // Set number of gradient averaging iterations and adjust metric distortion
    // weight for current level (cf. FreeSurfer's MRISinflateBrain function)
    if (inflate_brain) {
      distortion.Weight(distortion_weight * sqrt(double(navgs)));
      distortion.GradientAveraging(navgs);
    } else {
      model.GradientAveraging(navgs);
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
      cout << "Level " << level << "\n";
    }
    if (verbose > 1) {
      cout << "\n";
      if (inflate_brain) {
        PrintParameter(cout, "Distortion weight", distortion.Weight());
      }
      PrintParameter(cout, "No. of gradient averaging iterations", navgs);
      PrintParameter(cout, "Maximum length of steps", step_length);
      if (model.RemeshInterval() > 0) {
        PrintParameter(cout, "Minimum edge length", model.MinEdgeLength());
        PrintParameter(cout, "Maximum edge length", model.MaxEdgeLength());
      }
      if (repulsion.Weight()) {
        PrintParameter(cout, "Repulsion radius", repulsion.Radius());
      }
    }
    cout << endl;
    if (level_prefix) {
      char prefix[64];
      snprintf(prefix, 64, "%slevel_%d_", debug_prefix, level);
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
  return WritePointSet(POSARG(2), output, compress, ascii) ? 0 : 1;
}
