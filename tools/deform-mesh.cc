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

const int nsteps = 100;

// =============================================================================
// Help
// =============================================================================

// -----------------------------------------------------------------------------
void PrintHelp(const char *name)
{
  using namespace mirtk;
  cout << "usage: " << name << " <input> <output> [options]" << endl;
  cout << endl;
  cout << "Iteratively minimizes a deformable surface model energy functional." << endl;
  cout << "The gradient of the energy terms are the internal and external forces" << endl;
  cout << "of the deformable surface model." << endl;
  cout << endl;
  cout << "Options:" << endl;
  cout << "  -image <file>    Image on which external forces are based on. (default: none)" << endl;
  cout << "  -mask <file>     Mask defining region in which external forces are non-zero. (default: none)" << endl;
  cout << "  -steps <int>     Maximum number of iterations. (default: " << nsteps << ")" << endl;
  cout << "  -ascii/-binary   Write legacy VTK in ASCII or binary format. (default: binary)" << endl;
  cout << "  -[no]compress    Write XML VTK file with or without compression. (default: compress)" << endl;
  PrintCommonOptions(cout);
}

// =============================================================================
// Auxiliaries
// =============================================================================

// -----------------------------------------------------------------------------
/// Normalize curvature (sulcal depth measure tracked during cortical inflation process)
void NormalizeCurvature(vtkDataArray *curv, bool use_median = true)
{
  double mu, var;

  // Compute mean (or median)
  if (use_median) {
    vtkSmartPointer<vtkDataArray> sorted;
    sorted = vtkSmartPointer<vtkDataArray>::NewInstance(curv);
    sorted->DeepCopy(curv);
    vtkSortDataArray::Sort(sorted);
    mu = sorted->GetComponent(sorted->GetNumberOfTuples() / 2, 0);
  } else {
    mu = .0;
    for (vtkIdType id = 0; id < curv->GetNumberOfTuples(); ++id) {
      mu += curv->GetComponent(id, 0);
    }
    mu /= curv->GetNumberOfTuples();
  }

  // Compute variance
  var = .0;
  for (vtkIdType id = 0; id < curv->GetNumberOfTuples(); ++id) {
    var += pow(curv->GetComponent(id, 0) - mu, 2);
  }
  var /= curv->GetNumberOfTuples();

  // Z-score normalize curvature measures
  double sigma = (var == .0 ? 1.0 : sqrt(var));
  for (vtkIdType id = 0; id < curv->GetNumberOfTuples(); ++id) {
    curv->SetComponent(id, 0, (curv->GetComponent(id, 0) - mu) / sigma);
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
    cerr << "Warning: Cannot resample surface mesh at points corresponding to points of input mesh:"
         << " deformed mesh has not point data array named \"InitialPoints\"" << endl;
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
  const char *image_name     = NULL;
  const char *dmap_name      = NULL;
  double      dmap_offset    = .0;
  const char *mask_name      = NULL;
  const char *track_name     = NULL; // track normal movement of nodes
  const char *initial_name   = NULL;
  const char *debug_prefix   = "deformmesh_";
  double      padding        = numeric_limits<double>::quiet_NaN();
  bool        level_prefix   = true;
  bool        reset_status   = false;
  bool        ascii          = false;
  bool        compress       = true;
  bool        center_output  = false;
  bool        match_area     = false;
  bool        match_sampling = false;
  bool        save_status    = false;
  int         min_level      = 0;
  int         max_level      = 0;
  int         nsteps1        = 100; // iterations at level 1
  int         nstepsN        = 100; // iterations at level N
  double      max_step_length           = 1.0;
  double      step_length_magnification = 1.0;
  double      edge_length_magnification = 1.0;

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
      transform(arg.begin(), arg.end(), arg.begin(), ::tolower);
      if (arg == "none") {
        dof.reset(NULL);
      } else if (arg == "ffd") {
        dof.reset(new BSplineFreeFormTransformation3D(domain, dx, dy, dz));
      } else if (arg == "svffd") {
        dof.reset(new BSplineFreeFormTransformationSV(domain, dx, dy, dz));
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
    else if (OPTION("-track")) {
      if (HAS_ARGUMENT) track_name = ARGUMENT;
      else              track_name = "NormalDisplacement";
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
      distance.Weight(atof(ARGUMENT));
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
    else if (OPTION("-remesh"))        model.RemeshInterval(atoi(ARGUMENT));
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
    else if (OPTION("-save-status"))    save_status    = true;
    else if (OPTION("-ascii" ) || OPTION("-nobinary")) ascii = true;
    else if (OPTION("-binary") || OPTION("-noascii" )) ascii = false;
    else if (OPTION("-compress"))   compress = true;
    else if (OPTION("-nocompress")) compress = false;
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
    // Set number of iterations and length of each step
    double step_length = max_step_length * pow(step_length_magnification, level - 1);
    optimizer->Set("Maximum length of steps", ToString(step_length).c_str());
    optimizer->NumberOfSteps(level > min_level ? nstepsN : nsteps1);
    // Set edge length range
    model.MinEdgeLength(min_edge_length * pow(edge_length_magnification, level - 1));
    model.MaxEdgeLength(max_edge_length * pow(edge_length_magnification, level - 1));
    if (repulsion_radius == .0) {
      const double r = model.MinEdgeLength() + .5 * (model.MaxEdgeLength() - model.MinEdgeLength());
      repulsion.Radius(r);
    }
    // Set number of gradient averaging iterations
    model.GradientAveraging(level > 1 ? static_cast<int>(pow(2, level - 2)) : 0);
    model.AverageSignedGradients(false);
    model.AverageGradientMagnitude(true);
    // Adjust metric distortion weight for current level of brain surface
    // inflation and only average the metric distortion gradient, not the
    // spring force (cf. FreeSurfer's MRISinflateBrain function)
    if (inflation.Weight() != .0) {
      if (max_level != 0) {
        distortion.Weight(distortion_weight * sqrt(model.GradientAveraging()));
      }
      distortion.GradientAveraging(model.GradientAveraging());
      model.GradientAveraging(0);
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
      cout << "Level " << level << "\n\n";
      PrintParameter(cout, "Maximum length of steps", step_length);
      if (model.RemeshInterval() > 0) {
        PrintParameter(cout, "Minimum edge length", model.MinEdgeLength());
        PrintParameter(cout, "Maximum edge length", model.MaxEdgeLength());
      }
      if (repulsion.Weight()) {
        PrintParameter(cout, "Repulsion radius", repulsion.Radius());
      }
      cout << endl;
    }
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
  if (inflation.Weight() && track_name) {
    vtkDataArray *curv = outputPD->GetArray(track_name);
    if (curv) NormalizeCurvature(curv);
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
