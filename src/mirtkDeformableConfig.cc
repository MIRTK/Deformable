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

#include <mirtkDeformableConfig.h>
#include <mirtkObjectFactory.h>

#ifndef MIRTK_AUTO_REGISTER
  // Optimizers
  #include <mirtkEulerMethod.h>
  #include <mirtkEulerMethodWithDamping.h>
  #include <mirtkEulerMethodWithMomentum.h>
  // External forces
  #include <mirtkBalloonForce.h>
  #include <mirtkImageEdgeForce.h>
  #include <mirtkImplicitSurfaceDistance.h>
  #include <mirtkImplicitSurfaceSpringForce.h>
  // Internal forces
  #include <mirtkCurvatureConstraint.h>
  #include <mirtkInflationForce.h>
  #include <mirtkMetricDistortion.h>
  #include <mirtkNonSelfIntersectionConstraint.h>
  #include <mirtkQuadraticCurvatureConstraint.h>
  #include <mirtkRepulsiveForce.h>
  #include <mirtkSpringForce.h>
  #include <mirtkStretchingForce.h>
#endif


namespace mirtk {


// -----------------------------------------------------------------------------
static void RegisterOptimizers()
{
  #ifndef MIRTK_AUTO_REGISTER
    mirtkRegisterOptimizerMacro(EulerMethod);
    mirtkRegisterOptimizerMacro(EulerMethodWithDamping);
    mirtkRegisterOptimizerMacro(EulerMethodWithMomentum);
  #endif
}

// -----------------------------------------------------------------------------
static void RegisterExternalForces()
{
  #ifndef MIRTK_AUTO_REGISTER
    mirtkRegisterEnergyTermMacro(BalloonForce);
    mirtkRegisterEnergyTermMacro(ImageEdgeForce);
    mirtkRegisterEnergyTermMacro(ImplicitSurfaceDistance);
    mirtkRegisterEnergyTermMacro(ImplicitSurfaceSpringForce);
  #endif
}

// -----------------------------------------------------------------------------
static void RegisterInternalForces()
{
  #ifndef MIRTK_AUTO_REGISTER
    mirtkRegisterEnergyTermMacro(CurvatureConstraint);
    mirtkRegisterEnergyTermMacro(InflationForce);
    mirtkRegisterEnergyTermMacro(MetricDistortion);
    mirtkRegisterEnergyTermMacro(NonSelfIntersectionConstraint);
    mirtkRegisterEnergyTermMacro(QuadraticCurvatureConstraint);
    mirtkRegisterEnergyTermMacro(RepulsiveForce);
    mirtkRegisterEnergyTermMacro(SpringForce);
    mirtkRegisterEnergyTermMacro(StretchingForce);
  #endif
}

// -----------------------------------------------------------------------------
void InitializeDeformableLibrary()
{
  static bool initialized = false;
  if (!initialized) {
    RegisterOptimizers();
    RegisterExternalForces();
    RegisterInternalForces();
    initialized = true;
  }
}


} // namespace mirtk
