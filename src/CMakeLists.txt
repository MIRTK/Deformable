# ============================================================================
# Medical Image Registration ToolKit (MIRTK)
#
# Copyright 2013-2015 Imperial College London
# Copyright 2013-2015 Andreas Schuh
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

##############################################################################
# @file  CMakeLists.txt
# @brief Build configuration of MIRTK Deformable library.
##############################################################################

set(HEADERS
  mirtkBalloonForce.h
  mirtkCurvatureConstraint.h
  mirtkDeformableConfig.h
  mirtkDeformableSurfaceDebugger.h
  mirtkDeformableSurfaceLogger.h
  mirtkDeformableSurfaceModel.h
  mirtkEulerMethod.h
  mirtkEulerMethodWithDamping.h
  mirtkEulerMethodWithMomentum.h
  mirtkExternalForce.h
  mirtkExternalForceTerm.h
  mirtkImageEdgeForce.h
  mirtkImplicitSurfaceDistance.h
  mirtkImplicitSurfaceForce.h
  mirtkImplicitSurfaceSpringForce.h
  mirtkInflationForce.h
  mirtkInflationStoppingCriterion.h
  mirtkInternalForce.h
  mirtkInternalForceTerm.h
  mirtkMetricDistortion.h
  mirtkMinActiveStoppingCriterion.h
  mirtkNonSelfIntersectionConstraint.h
  mirtkPointSetForce.h
  mirtkQuadraticCurvatureConstraint.h
  mirtkRepulsiveForce.h
  mirtkSpringForce.h
  mirtkStretchingForce.h
  mirtkSurfaceConstraint.h
  mirtkSurfaceForce.h
)

set(SOURCES
  mirtkBalloonForce.cc
  mirtkCurvatureConstraint.cc
  mirtkDeformableConfig.cc
  mirtkDeformableSurfaceDebugger.cc
  mirtkDeformableSurfaceLogger.cc
  mirtkDeformableSurfaceModel.cc
  mirtkEulerMethod.cc
  mirtkEulerMethodWithDamping.cc
  mirtkEulerMethodWithMomentum.cc
  mirtkExternalForce.cc
  mirtkImageEdgeForce.cc
  mirtkImplicitSurfaceDistance.cc
  mirtkImplicitSurfaceForce.cc
  mirtkImplicitSurfaceSpringForce.cc
  mirtkInflationForce.cc
  mirtkInflationStoppingCriterion.cc
  mirtkInternalForce.cc
  mirtkMetricDistortion.cc
  mirtkMinActiveStoppingCriterion.cc
  mirtkNonSelfIntersectionConstraint.cc
  mirtkPointSetForce.cc
  mirtkQuadraticCurvatureConstraint.cc
  mirtkRepulsiveForce.cc
  mirtkSpringForce.cc
  mirtkStretchingForce.cc
  mirtkSurfaceConstraint.cc
  mirtkSurfaceForce.cc
)

set(DEPENDS
  LibCommon
  LibNumerics
  LibImage
  LibPointSet
  ${VTK_LIBRARIES}
)

if (TARGET TBB::tbb)
  list(APPEND DEPENDS TBB::tbb)
endif ()

mirtk_add_library(AUTO_REGISTER)
