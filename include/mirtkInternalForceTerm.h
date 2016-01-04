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

#ifndef MIRTK_InternalForceTerm_H
#define MIRTK_InternalForceTerm_H

#include <mirtkEnergyMeasure.h>


namespace mirtk {


// -----------------------------------------------------------------------------
enum InternalForceTerm
{
  IFT_Unknown             = IFT_Begin,              ///< Unknown/invalid internal force
  IFT_Distortion          = EM_MetricDistortion,    ///< Minimize metric distortion
  IFT_Stretching          = EM_Stretching,          ///< Stretching force (rest edge length)
  IFT_Curvature           = EM_Curvature,           ///< Minimize curvature of point set surface
  IFT_NonSelfIntersection = EM_NonSelfIntersection, ///< Repels too close non-neighboring triangles
  IFT_Repulsion           = EM_RepulsiveForce,      ///< Repels too close non-neighboring nodes
  IFT_Inflation           = EM_InflationForce,      ///< Inflate point set surface
};

// -----------------------------------------------------------------------------
inline string ToString(const InternalForceTerm &ift)
{
  EnergyMeasure em = static_cast<EnergyMeasure>(ift);
  if (em <= IFT_Begin || em >= IFT_End) return "Unknown";
  return ToString(em);
}

// -----------------------------------------------------------------------------
inline bool FromString(const char *str, InternalForceTerm &ift)
{
  EnergyMeasure em = EM_Unknown;
  if (FromString(str, em) && IFT_Begin < em && em < IFT_End) {
    ift = static_cast<InternalForceTerm>(em);
    return true;
  } else {
    ift = IFT_Unknown;
    return false;
  }
}


} // namespace mirtk

#endif // MIRTK_InternalForceTerm_H
