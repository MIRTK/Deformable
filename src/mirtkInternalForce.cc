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

#include <mirtkInternalForce.h>


namespace mirtk {


// =============================================================================
// Factory method
// =============================================================================

// -----------------------------------------------------------------------------
InternalForce *InternalForce::New(InternalForceTerm ift, const char *name, double w)
{
  enum EnergyMeasure em = static_cast<enum EnergyMeasure>(ift);
  if (IFT_Begin < em && em < IFT_End) {
    EnergyTerm *term = EnergyTerm::TryNew(em, name, w);
    if (term) return dynamic_cast<InternalForce *>(term);
    cerr << NameOfType() << "::New: Internal point set force not available: ";
  } else {
    cerr << NameOfType() << "::New: Energy term is not an internal point set force: ";
  }
  cerr << ToString(em) << " (" << em << ")" << endl;
  exit(1);
  return NULL;
}

// =============================================================================
// Construction/Destruction
// =============================================================================

// -----------------------------------------------------------------------------
InternalForce::InternalForce(const char *name, double weight)
:
  PointSetForce(name, weight)
{
}

// -----------------------------------------------------------------------------
InternalForce::InternalForce(const InternalForce &other)
:
  PointSetForce(other)
{
}

// -----------------------------------------------------------------------------
InternalForce &InternalForce::operator =(const InternalForce &other)
{
  PointSetForce::operator =(other);
  return *this;
}

// -----------------------------------------------------------------------------
InternalForce::~InternalForce()
{
}


} // namespace mirtk
