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

#ifndef MIRTK_InternalForce_H
#define MIRTK_InternalForce_H

#include <mirtkPointSetForce.h>
#include <mirtkInternalForceTerm.h>


namespace mirtk {


/**
 * Base class for a penalty term imposed on a transformed point set
 *
 * Subclasses represent in particular internal forces on deformable simplicial
 * complexes such as elasticity, strain, curvature, non-self-intersection (i.e.
 * triangle repulsion), and node repulsion.
 *
 * The penalty is minimized by the registration using an instance of
 * RegistrationEnergy with set data similarity and regularization terms.
 * Higher penalties lead to stronger enforcement of the constraint.
 *
 * The same force terms can be used in a "non-parametric" DeformableSurfaceModel
 * as internal force terms to regularize the evolution of the surface model.
 */
class InternalForce : public PointSetForce
{
  mirtkAbstractMacro(InternalForce);

  // ---------------------------------------------------------------------------
  // Construction/Destruction

protected:

  /// Constructor
  InternalForce(const char * = "", double = 1.0);

  /// Copy constructor
  InternalForce(const InternalForce &);

  /// Assignment operator
  InternalForce &operator =(const InternalForce &);

public:

  /// Instantiate new constraint representing specified internal forces
  static InternalForce *New(InternalForceTerm, const char * = "", double = 1.0);

  /// Destructor
  virtual ~InternalForce();

};


} // namespace mirtk

#endif // MIRTK_InternalForce_H
