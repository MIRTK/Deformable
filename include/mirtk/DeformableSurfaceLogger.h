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

#ifndef MIRTK_DeformableSurfaceLogger_H
#define MIRTK_DeformableSurfaceLogger_H

#include "mirtk/Observer.h"


namespace mirtk {


/**
 * Prints progress of deformable surface to output stream
 */
class DeformableSurfaceLogger : public Observer
{
  mirtkObjectMacro(DeformableSurfaceLogger);

  // ---------------------------------------------------------------------------
  // Attributes

  /// Verbosity level
  mirtkPublicAttributeMacro(int, Verbosity);

  /// Output stream for progress report
  mirtkPublicAggregateMacro(ostream, Stream);

  /// Whether to use SGR commands for colored terminal output
  mirtkPublicAttributeMacro(bool, Color);

  /// Whether to flush stream buffer after each printed message
  mirtkPublicAttributeMacro(bool, FlushBuffer);

  int _NumberOfIterations;    ///< Number of actual line search iterations
  int _NumberOfSteps;         ///< Number of iterative line search steps
  int _NumberOfGradientSteps; ///< Number of gradient descent steps

  // ---------------------------------------------------------------------------
  // Construction/Destruction
private:

  /// Copy construction
  /// \note Intentionally not implemented.
  DeformableSurfaceLogger(const DeformableSurfaceLogger &);

  /// Assignment operator
  /// \note Intentionally not implemented.
  DeformableSurfaceLogger &operator =(const DeformableSurfaceLogger &);

public:

  /// Constructor
  DeformableSurfaceLogger(ostream * = &cout);

  /// Destructor
  virtual ~DeformableSurfaceLogger();

  /// Handle event and print message to output stream
  void HandleEvent(Observable *, Event, const void *);

};


} // namespace mirtk

#endif // MIRTK_DeformableSurfaceLogger_H 
