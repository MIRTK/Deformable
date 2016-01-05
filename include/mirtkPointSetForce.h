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

#ifndef MIRTK_PointSetForce_H
#define MIRTK_PointSetForce_H

#include <mirtkEnergyTerm.h>

#include <mirtkMap.h>
#include <mirtkVector3D.h>
#include <mirtkRegisteredPointSet.h>

#include <vtkType.h>
#include <vtkSmartPointer.h>
#include <vtkDataArray.h>
#include <vtkPoints.h>


namespace mirtk {


/**
 * Base class for point set force terms
 *
 * Subclasses implement in particular internal and external forces for
 * deformable surface models. Internal force terms may further be used to
 * regularize the deformation of a surface during image/point set registration.
 */
class PointSetForce : public EnergyTerm
{
  mirtkAbstractMacro(PointSetForce);

  // ---------------------------------------------------------------------------
  // Types
public:

  /// Type of gradient w.r.t a single transformed data point
  typedef Vector3D<double> GradientType;

  // ---------------------------------------------------------------------------
  // Attributes
protected:

  /// Transformed point set
  mirtkPublicAggregateMacro(RegisteredPointSet, PointSet);

  /// Number of gradient averaging iterations
  mirtkPublicAttributeMacro(int, GradientAveraging);

  /// Whether to only average gradient vectors pointing in the same direction
  /// as the unsmoothed gradient at the central node (i.e., positive dot product)
  mirtkPublicAttributeMacro(bool, AverageSignedGradients);

  /// Whether to only average the magnitude of the gradient vectors
  mirtkPublicAttributeMacro(bool, AverageGradientMagnitude);

  /// Whether this force is only acting on the point set surface
  ///
  /// This read-only attribute must be set by the subclass constructor.
  /// It is in particular set by the SurfaceForce and SurfaceConstraint
  /// constructors.
  mirtkReadOnlyAttributeMacro(bool, SurfaceForce);

  /// Number of points
  mirtkReadOnlyAttributeMacro(int, NumberOfPoints);

  /// Negative node forces/gradient of external force term
  mirtkComponentMacro(GradientType, Gradient);

  /// Size of gradient vector
  mirtkAttributeMacro(int, GradientSize);

  /// Number of summands in gradient computation
  ///
  /// Intended for use by EvaluateGradient implementations only.
  /// Memory must be allocated by subclass, but will be freed by the base class.
  mirtkAggregateMacro(int, Count);

  /// Size of count vector
  mirtkAttributeMacro(int, CountSize);

  /// Whether Update has not been called since initialization
  mirtkAttributeMacro(bool, InitialUpdate);

  // ---------------------------------------------------------------------------
  // Point set attributes
private:

  typedef Map<string, string>        NameMap;
  typedef NameMap::iterator          NameMapIterator;
  typedef NameMap::const_iterator    NameMapConstIterator;

  /// Maps internal point data name to actual unique point data array name
  NameMap _PointDataName;

protected:

  /// Add given array to point data attributes of deformed point set
  ///
  /// This function should be used by subclasses to add point data arrays to
  /// the point set (or its surface, respectively, if \c _SurfaceForce is \c true).
  /// The added point data is interpolated at new node positions whenever the
  /// deformed point set is being remeshed during the optimization.
  ///
  /// \param[in] name   Name of array. The actual name of the point data array
  ///                   will be made unique by this function which stores an
  ///                   internal map from the given name to the unique array name.
  /// \param[in] data   Point data array.
  /// \param[in] global Whether point data array is shared among point forces.
  ///                   If \c true, the array \p name is used unmodified such
  ///                   that other force terms can reuse the array.
  void AddPointData(const char *name, vtkSmartPointer<vtkDataArray> &data, bool global = false);

  /// Add new point data array of given type with specified number of components
  ///
  /// This function should be used by subclasses to add point data arrays to
  /// the point set (or its surface, respectively, if \c _SurfaceForce is \c true).
  /// The added point data is interpolated at new node positions whenever the
  /// deformed point set is being remeshed during the optimization.
  ///
  /// If an array with the given \p name already exists, it is reused to avoid
  /// unnecessary allocations unless the data type or number of components mismatch.
  /// If _NumberOfPoints is set before by PointForce::Initialize, the
  /// corresponding number of array tuples are allocated by this function.
  /// Otherwise, the array is only instantiated, but not allocated.
  ///
  /// \param[in] name   Name of array. The actual name of the point data array
  ///                   will be made unique by this function which stores an
  ///                   internal map from the given name to the unique array name.
  /// \param[in] c      Number of components.
  /// \param[in] type   Type of data array (e.g., VTK_FLOAT, the default).
  /// \param[in] global Whether point data array is shared among point forces.
  ///                   If \c true, the array \p name is used unmodified such
  ///                   that other force terms can reuse the array.
  ///
  /// \return Pointer to (newly instantiated) array.
  vtkDataArray *AddPointData(const char *name, int c = 1, int type = VTK_FLOAT, bool global = false);

  /// Remove named array from point data attributes of deformed point set
  ///
  /// \param[in] name Name of array as used when the array was added before.
  ///                 This name may differ from the actual unique array name.
  void RemovePointData(const char *name);

  /// Get point data array of deformed point set
  ///
  /// \param[in] name     Name of array as used when the array was added before.
  ///                     This name may differ from the actual unique array name.
  /// \param[in] optional Whether the array may not exist. If \c false, this
  ///                     function raises an error if the array does not exist.
  ///
  /// \return Point data array or NULL if not found (only if \p optional = \c true).
  vtkDataArray *GetPointData(const char *name, bool optional = false) const;

  // ---------------------------------------------------------------------------
  // Construction/Destruction
protected:

  /// Constructor
  PointSetForce(const char * = "", double = 1.0);

  /// Copy constructor
  PointSetForce(const PointSetForce &);

  /// Assignment operator
  PointSetForce &operator =(const PointSetForce &);

  /// Allocate memory for (non-parametric) gradient
  void AllocateGradient(int);

  /// Allocate _Count memory
  void AllocateCount(int);

  /// Copy attributes of this class from another instance
  void CopyAttributes(const PointSetForce &);

public:

  /// Destructor
  virtual ~PointSetForce();

  // ---------------------------------------------------------------------------
  // Initialization
protected:

  /// Common (re-)initialization steps of this class only (non-virtual function!)
  void Init();

  /// Get initial points, possibly pre-transformed by global transformation
  vtkSmartPointer<vtkPoints> GetInitialPoints() const;

public:

  /// Initialize force term once input and parameters have been set
  virtual void Initialize();

  /// Reinitialize force term after change of input topology
  ///
  /// This function is called in particular when an input surface has been
  /// reparameterized, e.g., by a local remeshing filter.
  virtual void Reinitialize();

  // ---------------------------------------------------------------------------
  // Evaluation

  /// Update moving input points and internal state of force term
  virtual void Update(bool = true);

protected:

  /// Evaluate gradient of force term
  ///
  /// \param[in,out] gradient Gradient to which the computed gradient of the
  ///                         force term is added after multiplying by \p weight.
  /// \param[in]     step     Step length for finite differences (unused).
  /// \param[in]     weight   Weight of force term.
  virtual void EvaluateGradient(double *gradient, double step, double weight);

  // ---------------------------------------------------------------------------
  // Debugging
public:

  /// Write input of force term
  virtual void WriteDataSets(const char *, const char *, bool = true) const;

  /// Write gradient of force term
  virtual void WriteGradient(const char *, const char *) const;

};


} // namespace mirtk

#endif // MIRTK_PointSetForce_H 
