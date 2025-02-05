#ifndef __POD_BASIS_INTERFACE__
#define __POD_BASIS_INTERFACE__

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/base/table_handler.h>
#include <eigen/Eigen/Dense>
#include "physics/euler.h"

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {
using Eigen::MatrixXd;
using Eigen::VectorXd;

/// Interface for POD
template <int dim>
class PODBase
{
public:
    /// Virtual destructor
    virtual ~PODBase() = default;

    /// Function to return basis
    virtual std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> getPODBasis() = 0;

    /// Function to return reference state
    virtual dealii::LinearAlgebra::ReadWriteVector<double> getReferenceState() = 0;

    /// Function to return snapshot matrix
    virtual MatrixXd getSnapshotMatrix() = 0;

    /// Function to return Skew-Symmetric Q
    virtual std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> getSkewSymmetric() = 0;

    /// Function to return Vt
    virtual MatrixXd getTestBasis() = 0;

    /// Function to Calculate L2 Error
    virtual void CalculateL2Error(std::shared_ptr <dealii::TableHandler> L2error_data_table,
                   Physics::Euler<dim,dim+2,double> euler_physics_double,
                   double current_time,
                   int iteration) = 0;
};

}
}


#endif