#ifndef __POD_BASIS_OFFLINE__
#define __POD_BASIS_OFFLINE__

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/vector_operation.h>
#include <deal.II/numerics/vector_tools.h>

#include <eigen/Eigen/Dense>

#include "dg/dg_base.hpp"
#include "parameters/all_parameters.h"
#include "pod_basis_base.h"

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {
using Eigen::MatrixXd;
using Eigen::VectorXd;

/// Class for Offline Proper Orthogonal Decomposition basis. This class reads some previously computed snapshots stored as files and computes a POD basis.
template <int dim>
class OfflinePOD: public PODBase<dim>
{
public:
    /// Constructor
    explicit OfflinePOD(std::shared_ptr<DGBase<dim,double>> &dg_input);

    ///Function to get POD basis for all derived classes
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> getPODBasis() override;

    ///Function to get POD reference state
    dealii::LinearAlgebra::ReadWriteVector<double> getReferenceState() override;

    /// Function to get snapshot matrix used to build POD basis
    MatrixXd getSnapshotMatrix() override;

    /// Function to return Skew-Symmetric Q
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> getSkewSymmetric() override;

    /// Function to return Vt
    MatrixXd getTestBasis() override;

    /// Read snapshots to build POD basis
    bool getPODBasisFromSnapshots();

    /// POD basis
    void calculatePODBasis(MatrixXd snapshots, std::string reference_type);
    

    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> basis;

    /// Reference state
    dealii::LinearAlgebra::ReadWriteVector<double> referenceState;

    /// dg needed for sparsity pattern of system matrix
    std::shared_ptr<DGBase<dim,double>> dg;

    /// LAPACKFullMatrix for nice printing
    dealii::LAPACKFullMatrix<double> fullBasis;

    /// Matrix containing snapshots
    MatrixXd snapshotMatrix;

    /// Q - Symmetric Skew Matrix
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> Q;

    /// Vt - Test Galerkin Matrix
    Eigen::MatrixXd Vt;

    const MPI_Comm mpi_communicator; ///< MPI communicator.
    const int mpi_rank; ///< MPI rank.

    /// ConditionalOStream.
    /** Used as std::cout, but only prints if mpi_rank == 0
     */
    dealii::ConditionalOStream pcout;

    //📣 Code below is Hyper-Reduction, maybe move this depending on the requirements later on
    /// 

    bool getEntropyPODBasisFromSnapshots();

    bool getEntropyProjPODBasisFromSnapshots();

    bool enrichPOD();

    void debugMatrix(dealii::FullMatrix<double> M);
    /*
    void compute_hyper_reduction(MatrixXd V_target, MatrixXd w_target);
    */
};

}
}

#endif

