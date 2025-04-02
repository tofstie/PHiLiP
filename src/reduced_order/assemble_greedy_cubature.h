#ifndef __ASSEMBLE_GREEDY_CUBATURE__
#define __ASSEMBLE_GREEDY_CUBATURE__

#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <Epetra_CrsMatrix.h>
#include <Epetra_Vector.h>

#include "parameters/all_parameters.h"
#include "dg/dg_base.hpp"
#include "pod_basis_base.h"


namespace PHiLiP
{
namespace HyperReduction
{
    
template<int dim, int nstate>
class AssembleGreedyCubature{
    public:
    AssembleGreedyCubature(
        const PHiLiP::Parameters::AllParameters *const parameters_input,
        const dealii::ParameterHandler &parameter_handler_input,
        const dealii::LinearAlgebra::ReadWriteVector<double> &initial_weights,
        dealii::LinearAlgebra::ReadWriteVector<double> &b_input,
        const Eigen::MatrixXd &V_target_input,
        const double tolerance
        );
    
    /// Destructor
    ~AssembleGreedyCubature () {};

    /// Fill entries of A and b
    void build_problem();

    const Parameters::AllParameters *const all_parameters; ///< Pointer to all parameters

    /// Parameter handler for storing the .prm file being ran
    const dealii::ParameterHandler &parameter_handler;

    /// MPI communicator.
    const MPI_Comm mpi_communicator; 

    //private:
    /// Initial Target
    Eigen::MatrixXd V_target;
    //dealii::TrilinosWrappers::SparseMatrix V_target;

    /// Initial Weights
    dealii::LinearAlgebra::ReadWriteVector<double> initial_weights;

    /// Final Weights
    dealii::LinearAlgebra::distributed::Vector<double> final_weights;

    /// Matrix for the NNLS Problem
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> A;

    /// RHS Vector for the NNLS Problem
    dealii::LinearAlgebra::ReadWriteVector<double> b;

    /// Indices in set z
    std::vector<int> z_vector;

    /// Number of cells
    const int cell_count;
    /// Number of quad points
    const int n_quad_pts;
    dealii::LinearAlgebra::distributed::Vector<double> get_weights();
    std::vector<int> get_indices();
    
    void epetra_to_dealii(Epetra_Vector &epetra_vector, 
                          dealii::LinearAlgebra::distributed::Vector<double> &dealii_vector,
                          dealii::IndexSet index_set);

    double tolerance;
};

} // namespace HyperReduction
    
} // namespace PHiLiP



#endif
