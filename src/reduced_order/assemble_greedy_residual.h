#ifndef __ASSEMBLE_PROBLEM_GREEDY__
#define __ASSEMBLE_PROBLEM_GREEDY__

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
class AssembleGreedyRes{
    public:
    AssembleGreedyRes(
        const PHiLiP::Parameters::AllParameters *const parameters_input,
        const dealii::ParameterHandler &parameter_handler_input,
        std::shared_ptr<DGBase<dim,double>> &dg_input, 
        std::shared_ptr<ProperOrthogonalDecomposition::PODBase<dim>> pod,
        Parameters::ODESolverParam::ODESolverEnum &ode_solver_type);
    
    /// Destructor
    ~AssembleGreedyRes () {};

    /// Fill entries of A and b
    void build_problem();

    const Parameters::AllParameters *const all_parameters; ///< Pointer to all parameters

    /// Parameter handler for storing the .prm file being ran
    const dealii::ParameterHandler &parameter_handler;

    /// dg
    std::shared_ptr<DGBase<dim,double>> dg;

    /// POD
    std::shared_ptr<ProperOrthogonalDecomposition::PODBase<dim>> pod;

    /// MPI communicator.
    const MPI_Comm mpi_communicator; 

    /// ODE Solve Type/ Projection Type
    Parameters::ODESolverParam::ODESolverEnum ode_solver_type;

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

    void build_initial_weights();

    void build_initial_target();
    void epetra_to_dealii(Epetra_Vector &epetra_vector, 
                          dealii::LinearAlgebra::distributed::Vector<double> &dealii_vector,
                          dealii::IndexSet index_set);
};

} // namespace HyperReduction
    
} // namespace PHiLiP



#endif
