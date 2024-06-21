#ifndef __POD_GALERKIN_RK_ODE_SOLVER__
#define __POD_GALERKIN_RK_ODE_SOLVER__
#include "JFNK_solver/JFNK_solver.h"
#include "dg/dg_base.hpp"
#include "ode_solver_base.h"
#include "runge_kutta_methods/rk_tableau_base.h"
#include "reduced_order/pod_basis_base.h"

namespace PHiLiP{
namespace ODE {

#if PHILIP_DIM == 1
template <int dim, typename real, int n_rk_stages, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, int n_rk_stages, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class PODGalerkinRKODESolver: public ODESolverBase<dim, real, MeshType>
{
public:
    /// Default Constructor that will set the constants
    PODGalerkinRKODESolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input,
                         std::shared_ptr<RKTableauBase<dim,real,MeshType>> rk_tableau_input,
                         std::shared_ptr<ProperOrthogonalDecomposition::PODBase<dim>> pod);
    /// POD
    std::shared_ptr<ProperOrthogonalDecomposition::PODBase<dim>> pod;
    /// Destructor
    virtual ~PODGalerkinRKODESolver() {};

    /// Function to evaluate solution update
    void step_in_time(real dt, const bool pseudotime) override;

    /// Function to allocate ODE system
    void allocate_ode_system () override;

    /// Generate test basis
    std::shared_ptr<Epetra_CrsMatrix> generate_test_basis(const Epetra_CrsMatrix &epetra_system_matrix, const Epetra_CrsMatrix &pod_basis);

    /// Generate reduced LHS
    std::shared_ptr<Epetra_CrsMatrix> generate_reduced_lhs(const Epetra_CrsMatrix &epetra_system_matrix, const Epetra_CrsMatrix &test_basis);


    
protected:
    //std::shared_ptr<Epetra_CrsMatrix> reduced_lhs;
    /// Stores Butcher tableau a and b, which specify the RK method
    std::shared_ptr<RKTableauBase<dim,real,MeshType>> butcher_tableau;
    /// Implicit solver for diagonally-implicit RK methods, using Jacobian-free Newton-Krylov
    JFNKSolver<dim,real,MeshType> solver;
    /// Storage for the derivative at each Runge-Kutta stage
    std::vector<dealii::LinearAlgebra::distributed::Vector<double>> rk_stage;

    std::vector<dealii::LinearAlgebra::distributed::Vector<double>> reduced_rk_stage;
    /// Indicator for zero diagonal elements; used to toggle implicit solve.
    std::vector<bool> butcher_tableau_aii_is_zero;
    Epetra_CrsMatrix epetra_pod_basis;
    Epetra_CrsMatrix epetra_system_matrix;
    std::shared_ptr<Epetra_CrsMatrix> epetra_test_basis;
    std::shared_ptr<Epetra_CrsMatrix> epetra_reduced_lhs;
    int Multiply(Epetra_CrsMatrix &epetra_matrix, dealii::LinearAlgebra::distributed::Vector<double> &input_dealii_vector,
                 dealii::LinearAlgebra::distributed::Vector<double> &output_dealii_vector, dealii::IndexSet index_set, bool transpose);
    void epetra_to_dealii(Epetra_Vector &epetra_vector, dealii::LinearAlgebra::distributed::Vector<double> &dealii_vector, dealii::IndexSet index_set);
    void debug_Epetra(Epetra_CrsMatrix &epetra_matrix);
};
}
}
#endif