#ifndef __HYPER_REDUCTION_POD_GALERKIN_RUNGE_KUTTA__
#define __HYPER_REDUCTION_POD_GALERKIN_RUNGE_KUTTA__

#include "dg/dg_base.hpp"
#include "runge_kutta_base.h"
#include "reduced_order/pod_basis_base.h"

namespace PHiLiP::ODE {

#if PHILIP_DIM==1
template <int dim, typename real, int n_rk_stages, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, int n_rk_stages, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class HyperReducedPODGalerkinRungeKuttaODESolver: public RungeKuttaBase <dim, real, n_rk_stages, MeshType>
{
public:
    HyperReducedPODGalerkinRungeKuttaODESolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input, std::shared_ptr<RKTableauBase<dim,real,MeshType>> rk_tableau_input, std::shared_ptr<EmptyRRKBase<dim,real,MeshType>> RRK_object_input, std::shared_ptr<ProperOrthogonalDecomposition::PODBase<dim>> pod, Epetra_Vector weights);

    std::shared_ptr<ProperOrthogonalDecomposition::PODBase<dim>> pod;

    /// ECSW hyper-reduction weights
    Epetra_Vector ECSW_weights;

    /// Destructor
    virtual ~HyperReducedPODGalerkinRungeKuttaODESolver() {};

    void allocate_runge_kutta_system () override;

    void calculate_stage_solution (int istage, real dt, const bool pseudotime) override;

    void calculate_stage_derivative (int istage, real dt) override;

    void sum_stages (real dt, const bool pseudotime) override;
    
    void apply_limiter () override;

    real adjust_time_step(real dt) override;


private:

    Epetra_CrsMatrix epetra_pod_basis;

    /// POD Basis
    std::shared_ptr<Epetra_CrsMatrix> epetra_test_basis;

    /// Pointer to Epetra Matrix for LHS
    std::shared_ptr<Epetra_CrsMatrix> epetra_reduced_lhs;

    /// dealII indexset for FO solution
    dealii::IndexSet solution_index;

    /// dealII indexset for RO solution
    dealii::IndexSet reduced_index;

    /// Generate test basis
    std::shared_ptr<Epetra_CrsMatrix> generate_test_basis(const Epetra_CrsMatrix &pod_basis);

     /// Generate hyper-reduced residual
    std::shared_ptr<Epetra_Vector> generate_hyper_reduced_residual(Epetra_Vector epetra_right_hand_side, const Epetra_CrsMatrix &test_basis);

    /// Generate reduced LHS
    std::shared_ptr<Epetra_CrsMatrix> generate_reduced_lhs(const Epetra_CrsMatrix &system_matrix, const Epetra_CrsMatrix &test_basis);

    /// Function to multiply a dealii vector by an Epetra Matrix
    int multiply(Epetra_CrsMatrix &epetra_matrix,
                 dealii::LinearAlgebra::distributed::Vector<double> &input_dealii_vector,
                 dealii::LinearAlgebra::distributed::Vector<double> &output_dealii_vector,
                 const dealii::IndexSet &index_set,
                 const bool transpose);

    /// Function to convert a epetra_vector to dealii
    void epetra_to_dealii(Epetra_Vector &epetra_vector,
                          dealii::LinearAlgebra::distributed::Vector<double> &dealii_vector,
                          const dealii::IndexSet &index_set);
};
}

#endif