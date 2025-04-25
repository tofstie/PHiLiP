#ifndef __HYPER_REDUCTION_POD_GALERKIN_RUNGE_KUTTA__
#define __HYPER_REDUCTION_POD_GALERKIN_RUNGE_KUTTA__

#include "dg/dg_base.hpp"
#include "runge_kutta_base.h"
#include "reduced_order/pod_basis_base.h"
#include "runge_kutta_methods/rk_tableau_base.h"
#include "relaxation_runge_kutta/empty_RRK_base.h"

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

    /// Destructor
    virtual ~HyperReducedPODGalerkinRungeKuttaODESolver() {};

    void allocate_runge_kutta_system () override;

    void calculate_stage_solution (int istage, real dt, const bool pseudotime) override;

    void calculate_stage_derivative (int istage, real dt) override;

    void sum_stages (real dt, const bool pseudotime) override;
    
    void apply_limiter () override;

    real adjust_time_step(real dt) override;

protected:
    std::shared_ptr<ProperOrthogonalDecomposition::PODBase<dim>> pod;

    /// ECSW hyper-reduction weights
    Epetra_Vector ECSW_weights;

    /// Stores Butcher tableau a and b, which specify the RK method
    std::shared_ptr<RKTableauBase<dim,real,MeshType>> butcher_tableau;

    /// Reduced Space sized Runge Kutta Stages
    std::vector<dealii::LinearAlgebra::distributed::Vector<double>> reduced_rk_stage;

private:

    Epetra_CrsMatrix epetra_pod_basis;

    /// Test Basis
    std::shared_ptr<Epetra_CrsMatrix> epetra_test_basis;

    /// Trial Basis
    std::shared_ptr<Epetra_CrsMatrix> epetra_trial_basis;

    /// Pointer to Epetra Matrix for LHS
    std::shared_ptr<Epetra_CrsMatrix> epetra_reduced_lhs;

    /// Pointer to Qtx
    std::shared_ptr<Epetra_CrsMatrix> Qtx;
    /// Pointer to Qty
    std::shared_ptr<Epetra_CrsMatrix> Qty;
    /// Pointer to Qtz
    std::shared_ptr<Epetra_CrsMatrix> Qtz;
    /// Pointer to BEtx
    std::shared_ptr<Epetra_CrsMatrix> BEtx;

    /// dealII indexset for FO solution
    dealii::IndexSet solution_index;

    /// dealII indexset for RO solution
    dealii::IndexSet reduced_index;

    /// Generate Hyper-Reduced Mass Matrix
    std::shared_ptr<Epetra_CrsMatrix> generate_hyper_reduced_mass_matrix(const dealii::TrilinosWrappers::SparseMatrix& mass_matrix);

    /// Generate test basis
    std::shared_ptr<Epetra_CrsMatrix> generate_test_basis(const Epetra_CrsMatrix &pod_basis, const bool trial_basis);

    std::shared_ptr<Epetra_CrsMatrix> generate_hyper_test_basis(const Epetra_CrsMatrix &pod_basis);

     /// Generate hyper-reduced residual
    std::shared_ptr<Epetra_Vector> generate_hyper_reduced_residual(Epetra_Vector epetra_right_hand_side, const Epetra_CrsMatrix &test_basis);

    /// Generate reduced LHS
    std::shared_ptr<Epetra_CrsMatrix> generate_reduced_lhs(const Epetra_CrsMatrix &system_matrix, const Epetra_CrsMatrix &test_basis, const Epetra_CrsMatrix &trial_basis);

    /// Generate a reduced LHS from quadrature
    std::shared_ptr<Epetra_CrsMatrix> generate_hyper_reduced_lhs(const Epetra_CrsMatrix &system_matrix, const Epetra_CrsMatrix &test_basis, const Epetra_CrsMatrix &trial_basis);

    /// Function to multiply a dealii vector by an Epetra Matrix
    int multiply(Epetra_CrsMatrix &epetra_matrix,
                 dealii::LinearAlgebra::distributed::Vector<double> &input_dealii_vector,
                 dealii::LinearAlgebra::distributed::Vector<double> &output_dealii_vector,
                 dealii::LinearAlgebra::distributed::Vector<double> &index_vector,
                 const bool transpose);

    /// Function to convert a epetra_vector to dealii
    void epetra_to_dealii(Epetra_Vector &epetra_vector,
                          dealii::LinearAlgebra::distributed::Vector<double> &dealii_vector,
                          dealii::LinearAlgebra::distributed::Vector<double> &index_vector);
};
}

#endif
