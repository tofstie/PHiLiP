#include "pod_galerkin_rk_ode_solver.h"

#include <Amesos_Lapack.h>
#include <Epetra_LinearProblem.h>
#include <Epetra_Vector.h>
#include <EpetraExt_MatrixMatrix.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <iostream>
#include "Amesos_BaseSolver.h"

namespace PHiLiP {
namespace ODE {

template <int dim, typename real, int n_rk_stages, typename MeshType>
PODGalerkinRKODESolver<dim,real,n_rk_stages,MeshType>::PODGalerkinRKODESolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input,
                                                                          std::shared_ptr<RKTableauBase<dim,real,MeshType>> rk_tableau_input,
                                                                          std::shared_ptr<ProperOrthogonalDecomposition::PODBase<dim>> pod)
                                                                          : ODESolverBase<dim,real,MeshType>(dg_input)
                                                                          , pod(pod) 
                                                                          , butcher_tableau(rk_tableau_input)
                                                                          , solver(dg_input)
{}

template <int dim, typename real, int n_rk_stages, typename MeshType>
void PODGalerkinRKODESolver<dim, real, n_rk_stages, MeshType>::step_in_time (real dt, const bool pseudotime)
{
    this->original_time_step = dt;
    this->solution_update = this->dg->solution;

    Epetra_CrsMatrix epetra_system_matrix = this->dg->global_mass_matrix.trilinos_matrix();
    const Epetra_CrsMatrix epetra_pod_basis = pod->getPODBasis()->trilinos_matrix();
    std::shared_ptr<Epetra_CrsMatrix> epetra_test_basis = generate_test_basis(epetra_pod_basis, epetra_pod_basis);
    std::shared_ptr<Epetra_CrsMatrix> epetra_reduced_lhs = generate_reduced_lhs(epetra_system_matrix, *epetra_test_basis); // Memory leak

    for (int i = 0; i < n_rk_stages; ++i) {
         this->rk_stage[i] = 0.0;

        for(int j = 0; j < i; ++j){
            if(this->butcher_tableau->get_a(i,j) != 0){
                Epetra_Vector epetra_rk_stage_j(Epetra_DataAccess::View, epetra_system_matrix.RowMap(), this->rk_stage[j].begin());
                dealii::LinearAlgebra::distributed::Vector<double> dealii_rk_stage_j;
                Epetra_Vector epetra_reduced_stage_j(epetra_system_matrix.RangeMap());
                epetra_test_basis->Multiply(false, epetra_rk_stage_j, epetra_reduced_stage_j);
                epetra_to_dealii(epetra_reduced_stage_j,dealii_rk_stage_j); // Fix input types
                this->rk_stage[i].add(this->butcher_tableau->get_a(i,j),dealii_rk_stage_j);
            }
        } //sum(a_ij *k_j), explicit part
        if(pseudotime) {
            const double CFL = dt;
            this->dg->time_scale_solution_update(rk_stage[i], CFL);
        }else {
            this->rk_stage[i]*=dt; 
        }//dt * sum(a_ij * k_j)
        /* Note sure if this section is needed
        Epetra_Vector epetra_reduced_x(Epetra_DataAccess::View, epetra_system_matrix.RowMap(), this->solution_update.begin());
        Epetra_Vector epetra_W_reduced_x(epetra_test_basis->DomainMap()); // W*x̂
        epetra_test_basis.Multiply(false, epetra_reduced_x, epetra_W_reduced_x);
        dealii::LinearAlgebra::distributed::Vector<double> dealii_W_reduced_x;
        epetra_to_dealii(epetra_W_reduced_x,dealii_W_reduced_x);
        this->rk_stage[i].add(1.0,dealii_W_reduced_x,1.0,reference_solution); //+u_o + u_n + dt * sum(a_ij * W * k_j)
        */
        this->rk_stage[i].add(1.0,this->solution_update);
        if (!this->butcher_tableau_aii_is_zero[i]){
            // Implicit, not sure on JFNK works for reduced order or not, assuming it does for now
            solver.solve(dt*this->butcher_tableau->get_a(i,i), this->rk_stage[i]);
            this->rk_stage[i] = solver.current_solution_estimate;
        }
        this->dg->solution = this->rk_stage[i];

        // Not including limiter yet as not sure of implications on ROM

        //set the DG current time for unsteady source terms
        this->dg->set_current_time(this->current_time + this->butcher_tableau->get_c(i)*dt);
        //solve the system's right hand side
        this->dg->assemble_residual(); //RHS : du/dt = RHS = F(u_n + dt* sum(a_ij*k_j) + dt * a_ii * u^(i)))
        if(this->all_parameters->use_inverse_mass_on_the_fly){
            //this->dg->apply_inverse_global_mass_matrix(this->dg->right_hand_side, this->rk_stage[i]); //rk_stage[i] = IMM*RHS = F(u_n + dt*sum(a_ij*k_j))
            // Not implmenting this yet as requires editting DG base
        } else{
            dealii::LinearAlgebra::distributed::Vector<double> dealii_reduced_rhs;
            Epetra_Vector eptra_rhs(Epetra_DataAccess::View, epetra_test_basis->RowMap(), this->dg->right_hand_side.begin());
            Epetra_Vector epetra_reduced_rhs(epetra_test_basis->DomainMap());
            
            epetra_test_basis->Multiply(true,eptra_rhs,epetra_reduced_rhs);
            Epetra_Vector epetra_rk_stage_i(epetra_reduced_lhs->DomainMap());
            Epetra_LinearProblem linearProblem(epetra_reduced_lhs.get(), &epetra_rk_stage_i, &epetra_reduced_rhs);
            //std::cout << linearProblem.CheckInput() << std::endl;
            Amesos_Lapack Solver(linearProblem);
            Teuchos::ParameterList List;
            //List.set("MatrixType", "general");
            //std::cout << Solver.MatrixShapeOK() << std::endl;
            Solver.SetParameters(List); //Deprecated, change?
            Solver.SymbolicFactorization();
            Solver.NumericFactorization();
            //std::cout << epetra_reduced_rhs.GlobalLength() << std::endl;
            //std::cout << epetra_rk_stage_i.GlobalLength() << std::endl;
            //std::cout << "Rows :" << epetra_reduced_lhs->NumGlobalRows() << "Cols :" << epetra_reduced_lhs->NumGlobalCols() <<std::endl;
            Solver.Solve();
            epetra_to_dealii(epetra_rk_stage_i,dealii_reduced_rhs);
            this->rk_stage[i] = dealii_reduced_rhs;
            
        }
}

this->modified_time_step = dt;
for (int i = 0; i < n_rk_stages; ++i){
    // Be careful with the pseudotimestep as not sure about that block
    Epetra_Vector epetra_rk_stage_i(Epetra_DataAccess::View, epetra_system_matrix.RowMap(), this->rk_stage[i].begin());
    Epetra_Vector epetra_reduced_stage_i(epetra_system_matrix.RangeMap());
    epetra_test_basis->Multiply(false, epetra_rk_stage_i, epetra_reduced_stage_i);
    dealii::LinearAlgebra::distributed::Vector<double> dealii_rk_stage_i;
    epetra_to_dealii(epetra_reduced_stage_i,dealii_rk_stage_i);
    if (pseudotime){
        const double CFL = this->butcher_tableau->get_b(i) * dt;
        this->dg->time_scale_solution_update(dealii_rk_stage_i, CFL);
        this->solution_update.add(1.0, dealii_rk_stage_i); 
    } else {
       
        this->solution_update.add(dt* this->butcher_tableau->get_b(i),dealii_rk_stage_i); 
    }
}
epetra_test_basis.reset();
epetra_reduced_lhs.reset();
this->dg->solution = this->solution_update; // u_0 + W*u_np1 = u_0 + W*u_n + dt* sum(W * k_i * b_i)

++(this->current_iteration);
this->current_time += dt;

}

template <int dim, typename real, int n_rk_stages, typename MeshType>
void PODGalerkinRKODESolver<dim, real, n_rk_stages, MeshType>::allocate_ode_system()
{
    this->pcout << "Allocating ODE system..." << std::endl;
    // Runge-Kutta Allocation
    this->solution_update.reinit(this->dg->right_hand_side);
    if(this->all_parameters->use_inverse_mass_on_the_fly == false) {
        this->pcout << " evaluating inverse mass matrix..." << std::flush;
        this->dg->evaluate_mass_matrices(true); // creates and stores global inverse mass matrix

    }

    this->rk_stage.resize(n_rk_stages);
    for (int i=0; i<n_rk_stages; ++i) {
        this->rk_stage[i].reinit(this->dg->solution);
    }

    this->butcher_tableau->set_tableau();
    
    this->butcher_tableau_aii_is_zero.resize(n_rk_stages);
    std::fill(this->butcher_tableau_aii_is_zero.begin(),
              this->butcher_tableau_aii_is_zero.end(),
              false); 
    for (int i=0; i<n_rk_stages; ++i) {
        if (this->butcher_tableau->get_a(i,i)==0.0)     this->butcher_tableau_aii_is_zero[i] = true;
    }
    // ROM Allocation
    /*Projection of initial conditions on reduced-order subspace, refer to Equation 19 in:
    Washabaugh, K. M., Zahr, M. J., & Farhat, C. (2016).
    On the use of discrete nonlinear reduced-order models for the prediction of steady-state flows past parametrically deformed complex geometries.
    In 54th AIAA Aerospace Sciences Meeting (p. 1814).
    */

    dealii::LinearAlgebra::distributed::Vector<double> reference_solution(this->dg->solution);
    reference_solution.import(pod->getReferenceState(), dealii::VectorOperation::values::insert);

    dealii::LinearAlgebra::distributed::Vector<double> initial_condition(this->dg->solution);
    initial_condition -= reference_solution;

    const Epetra_CrsMatrix epetra_pod_basis = pod->getPODBasis()->trilinos_matrix();
    Epetra_Vector epetra_reduced_solution(epetra_pod_basis.DomainMap());
    Epetra_Vector epetra_initial_condition(Epetra_DataAccess::Copy, epetra_pod_basis.RangeMap(), initial_condition.begin());

    epetra_pod_basis.Multiply(true, epetra_initial_condition, epetra_reduced_solution);

    Epetra_Vector epetra_projection_tmp(epetra_pod_basis.RangeMap());
    epetra_pod_basis.Multiply(false, epetra_reduced_solution, epetra_projection_tmp);

    Epetra_Vector epetra_solution(Epetra_DataAccess::View, epetra_pod_basis.RangeMap(), this->dg->solution.begin());

    epetra_solution = epetra_projection_tmp;
    this->dg->solution += reference_solution; // This here is the issue
}


template <int dim, typename real, int n_rk_stages, typename MeshType>
std::shared_ptr<Epetra_CrsMatrix> PODGalerkinRKODESolver<dim,real,n_rk_stages,MeshType>::generate_test_basis(const Epetra_CrsMatrix &/*system_matrix*/, const Epetra_CrsMatrix &pod_basis)
{
    return std::make_shared<Epetra_CrsMatrix>(pod_basis);
}

template <int dim, typename real, int n_rk_stages, typename MeshType>
std::shared_ptr<Epetra_CrsMatrix> PODGalerkinRKODESolver<dim,real,n_rk_stages,MeshType>::generate_reduced_lhs(const Epetra_CrsMatrix system_matrix, Epetra_CrsMatrix test_basis)
{   
    //std::ofstream system_file("system.txt");
    //epetra_reduced_lhs_tmp.Print(std::cout);
    //system_matrix.Print(system_file);
    Epetra_CrsMatrix epetra_reduced_lhs(Epetra_DataAccess::View, test_basis.DomainMap(), test_basis.NumGlobalCols());
    Epetra_CrsMatrix epetra_reduced_lhs_tmp(Epetra_DataAccess::View, system_matrix.DomainMap(), system_matrix.NumGlobalCols());
    EpetraExt::MatrixMatrix::Multiply(system_matrix, false, test_basis, false, epetra_reduced_lhs_tmp); // Memory leak
    //epetra_reduced_lhs_tmp.Print(std::cout);
    EpetraExt::MatrixMatrix::Multiply(test_basis, true, epetra_reduced_lhs_tmp, false, epetra_reduced_lhs); // Memory leak
    //epetra_reduced_lhs.Print(system_file);
    return std::make_shared<Epetra_CrsMatrix>(epetra_reduced_lhs);
}

template <int dim, typename real, int n_rk_stages, typename MeshType>
void PODGalerkinRKODESolver<dim,real,n_rk_stages,MeshType>::epetra_to_dealii(Epetra_Vector &epetra_vector, dealii::LinearAlgebra::distributed::Vector<double> &dealii_vector){
    const Epetra_BlockMap &epetra_map = epetra_vector.Map();

    dealii_vector.reinit(this->dg->solution,MPI_COMM_WORLD);
    for(int i = 0; i < epetra_map.NumMyElements();++i){
        int global_idx = epetra_map.GID(i);
        if(dealii_vector.in_local_range(global_idx)){
            dealii_vector[global_idx] = epetra_vector[i];
        }
    }
    dealii_vector.compress(dealii::VectorOperation::insert);

}
template class PODGalerkinRKODESolver<PHILIP_DIM, double,1, dealii::Triangulation<PHILIP_DIM> >;
template class PODGalerkinRKODESolver<PHILIP_DIM, double,2, dealii::Triangulation<PHILIP_DIM> >;
template class PODGalerkinRKODESolver<PHILIP_DIM, double,3, dealii::Triangulation<PHILIP_DIM> >;
template class PODGalerkinRKODESolver<PHILIP_DIM, double,4, dealii::Triangulation<PHILIP_DIM> >;
template class PODGalerkinRKODESolver<PHILIP_DIM, double,1, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
template class PODGalerkinRKODESolver<PHILIP_DIM, double,2, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
template class PODGalerkinRKODESolver<PHILIP_DIM, double,3, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
template class PODGalerkinRKODESolver<PHILIP_DIM, double,4, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
#if PHILIP_DIM != 1
    template class PODGalerkinRKODESolver<PHILIP_DIM, double,1, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class PODGalerkinRKODESolver<PHILIP_DIM, double,2, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class PODGalerkinRKODESolver<PHILIP_DIM, double,3, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class PODGalerkinRKODESolver<PHILIP_DIM, double,4, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
#endif
} // ODESolver name space
} // PHiLiP name space