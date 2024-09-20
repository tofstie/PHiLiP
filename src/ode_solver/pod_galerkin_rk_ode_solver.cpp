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
                                                                          : ODESolverBase<dim,real,MeshType>(dg_input, pod)
                                                                          , butcher_tableau(rk_tableau_input)
                                                                          , solver(dg_input)
                                                                          , epetra_pod_basis(pod->getPODBasis()->trilinos_matrix())
                                                                          , epetra_system_matrix(Epetra_DataAccess::View, epetra_pod_basis.RowMap(), epetra_pod_basis.NumGlobalRows())
                                                                          , epetra_test_basis(nullptr)
                                                                          , epetra_reduced_lhs(nullptr)
{}                                          

template <int dim, typename real, int n_rk_stages, typename MeshType>
void PODGalerkinRKODESolver<dim, real, n_rk_stages, MeshType>::step_in_time (real dt, const bool pseudotime)
{
    this->original_time_step = dt;
    this->solution_update = this->dg->solution;

    const dealii::IndexSet &solution_index = this->dg->solution.locally_owned_elements();
    const dealii::IndexSet &reduced_index = this->reduced_rk_stage[0].locally_owned_elements();
    /*
    const Epetra_Comm &epetra_comm_reduced = epetra_pod_basis.Comm();

    int reduced_size = pod->getPODBasis()->n();
    int numMyElements = reduced_size / epetra_comm_reduced.NumProc();
    if (this->mpi_rank < reduced_size % epetra_comm_reduced.NumProc()) {
        numMyElements += 1;
    }
    Epetra_Map reduced_map(reduced_size,numMyElements,0,epetra_comm_reduced);
    */
    //Epetra_Map &reduced_map = epetra_test_basis->DomainMap();
    //Printing some matrices for debugging 

    std::ofstream mass_file("mass_file" + std::to_string(this->mpi_rank) + ".txt");
    epetra_reduced_lhs->Print(mass_file);

    for (int i = 0; i < n_rk_stages; ++i) {
        this->rk_stage[i] = 0.0;
        this->reduced_rk_stage[i] = 0.0;
        for(int j = 0; j < i; ++j){
            if(this->butcher_tableau->get_a(i,j) != 0){
                dealii::LinearAlgebra::distributed::Vector<double> dealii_rk_stage_j;
                Multiply(*epetra_test_basis,this->reduced_rk_stage[j],dealii_rk_stage_j,solution_index,false);
                this->rk_stage[i].add(this->butcher_tableau->get_a(i,j),dealii_rk_stage_j);
            }
        } //sum(a_ij*V*k_j), explicit part
        if(pseudotime) {
            const double CFL = dt;
            this->dg->time_scale_solution_update(rk_stage[i], CFL);
        }else {
            this->rk_stage[i]*=dt; 
        }//dt * sum(a_ij * k_j)
        this->rk_stage[i].add(1.0,this->solution_update);
        if (!this->butcher_tableau_aii_is_zero[i]){
            // Implicit, not sure on JFNK works for reduced order or not, assuming it does for now
            std::cout << "Implicit, abort" << std::endl;
            solver.solve(dt*this->butcher_tableau->get_a(i,i), this->rk_stage[i]);
            this->rk_stage[i] = solver.current_solution_estimate;
        }
        this->dg->solution = this->rk_stage[i];

        // Not including limiter yet as not sure of implications on ROM
        if (this->limiter) {
            this->limiter->limit(this->dg->solution,
                this->dg->dof_handler,
                this->dg->fe_collection,
                this->dg->volume_quadrature_collection,
                this->dg->high_order_grid->fe_system.tensor_degree(),
                this->dg->max_degree,
                this->dg->oneD_fe_collection_1state,
                this->dg->oneD_quadrature_collection);
        }
        //set the DG current time for unsteady source terms
        this->dg->set_current_time(this->current_time + this->butcher_tableau->get_c(i)*dt);
        //solve the system's right hand side
        if(this->all_parameters->reduced_order_param.entropy_varibles_in_snapshots){
            dealii::TrilinosWrappers::SparseMatrix pod_basis;
            pod_basis.reinit(epetra_pod_basis);
            this->dg->calculate_global_entropy();
            this->dg->calculate_ROM_projected_entropy(pod_basis);
        }
        this->dg->assemble_residual(); //RHS : du/dt = RHS = F(u_n + dt* sum(a_ij*V*k_j) + dt * a_ii * u^(i)))
        std::ofstream RHS_file("RHS" + std::to_string(i) + "Processor " + std::to_string(this->mpi_rank) + ".txt");
        this->dg->right_hand_side.print(RHS_file);
        if(this->all_parameters->use_inverse_mass_on_the_fly){
            //this->dg->apply_inverse_global_mass_matrix(this->dg->right_hand_side, this->rk_stage[i]); //rk_stage[i] = IMM*RHS = F(u_n + dt*sum(a_ij*k_j))
            // Not implmenting this yet as requires editting DG base
            assert(1 == 0 && "Not Implemented: use_inverse_mass_on_the_fly=true && ode_solver_type=pod_galerkin_rk_solver");
        } else{
            // Creating Reduced RHS
            dealii::LinearAlgebra::distributed::Vector<double> dealii_reduced_stage_i;
            Epetra_Vector eptra_rhs(Epetra_DataAccess::View, epetra_test_basis->RowMap(), this->dg->right_hand_side.begin()); // Flip to range map?
            Epetra_Vector epetra_reduced_rhs(epetra_test_basis->DomainMap());
            std::ofstream epetra_rhs_file("eptra_rhs" + std::to_string(i) + "Processor " + std::to_string(this->mpi_rank) + ".txt");
            eptra_rhs.Print(epetra_rhs_file);
            epetra_test_basis->Multiply(true,eptra_rhs,epetra_reduced_rhs);
            std::ofstream reduced_file("epetra_reduced_rhs" + std::to_string(i) + "Processor " + std::to_string(this->mpi_rank) + ".txt");
            epetra_reduced_rhs.Print(reduced_file);
            // Creating Linear Problem to find stage
            Epetra_Vector epetra_rk_stage_i(epetra_reduced_lhs->DomainMap()); // Ensure this is correct as well, since LHS is not transpose might need to be rangeMap
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
            epetra_to_dealii(epetra_rk_stage_i,dealii_reduced_stage_i, reduced_index);
            std::ofstream system_file("epetra_rk_stage_i" + std::to_string(i) + "Processor" + std::to_string(this->mpi_rank) + ".txt");
            std::ofstream dealii_rk_stage_file("dealii_rk_stage_i" + std::to_string(i) + "Processor" + std::to_string(this->mpi_rank) + ".txt");
            epetra_rk_stage_i.Print(system_file);
            dealii_reduced_stage_i.print(dealii_rk_stage_file);
            this->reduced_rk_stage[i] = dealii_reduced_stage_i;
        }
    }

    //std::cout << "Cut" << std::endl;
    this->modified_time_step = dt;
    dealii::LinearAlgebra::distributed::Vector<double> reduced_sum;
    reduced_sum.reinit(this->reduced_rk_stage[0]);
    for (int i = 0; i < n_rk_stages; ++i){
        // Be careful with the pseudotimestep as not sure about that block
        if (pseudotime){
            // const double CFL = this->butcher_tableau->get_b(i) * dt;
            // this->dg->time_scale_solution_update(dealii_rk_stage_i, CFL); This line causes problems from the time_scale_solution_update ... Will need to do more costly operation inside this block
            reduced_sum.add(1.0, this->reduced_rk_stage[i]); 
        } else {
            reduced_sum.add(dt* this->butcher_tableau->get_b(i),this->reduced_rk_stage[i]); 
        }
    }
    // Convert Reduced order step to Full order step
    dealii::LinearAlgebra::distributed::Vector<double> dealii_update;
    Multiply(*epetra_test_basis,reduced_sum,dealii_update,solution_index,false);
    this->solution_update.add(1.0,dealii_update);

    this->dg->solution = this->solution_update; // u_0 + W*u_np1 = u_0 + W*u_n + dt* sum(W * k_i * b_i)
    if (this->limiter) {
        this->limiter->limit(this->dg->solution,
            this->dg->dof_handler,
            this->dg->fe_collection,
            this->dg->volume_quadrature_collection,
            this->dg->high_order_grid->fe_system.tensor_degree(),
            this->dg->max_degree,
            this->dg->oneD_fe_collection_1state,
            this->dg->oneD_quadrature_collection);
    }
    ++(this->current_iteration);
    //this->pcout << this->current_iteration << std::endl;
    this->current_time += dt;
}

template <int dim, typename real, int n_rk_stages, typename MeshType>
void PODGalerkinRKODESolver<dim, real, n_rk_stages, MeshType>::allocate_ode_system()
{
    this->pcout << "Allocating ODE system..." << std::endl;
    // Convert dg->solution Map to an Epetra_Map Unsure if this is needed
    std::vector<int> global_indicies;
    for(auto idx : this->dg->solution.locally_owned_elements()){
        global_indicies.push_back((int) idx);
    }
    int solution_size = this->dg->solution.size();

    Epetra_MpiComm epetra_comm(this->mpi_communicator);
    Epetra_Map solution_map(solution_size,global_indicies.size(),global_indicies.data(),0,epetra_comm);

    // Creating block here for now to auto delete this large matrix
    {
        Epetra_CrsMatrix old_pod_basis = epetra_pod_basis;
        Epetra_Import basis_importer(solution_map, old_pod_basis.RowMap());
        epetra_pod_basis = Epetra_CrsMatrix(old_pod_basis, basis_importer);
        epetra_pod_basis.FillComplete();
    }
    Epetra_Map reduced_map = epetra_pod_basis.DomainMap();
    // Setting up Mass and Test Matrix
    Epetra_CrsMatrix old_epetra_system_matrix = this->dg->global_mass_matrix.trilinos_matrix();
    // Giving the system matrix the same map as pod matrix
    const Epetra_Map& pod_map = epetra_pod_basis.RowMap();
    Epetra_Import importer(pod_map, old_epetra_system_matrix.RowMap());
    // Reordering 
    epetra_system_matrix = Epetra_CrsMatrix(old_epetra_system_matrix, importer, &pod_map, &pod_map);
    try {
        int glerror = epetra_system_matrix.FillComplete();
        if (glerror != 0){
            std::cerr << "Fill complete failed with error code " << std::to_string(glerror) << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Fill complete failed with error code " << e.what() << std::endl;
        throw;
    }
    this->pcout << "System Matrix Imported" << std::endl;

    // Generating test_basis and Reduced LHS
    epetra_test_basis = generate_test_basis(epetra_pod_basis, epetra_pod_basis); // These two lines will need to be updated for LSPG
    epetra_reduced_lhs = generate_reduced_lhs(epetra_system_matrix, *epetra_test_basis); // They need to be reinitialized every step for LSPG
    
    // Evaluating Mass Matrix
    this->solution_update.reinit(this->dg->right_hand_side);
    if(this->all_parameters->use_inverse_mass_on_the_fly == false) {
        this->pcout << " evaluating inverse mass matrix..." << std::flush;
        this->dg->evaluate_mass_matrices(true); // creates and stores global inverse mass matrix
    }




    // Parrallezing reduced RK Stage
    dealii::IndexSet reduced_index(reduced_map);

    this->reduced_rk_stage.resize(n_rk_stages);
    this->rk_stage.resize(n_rk_stages);
    for (int i=0; i<n_rk_stages; ++i){
        this->reduced_rk_stage[i].reinit(reduced_index, this->mpi_communicator); // Add IndexSet
        this->rk_stage[i].reinit(this->dg->solution);
    }

    // Setting up butcher_tableau
    this->butcher_tableau->set_tableau();
    
    this->butcher_tableau_aii_is_zero.resize(n_rk_stages);
    std::fill(this->butcher_tableau_aii_is_zero.begin(),
              this->butcher_tableau_aii_is_zero.end(),
              false); 
    for (int i=0; i<n_rk_stages; ++i) {
        if (this->butcher_tableau->get_a(i,i)==0.0)     this->butcher_tableau_aii_is_zero[i] = true;
    }
}   


template <int dim, typename real, int n_rk_stages, typename MeshType>
std::shared_ptr<Epetra_CrsMatrix> PODGalerkinRKODESolver<dim,real,n_rk_stages,MeshType>::generate_test_basis(const Epetra_CrsMatrix &/*system_matrix*/, const Epetra_CrsMatrix &pod_basis)
{
    return std::make_shared<Epetra_CrsMatrix>(pod_basis);
}

template <int dim, typename real, int n_rk_stages, typename MeshType>
std::shared_ptr<Epetra_CrsMatrix> PODGalerkinRKODESolver<dim,real,n_rk_stages,MeshType>::generate_reduced_lhs(const Epetra_CrsMatrix &system_matrix, const Epetra_CrsMatrix &test_basis)
{   
    //std::ofstream system_file("system.txt");
    //epetra_reduced_lhs_tmp.Print(std::cout);
    //system_matrix.Print(system_file);
    if (test_basis.RowMap().SameAs(system_matrix.RowMap()) && test_basis.NumGlobalRows() == system_matrix.NumGlobalRows()){
        Epetra_CrsMatrix epetra_reduced_lhs(Epetra_DataAccess::Copy, test_basis.DomainMap(), test_basis.NumGlobalCols()); // Consider Changing to copy
        Epetra_CrsMatrix epetra_reduced_lhs_tmp(Epetra_DataAccess::Copy, system_matrix.RowMap(), test_basis.NumGlobalCols());
        std::cout << "First Matrix Multiply" << std::endl;
        if (EpetraExt::MatrixMatrix::Multiply(system_matrix, false, test_basis, false, epetra_reduced_lhs_tmp) != 0){
            std::cerr << "Error in first Matrix Multiplication" << std::endl;
            return nullptr;
        }; // Memory leak
        
        //epetra_reduced_lhs_tmp.Print(std::cout);
        std::cout << "Second Matrix Multiply" << std::endl;
        if (EpetraExt::MatrixMatrix::Multiply(test_basis, true, epetra_reduced_lhs_tmp, false, epetra_reduced_lhs) != 0){
            std::cerr << "Error in second Matrix Multiplication" << std::endl;
            return nullptr;
        }; // Memory leak
        //epetra_reduced_lhs.Print(system_file);
        std::cout << "Returning" << std::endl;
        return std::make_shared<Epetra_CrsMatrix>(epetra_reduced_lhs);
    } else {
        if(!(test_basis.RowMap().SameAs(system_matrix.RowMap()))){
            std::cerr << "Error: Inconsistent maps" << std::endl;
        } else {
            std::cerr << "Error: Inconsistent row sizes" << std::endl 
            << "System: " << std::to_string(system_matrix.NumGlobalRows()) << std::endl 
            << "Test: " << std::to_string(test_basis.NumGlobalRows()) << std::endl;
        }
    }
    return nullptr;
}

template<int dim, typename real, int n_rk_stages, typename MeshType>
int PODGalerkinRKODESolver<dim,real,n_rk_stages,MeshType>::Multiply(Epetra_CrsMatrix &epetra_matrix,
                                                                    dealii::LinearAlgebra::distributed::Vector<double> &input_dealii_vector,
                                                                    dealii::LinearAlgebra::distributed::Vector<double> &output_dealii_vector,
                                                                    dealii::IndexSet index_set,
                                                                    bool transpose//Careful with tranpose as correct maps are not set up
                                                                    )
{
    Epetra_Vector epetra_input(Epetra_DataAccess::View, epetra_matrix.DomainMap(), input_dealii_vector.begin());
    Epetra_Vector epetra_output(epetra_matrix.RangeMap());
    if(epetra_matrix.RangeMap().SameAs(epetra_output.Map()) && epetra_matrix.DomainMap().SameAs(epetra_input.Map())){
        epetra_matrix.Multiply(transpose, epetra_input, epetra_output);
        epetra_to_dealii(epetra_output,output_dealii_vector,index_set);
        return 0;
    } else {
        if(!epetra_matrix.RangeMap().SameAs(epetra_output.Map())){
            std::cerr << "Output Map is not the same as Matrix Range Map" << std::endl;
        } else {
            std::cerr << "Input Map is not the same as the Matrix Domain Map" << std::endl;
        }
    }
    return -1;
}

template <int dim, typename real, int n_rk_stages, typename MeshType>
void PODGalerkinRKODESolver<dim,real,n_rk_stages,MeshType>::epetra_to_dealii(Epetra_Vector &epetra_vector, 
                                                                             dealii::LinearAlgebra::distributed::Vector<double> &dealii_vector,
                                                                             dealii::IndexSet index_set)
{
    const Epetra_BlockMap &epetra_map = epetra_vector.Map();
    /*
    std::ofstream dealii_to_epetra_map_file("dealii_map" + std::to_string(this->mpi_rank) + ".txt");
    std::ofstream epetra_to_dealii_map_file("epetra_map" + std::to_string(this->mpi_rank) + ".txt");
    dealii::LinearAlgebra::distributed::Vector<double> dealii_to_epetra_map(index_set, this->mpi_communicator);
    for(unsigned int i = 0; i < dealii_to_epetra_map.size(); i++){
        if(dealii_to_epetra_map.in_local_range(i)){
            dealii_to_epetra_map[i] = i;
        }
    }
    dealii_to_epetra_map.print(dealii_to_epetra_map_file);
    Epetra_Vector epetra_to_dealii_map(Epetra_DataAccess::Copy, epetra_map, dealii_to_epetra_map.begin());
    epetra_to_dealii_map.Print(epetra_to_dealii_map_file);
    */
    dealii_vector.reinit(index_set,this->mpi_communicator); // Need one for different size (reduced), one approach is to take in an IndexSet
    for(int i = 0; i < epetra_map.NumMyElements();++i){
        int epetra_global_idx = epetra_map.GID(i);
        int dealii_global_idx = epetra_global_idx;
        if(dealii_vector.in_local_range(dealii_global_idx)){
            dealii_vector[dealii_global_idx] = epetra_vector[epetra_global_idx];
        }
    }
    dealii_vector.compress(dealii::VectorOperation::insert);

}
/*
template <int dim, typename real, int n_rk_stages, typename MeshType>
void PODGalerkinRKODESolver<dim, real, n_rk_stages,MeshType>::dealii_to_epetra(dealii::LinearAlgebra::distributed::Vector<double> &dealii_vector,
                                                                               Epetra_Vector &epetra_vector)
{
    for(unsigned int i = 0; i < dealii_vector.size(); i++){
        if(dealii_vector.in_local_range(i)){
            epetra_vector.ReplaceGlobalValues(1,&dealii_vector[i],&i);
        }
    }
    return;
}
*/
template <int dim, typename real, int n_rk_stages, typename MeshType>
void PODGalerkinRKODESolver<dim, real, n_rk_stages, MeshType>::debug_Epetra(Epetra_CrsMatrix &epetra_matrix){
        this->pcout << "Number of rows: " << epetra_matrix.NumGlobalRows() << std::endl;
        this->pcout << "Number of cols: " << epetra_matrix.NumGlobalCols() << std::endl;
        this->pcout << "Number of non-zeros: " << epetra_matrix.NumGlobalNonzeros() << std::endl;
        this->pcout << "Row Map: " << epetra_matrix.RowMap() << std::endl;
        this->pcout << "Col Map: " << epetra_matrix.ColMap() << std::endl;
}

template <int dim, typename real, int n_rk_stages, typename MeshType>
void PODGalerkinRKODESolver<dim, real, n_rk_stages, MeshType>::head(Epetra_Vector &epetra_vector, int number){
    std::cout << "|\tLocal Index\t|\tValue\t|\tRank\t|" << std::endl;
    for(int i = 0; i < number; i++){
        std::cout <<"|\t" << std::to_string(i) <<"\t\t|" << std::scientific << std::to_string(epetra_vector[i]) << "|\t" << std::to_string(this->mpi_rank) << "\t|" << std::endl;
    }
    return;
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