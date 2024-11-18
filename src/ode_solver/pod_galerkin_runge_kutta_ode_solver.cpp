#include "pod_galerkin_runge_kutta_ode_solver.h"

#include <Amesos_Lapack.h>
#include <Epetra_LinearProblem.h>
#include <Epetra_Vector.h>
#include <EpetraExt_MatrixMatrix.h>
#include "Amesos_BaseSolver.h"

namespace PHiLiP::ODE {
template <int dim, typename real, int n_rk_stages, typename MeshType>
PODGalerkinRungeKuttaODESolver<dim,real,n_rk_stages,MeshType>::PODGalerkinRungeKuttaODESolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input,
            std::shared_ptr<RKTableauBase<dim,real,MeshType>> rk_tableau_input,
            std::shared_ptr<EmptyRRKBase<dim,real,MeshType>> RRK_object_input,
            std::shared_ptr<ProperOrthogonalDecomposition::PODBase<dim>> pod) 
            : RungeKuttaBase<dim,real,n_rk_stages,MeshType>(dg_input, RRK_object_input, pod)
            , butcher_tableau(rk_tableau_input)
            , epetra_pod_basis(pod->getPODBasis()->trilinos_matrix())
            , epetra_system_matrix(Epetra_DataAccess::View, epetra_pod_basis.RowMap(), epetra_pod_basis.NumGlobalRows())
            , epetra_test_basis(nullptr)
            , epetra_reduced_lhs(nullptr)
{}

template <int dim, typename real, int n_rk_stages, typename MeshType>
void PODGalerkinRungeKuttaODESolver<dim,real,n_rk_stages,MeshType>::calculate_stage_solution(int istage, real dt, const bool /*pseudotime*/)
{
    this->rk_stage[istage] = 0.0;
    this->reduced_rk_stage[istage] = 0.0;
    for(int j = 0; j < istage; ++j){
        if(this->butcher_tableau->get_a(istage,j) != 0){
            dealii::LinearAlgebra::distributed::Vector<double> dealii_rk_stage_j;
            multiply(*epetra_test_basis,this->reduced_rk_stage[j],dealii_rk_stage_j,this->dg->solution,false);
            this->rk_stage[istage].add(this->butcher_tableau->get_a(istage,j),dealii_rk_stage_j);
        }
    } //sum(a_ij*V*k_j), explicit part
    this->rk_stage[istage]*=dt;
    //dt * sum(a_ij * k_j)
    this->rk_stage[istage].add(1.0,this->solution_update);
    if (!this->butcher_tableau_aii_is_zero[istage]){
        // Implicit, looks fine on testing
        this->solver.solve(dt*this->butcher_tableau->get_a(istage,istage), this->rk_stage[istage]);
        this->rk_stage[istage] = this->solver.current_solution_estimate;
    }
    this->dg->solution = this->rk_stage[istage];
    int rank = dealii::Utilities::MPI::this_mpi_process(this->dg->solution.get_mpi_communicator());
    std::ofstream file("solution_" + std::to_string(rank)+ ".txt");
    print_dealii(file,this->solution_update);
    this->dg->solution /= 1;
}

template <int dim, typename real, int n_rk_stages, typename MeshType>
void PODGalerkinRungeKuttaODESolver<dim,real,n_rk_stages,MeshType>::calculate_stage_derivative(int istage, real dt)
{
    this->dg->set_current_time(this->current_time + this->butcher_tableau->get_c(istage)*dt);
    if(this->all_parameters->reduced_order_param.entropy_varibles_in_snapshots){
        dealii::TrilinosWrappers::SparseMatrix pod_basis;
        pod_basis.reinit(epetra_pod_basis);
        this->dg->calculate_global_entropy();
        this->dg->calculate_ROM_projected_entropy(pod_basis);
    }
    this->dg->assemble_residual(); //RHS : du/dt = RHS = F(u_n + dt* sum(a_ij*V*k_j) + dt * a_ii * u^(istage)))
    if(this->all_parameters->use_inverse_mass_on_the_fly){
        assert(1 == 0 && "Not Implemented: use_inverse_mass_on_the_fly=true && ode_solver_type=pod_galerkin_rk_solver\n Please set use_inverse_mass_on_the_fly=false and try again");
    } else{
        // Creating Reduced RHS
        dealii::LinearAlgebra::distributed::Vector<double> dealii_reduced_stage_i;

        Epetra_Vector epetra_rhs(Epetra_DataAccess::Copy, epetra_test_basis->RowMap(), this->dg->right_hand_side.begin()); // Flip to range map?
        int rank = dealii::Utilities::MPI::this_mpi_process(this->dg->solution.get_mpi_communicator());
        std::ofstream dealii_rhs("rhs_dealii_"+ std::to_string(rank)+ ".txt");
        dealii::LinearAlgebra::distributed::Vector<double> rhs = this->dg->right_hand_side;
        print_dealii(dealii_rhs,rhs);
        std::ofstream rhs_file("rhs_file_"+ std::to_string(rank)+ ".txt");
        epetra_rhs.Print(rhs_file);
        Epetra_Vector epetra_reduced_rhs(epetra_test_basis->DomainMap());
        epetra_test_basis->Multiply(true,epetra_rhs,epetra_reduced_rhs);
        std::ofstream reduced_rhs_file("reduced_rhs_file_"+ std::to_string(rank)+ ".txt");
        epetra_reduced_rhs.Print(reduced_rhs_file);
        // Creating Linear Problem to find stage
        Epetra_Vector epetra_rk_stage_i(epetra_reduced_lhs->DomainMap()); // Ensure this is correct as well, since LHS is not transpose might need to be rangeMap
        Epetra_LinearProblem linearProblem(epetra_reduced_lhs.get(), &epetra_rk_stage_i, &epetra_reduced_rhs);
        Amesos_Lapack Solver(linearProblem);
        Teuchos::ParameterList List;
        Solver.SetParameters(List); //Deprecated in future update, change?
        Solver.SymbolicFactorization();
        Solver.NumericFactorization();
        Solver.Solve();
        epetra_to_dealii(epetra_rk_stage_i,dealii_reduced_stage_i, this->reduced_rk_stage[istage]);
        this->reduced_rk_stage[istage] = dealii_reduced_stage_i;
    }
    int rank = dealii::Utilities::MPI::this_mpi_process(this->dg->solution.get_mpi_communicator());
    std::ofstream reduced_file("reduced_file_"+ std::to_string(rank)+ "_stage_"+std::to_string(istage)+".txt");
    print_dealii(reduced_file,this->reduced_rk_stage[istage]);
}

template <int dim, typename real, int n_rk_stages, typename MeshType>
void PODGalerkinRungeKuttaODESolver<dim,real,n_rk_stages,MeshType>::sum_stages(real dt, const bool /*pseudotime*/)
{
    dealii::LinearAlgebra::distributed::Vector<double> reduced_sum;
    reduced_sum = this->reduced_rk_stage[0];
    reduced_sum *= dt* this->butcher_tableau->get_b(0);
    for (int istage = 1; istage < n_rk_stages; ++istage){
        reduced_sum.add(dt* this->butcher_tableau->get_b(istage),this->reduced_rk_stage[istage]);
    }
    int rank = dealii::Utilities::MPI::this_mpi_process(this->dg->solution.get_mpi_communicator());
    std::ofstream sum_file("sum_file_"+std::to_string(rank)+".txt");
    print_dealii(sum_file,reduced_sum);
    // Convert Reduced order step to Full order step
    dealii::LinearAlgebra::distributed::Vector<double> dealii_update;
    multiply(*epetra_test_basis,reduced_sum,dealii_update,this->dg->solution,false);
    std::ofstream update_file("update_file_"+std::to_string(rank)+".txt");
    print_dealii(update_file,dealii_update);
    this->solution_update.add(1.0,dealii_update);
    std::ofstream new_solution_file("new_solution_file_"+std::to_string(rank)+".txt");
    print_dealii(new_solution_file,this->solution_update);
    
}


template <int dim, typename real, int n_rk_stages, typename MeshType>
void PODGalerkinRungeKuttaODESolver<dim,real,n_rk_stages,MeshType>::apply_limiter()
{
    // Empty Function
}
template <int dim, typename real, int n_rk_stages, typename MeshType>
real PODGalerkinRungeKuttaODESolver<dim,real,n_rk_stages,MeshType>::adjust_time_step(real dt)
{
    this->modified_time_step = dt;
    return dt;
}
template <int dim, typename real, int n_rk_stages, typename MeshType>
void PODGalerkinRungeKuttaODESolver<dim,real,n_rk_stages,MeshType>::allocate_runge_kutta_system()
{

    this->butcher_tableau->set_tableau();
    
    this->butcher_tableau_aii_is_zero.resize(n_rk_stages);
    std::fill(this->butcher_tableau_aii_is_zero.begin(),
              this->butcher_tableau_aii_is_zero.end(),
              false); 
    for (int istage=0; istage<n_rk_stages; ++istage) {
        if (this->butcher_tableau->get_a(istage,istage)==0.0)     this->butcher_tableau_aii_is_zero[istage] = true;
    
    }
    // Convert dg->solution Map to an Epetra_Map Unsure if this is needed
    std::vector<int> global_indicies;
    for(auto idx : this->dg->solution.locally_owned_elements()){
        global_indicies.push_back(static_cast<int>(idx));
    }
    Epetra_MpiComm epetra_comm(this->mpi_communicator);
    // Creating block here for now to auto delete this large matrix
        int solution_size = this->dg->solution.size();
        Epetra_Map solution_map(solution_size,global_indicies.size(),global_indicies.data(),0,epetra_comm);
        if(!solution_map.SameAs(epetra_pod_basis.RowMap())) {
            Epetra_CrsMatrix old_pod_basis = epetra_pod_basis;

            Epetra_Import basis_importer(solution_map, old_pod_basis.RowMap());

            MPI_Barrier(MPI_COMM_WORLD);
            epetra_pod_basis = Epetra_CrsMatrix(old_pod_basis, basis_importer);

        }
    //epetra_pod_basis.FillComplete();
    Epetra_Map reduced_map = epetra_pod_basis.DomainMap();
    PrintMapInfo(reduced_map);
    // Setting up Mass and Test Matrix
    Epetra_CrsMatrix old_epetra_system_matrix = this->dg->global_mass_matrix.trilinos_matrix();
    // Giving the system matrix the same map as pod matrix
    const Epetra_Map pod_map = epetra_pod_basis.RowMap();
    //PrintMapInfo(pod_map);
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
    // Evaluating Mass Matrix (after calling for mass matrix?)
    this->solution_update.reinit(this->dg->right_hand_side);
    if(this->all_parameters->use_inverse_mass_on_the_fly == false) {
        this->pcout << " evaluating inverse mass matrix..." << std::flush;
        this->dg->evaluate_mass_matrices(true); // creates and stores global inverse mass matrix
    }
    // parallelizing reduced RK Stage
    reduced_index = dealii::IndexSet(reduced_map);
    solution_index = this->dg->solution.locally_owned_elements();
    this->reduced_rk_stage.resize(n_rk_stages);
    for (int istage=0; istage<n_rk_stages; ++istage){
        this->reduced_rk_stage[istage].reinit(reduced_index, this->mpi_communicator); // Add IndexSet
    }
if(this->all_parameters->reduced_order_param.entropy_varibles_in_snapshots) {
        dealii::TrilinosWrappers::SparseMatrix pod_basis;
        pod_basis.reinit(epetra_pod_basis);
        int rank = epetra_pod_basis.Comm().MyPID();
        std::ofstream sum_file("Epetra_ODE_"+std::to_string(rank)+".txt");
        epetra_pod_basis.Print(sum_file);
        this->dg->calculate_projection_matrix(pod_basis);
    }

}

template <int dim, typename real, int n_rk_stages, typename MeshType>
std::shared_ptr<Epetra_CrsMatrix> PODGalerkinRungeKuttaODESolver<dim,real,n_rk_stages,MeshType>::generate_test_basis(const Epetra_CrsMatrix &/*system_matrix*/, const Epetra_CrsMatrix &pod_basis)
{
    return std::make_shared<Epetra_CrsMatrix>(pod_basis);
}

template <int dim, typename real, int n_rk_stages, typename MeshType>
std::shared_ptr<Epetra_CrsMatrix> PODGalerkinRungeKuttaODESolver<dim,real,n_rk_stages,MeshType>::generate_reduced_lhs(const Epetra_CrsMatrix &system_matrix, const Epetra_CrsMatrix &test_basis)
{   
    if (test_basis.RowMap().SameAs(system_matrix.RowMap()) && test_basis.NumGlobalRows() == system_matrix.NumGlobalRows()){
        Epetra_CrsMatrix epetra_reduced_lhs(Epetra_DataAccess::Copy, test_basis.DomainMap(), test_basis.NumGlobalCols());
        Epetra_CrsMatrix epetra_reduced_lhs_tmp(Epetra_DataAccess::Copy, system_matrix.RowMap(), test_basis.NumGlobalCols());
        //PrintMapInfo(epetra_reduced_lhs_tmp.RowMap());
        MPI_Barrier(MPI_COMM_WORLD);
        std::cout << "First Matrix Multiply" << std::endl;
        if (EpetraExt::MatrixMatrix::Multiply(system_matrix, false, test_basis, false, epetra_reduced_lhs_tmp) != 0){
            std::cerr << "Error in first Matrix Multiplication" << std::endl;
            return nullptr;
        };
        std::cout << "Second Matrix Multiply" << std::endl;


        //PrintMapInfo(test_basis.DomainMap());
        if (EpetraExt::MatrixMatrix::Multiply(test_basis, true, epetra_reduced_lhs_tmp, false, epetra_reduced_lhs) != 0){
            std::cerr << "Error in second Matrix Multiplication" << std::endl;
            return nullptr;
        };
        std::cout << "Returning lhs matrix" << std::endl;
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
int PODGalerkinRungeKuttaODESolver<dim,real,n_rk_stages,MeshType>::multiply(Epetra_CrsMatrix &epetra_matrix,
                                                                    dealii::LinearAlgebra::distributed::Vector<double> &input_dealii_vector,
                                                                    dealii::LinearAlgebra::distributed::Vector<double> &output_dealii_vector,
                                                                    dealii::LinearAlgebra::distributed::Vector<double> &index_vector,
                                                                    const bool transpose // Transpose needs to be used with care of maps
                                                                    )
{
    Epetra_Vector epetra_input(Epetra_DataAccess::View, epetra_matrix.DomainMap(), input_dealii_vector.begin());
    Epetra_Vector epetra_output(epetra_matrix.RangeMap());
    if(epetra_matrix.RangeMap().SameAs(epetra_output.Map()) && epetra_matrix.DomainMap().SameAs(epetra_input.Map())){
        epetra_matrix.Multiply(transpose, epetra_input, epetra_output);
        epetra_to_dealii(epetra_output,output_dealii_vector,index_vector);
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
void PODGalerkinRungeKuttaODESolver<dim,real,n_rk_stages,MeshType>::epetra_to_dealii(Epetra_Vector &epetra_vector,
                                                                             dealii::LinearAlgebra::distributed::Vector<double> &dealii_vector,
                                                                             dealii::LinearAlgebra::distributed::Vector<double> &index_vector)
{
    const Epetra_BlockMap &epetra_map = epetra_vector.Map();
    dealii_vector = index_vector;
    for(int i = 0; i < epetra_map.NumMyElements();++i){
        int global_idx = epetra_map.GID(i);
        if(dealii_vector.in_local_range(global_idx)){
            dealii_vector[global_idx] = epetra_vector[i];
        }
    }
    dealii_vector.update_ghost_values();

}
template <int dim, typename real, int n_rk_stages, typename MeshType>
void PODGalerkinRungeKuttaODESolver<dim,real,n_rk_stages,MeshType>::PrintMapInfo(const Epetra_Map &map) {
    std::cout << "Map Information:" << std::endl;
    std::cout << "  Number of Global Elements: " << map.NumGlobalElements() << std::endl;
    std::cout << "  Number of Local Elements: " << map.NumMyElements() << std::endl;
    std::cout << "  Min Global ID: " << map.MinAllGID() << std::endl;
    std::cout << "  Max Global ID: " << map.MaxAllGID() << std::endl;
    std::cout << "  Index Base: " << map.IndexBase() << std::endl;
    std::cout << "  Number of Processes: " << map.Comm().NumProc() << std::endl;
    std::cout << "  Process Rank: " << map.Comm().MyPID() << std::endl;
}
template <int dim, typename real, int n_rk_stages, typename MeshType>
void PODGalerkinRungeKuttaODESolver<dim,real,n_rk_stages,MeshType>::print_dealii(
    std::ofstream &file,
    dealii::LinearAlgebra::distributed::Vector<double> &vec){
    for(unsigned int i = 0; i < vec.size(); i++){
        if(vec.in_local_range(i)){
            file << vec[i] << '\n';
        }
    }
    file.close();
}

template class PODGalerkinRungeKuttaODESolver<PHILIP_DIM, double,1, dealii::Triangulation<PHILIP_DIM> >;
template class PODGalerkinRungeKuttaODESolver<PHILIP_DIM, double,2, dealii::Triangulation<PHILIP_DIM> >;
template class PODGalerkinRungeKuttaODESolver<PHILIP_DIM, double,3, dealii::Triangulation<PHILIP_DIM> >;
template class PODGalerkinRungeKuttaODESolver<PHILIP_DIM, double,4, dealii::Triangulation<PHILIP_DIM> >;
template class PODGalerkinRungeKuttaODESolver<PHILIP_DIM, double,1, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
template class PODGalerkinRungeKuttaODESolver<PHILIP_DIM, double,2, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
template class PODGalerkinRungeKuttaODESolver<PHILIP_DIM, double,3, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
template class PODGalerkinRungeKuttaODESolver<PHILIP_DIM, double,4, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
#if PHILIP_DIM != 1
    template class PODGalerkinRungeKuttaODESolver<PHILIP_DIM, double,1, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class PODGalerkinRungeKuttaODESolver<PHILIP_DIM, double,2, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class PODGalerkinRungeKuttaODESolver<PHILIP_DIM, double,3, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class PODGalerkinRungeKuttaODESolver<PHILIP_DIM, double,4, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
#endif
} // PHiLiP::ODE namespace