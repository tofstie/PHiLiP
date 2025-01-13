#include "hyper_reduced_pod_galerkin_runge_kutta_ode_solver.h"
#include <EpetraExt_MatrixMatrix.h>
#include <Epetra_LinearProblem.h>
#include <Amesos_Lapack.h>
namespace PHiLiP::ODE {

template <int dim, typename real, int n_rk_stages, typename MeshType>
HyperReducedPODGalerkinRungeKuttaODESolver<dim, real, n_rk_stages, MeshType>::HyperReducedPODGalerkinRungeKuttaODESolver(
                                                                                std::shared_ptr< DGBase<dim, real, MeshType> > dg_input,
                                                                                std::shared_ptr<RKTableauBase<dim,real,MeshType>> rk_tableau_input,
                                                                                std::shared_ptr<EmptyRRKBase<dim,real,MeshType>> RRK_object_input, 
                                                                                std::shared_ptr<ProperOrthogonalDecomposition::PODBase<dim>> pod, 
                                                                                const Epetra_Vector weights)
                                                                                : RungeKuttaBase<dim,real,n_rk_stages,MeshType>(dg_input, RRK_object_input, pod)
                                                                                , pod(pod)
                                                                                , ECSW_weights(weights)
                                                                                , butcher_tableau(rk_tableau_input)
                                                                                , epetra_pod_basis(pod->getPODBasis()->trilinos_matrix())
                                                                                , epetra_test_basis(nullptr)
                                                                                , epetra_trial_basis(nullptr)
{}

template <int dim, typename real, int n_rk_stages, typename MeshType>
void HyperReducedPODGalerkinRungeKuttaODESolver<dim, real, n_rk_stages, MeshType>::calculate_stage_solution (int istage, real dt, const bool /*pseudotime*/) {
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
}

template <int dim, typename real, int n_rk_stages, typename MeshType>
void HyperReducedPODGalerkinRungeKuttaODESolver<dim, real, n_rk_stages, MeshType>::calculate_stage_derivative (int istage, real dt) {
    this->dg->set_current_time(this->current_time + this->butcher_tableau->get_c(istage)*dt);
    if(this->all_parameters->reduced_order_param.entropy_varibles_in_snapshots){
        dealii::TrilinosWrappers::SparseMatrix pod_basis;
        pod_basis.reinit(epetra_pod_basis);
        this->dg->calculate_global_entropy();
        this->dg->calculate_ROM_projected_entropy(pod_basis);
    }
    this->dg->assemble_residual(); //RHS : du/dt = RHS = F(u_n + dt* sum(a_ij*V*k_j) + dt * a_ii * u^(istage)))
    Epetra_Vector epetra_right_hand_side(Epetra_DataAccess::View, epetra_trial_basis->RowMap(), this->dg->right_hand_side.begin());
    std::shared_ptr<Epetra_Vector> hyper_reduced_rhs = generate_hyper_reduced_residual(epetra_right_hand_side, *epetra_trial_basis);
    hyper_reduced_rhs->Scale(-1.0);
    if(this->all_parameters->use_inverse_mass_on_the_fly){
        assert(1 == 0 && "Not Implemented: use_inverse_mass_on_the_fly=true && ode_solver_type=pod_galerkin_rk_solver\n Please set use_inverse_mass_on_the_fly=false and try again");
    } else{
        // Creating Reduced RHS
        dealii::LinearAlgebra::distributed::Vector<double> dealii_reduced_stage_i;

        Epetra_Vector epetra_rhs(*hyper_reduced_rhs); // Flip to range map?
        //int rank = dealii::Utilities::MPI::this_mpi_process(this->dg->solution.get_mpi_communicator());
        //std::ofstream dealii_rhs("rhs_dealii_"+ std::to_string(rank)+ ".txt");
        //print_dealii(dealii_rhs,rhs);
        //std::ofstream rhs_file("rhs_file_"+ std::to_string(rank)+ ".txt");
        //epetra_rhs.Print(rhs_file);
        Epetra_Vector epetra_reduced_rhs(epetra_test_basis->DomainMap());
        epetra_test_basis->Multiply(true,epetra_rhs,epetra_reduced_rhs);
        //std::ofstream reduced_rhs_file("reduced_rhs_file_"+ std::to_string(rank)+ ".txt");
        //epetra_reduced_rhs.Print(reduced_rhs_file);
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
    return;
}

template <int dim, typename real, int n_rk_stages, typename MeshType>
void HyperReducedPODGalerkinRungeKuttaODESolver<dim, real, n_rk_stages, MeshType>::sum_stages (real dt, const bool /*pseudotime*/) {
    dealii::LinearAlgebra::distributed::Vector<double> reduced_sum;
    reduced_sum.reinit(this->reduced_rk_stage[0]);
    for (int istage = 0; istage < n_rk_stages; ++istage){
        reduced_sum.add(dt* this->butcher_tableau->get_b(istage),this->reduced_rk_stage[istage]);
    }
    // Convert Reduced order step to Full order step
    dealii::LinearAlgebra::distributed::Vector<double> dealii_update;
    multiply(*epetra_test_basis,reduced_sum,dealii_update,this->dg->solution,false);
    this->solution_update.add(1.0,dealii_update);
}

template <int dim, typename real, int n_rk_stages, typename MeshType>
void HyperReducedPODGalerkinRungeKuttaODESolver<dim, real, n_rk_stages, MeshType>::allocate_runge_kutta_system(){

    this->butcher_tableau->set_tableau();

    this->butcher_tableau_aii_is_zero.resize(n_rk_stages);
    std::fill(this->butcher_tableau_aii_is_zero.begin(),
              this->butcher_tableau_aii_is_zero.end(),
              false);
    for (int istage=0; istage<n_rk_stages; ++istage) {
        if (this->butcher_tableau->get_a(istage,istage)==0.0)     this->butcher_tableau_aii_is_zero[istage] = true;

    }
    this->solution_update.reinit(this->dg->right_hand_side);
    // Not syncing maps for now
    Epetra_Vector epetra_reduced_solution(epetra_pod_basis.DomainMap());

    const Epetra_Map reduced_map = epetra_pod_basis.DomainMap();
    reduced_index = dealii::IndexSet(reduced_map);
    solution_index = this->dg->solution.locally_owned_elements();
    this->reduced_rk_stage.resize(n_rk_stages);
    for (int istage=0; istage<n_rk_stages; ++istage){
        this->reduced_rk_stage[istage].reinit(reduced_index, this->mpi_communicator); // Add IndexSet
    }
    Epetra_CrsMatrix epetra_mass_matrix(this->dg->global_mass_matrix.trilinos_matrix());

    epetra_test_basis = generate_test_basis(epetra_pod_basis, false);
    epetra_trial_basis = generate_test_basis(epetra_pod_basis, true);

    epetra_reduced_lhs = generate_reduced_lhs(epetra_mass_matrix,*epetra_test_basis,*epetra_trial_basis);

    // Store weights into DG (FIX FOR MULTICORE LATER)
    dealii::Vector<double> weights_dealii(ECSW_weights.GlobalLength());
    for(unsigned int i = 0; i < weights_dealii.size(); ++i) {
        weights_dealii[i] = ECSW_weights[i];
    }
    this->dg->reduced_mesh_weights = weights_dealii;
}

template <int dim, typename real, int n_rk_stages, typename MeshType>
void HyperReducedPODGalerkinRungeKuttaODESolver<dim, real, n_rk_stages, MeshType>::apply_limiter(){
    return;
}

template <int dim, typename real, int n_rk_stages, typename MeshType>
real HyperReducedPODGalerkinRungeKuttaODESolver<dim, real, n_rk_stages, MeshType>::adjust_time_step(real dt){
    return dt;
}

template <int dim, typename real, int n_rk_stages, typename MeshType>
std::shared_ptr<Epetra_CrsMatrix> HyperReducedPODGalerkinRungeKuttaODESolver<dim, real, n_rk_stages, MeshType>::generate_hyper_reduced_mass_matrix(
    const dealii::TrilinosWrappers::SparseMatrix& mass_matrix)
{
    Epetra_MpiComm epetra_comm(MPI_COMM_WORLD);
    Epetra_CrsMatrix epetra_mass_matrix = mass_matrix.trilinos_matrix();
    const unsigned int max_dofs_per_cell = this->dg->dof_handler.get_fe_collection().max_dofs_per_cell();
    std::vector<dealii::types::global_dof_index> current_dofs_indices(max_dofs_per_cell);
    /// GET ids
    std::set<unsigned int> rowIDs;
    for (const auto &cell : this->dg->dof_handler.active_cell_iterators()) {
        if (ECSW_weights[cell->active_cell_index()] != 0 ) {
            for (unsigned int quad_id : current_dofs_indices) {
                rowIDs.insert(quad_id);
            }
        }
    }
    const int hyper_reduced_size = rowIDs.size();
    Epetra_Map hyper_reduced_map(hyper_reduced_size,0,epetra_comm);
    Epetra_CrsMatrix hyper_reduced_basis(Epetra_DataAccess::Copy, hyper_reduced_map,hyper_reduced_size);

    return std::make_shared<Epetra_CrsMatrix>(hyper_reduced_basis);
}

template <int dim, typename real, int n_rk_stages, typename MeshType>
std::shared_ptr<Epetra_CrsMatrix> HyperReducedPODGalerkinRungeKuttaODESolver<dim, real, n_rk_stages, MeshType>::generate_test_basis(const Epetra_CrsMatrix &pod_basis,
                                                                                                                                        const bool trial_basis) {
    Epetra_MpiComm epetra_comm(MPI_COMM_WORLD);
    /*
    Epetra_Map column_map = pod_basis.DomainMap();
    int num_of_modes = column_map.NumGlobalElements();
    const unsigned int max_dofs_per_cell = this->dg->dof_handler.get_fe_collection().max_dofs_per_cell();
    std::vector<dealii::types::global_dof_index> current_dofs_indices(max_dofs_per_cell);
    /// GET ids
    std::set<unsigned int> rowIDs;
    for (const auto &cell : this->dg->dof_handler.active_cell_iterators()) {
        if (ECSW_weights[cell->active_cell_index()] != 0 ) {
            for (unsigned int quad_id : current_dofs_indices) {
                rowIDs.insert(quad_id);
            }
        }
    }
    const int hyper_reduced_size = rowIDs.size();
    Epetra_Map hyper_reduced_row_map(hyper_reduced_size,0,epetra_comm);
    /// Filter Test Basis using Row ids
    Epetra_CrsMatrix hyper_reduced_basis(Epetra_DataAccess::Copy, hyper_reduced_row_map,num_of_modes);
    int hyper_rowID = 0;
    for( int FOM_rowID : rowIDs ) {
        int num_entries = 0;
        double *global_row = new double [num_of_modes];
        int *indicies = new int [num_of_modes];
        pod_basis.ExtractGlobalRowCopy(FOM_rowID,num_of_modes,num_entries,global_row,indicies);
        hyper_reduced_basis.InsertGlobalValues(hyper_rowID,num_of_modes,global_row,indicies);
        hyper_rowID++;
    }
    hyper_reduced_basis.FillComplete(column_map,hyper_reduced_row_map);
    */
    Epetra_Map basis_rowmap = pod_basis.RowMap();
    Epetra_Map basis_domainmap = pod_basis.DomainMap();
    Epetra_CrsMatrix hyper_reduced_basis(Epetra_DataAccess::Copy, basis_rowmap,pod_basis.NumGlobalCols());
    const int N = pod_basis.NumGlobalRows();
    const int number_modes = pod_basis.NumGlobalCols();
    const unsigned int max_dofs_per_cell = this->dg->dof_handler.get_fe_collection().max_dofs_per_cell();
    std::vector<dealii::types::global_dof_index> current_dofs_indices(max_dofs_per_cell);
    std::vector<dealii::types::global_dof_index> neighbour_dofs_indices(max_dofs_per_cell);
    for (const auto &cell : this->dg->dof_handler.active_cell_iterators()) {
        if (ECSW_weights[cell->active_cell_index()] != 0 ) {
            const int fe_index_curr_cell = cell->active_fe_index();
            const dealii::FESystem<dim,dim> &current_fe_ref = this->dg->fe_collection[fe_index_curr_cell];
            const int n_dofs_curr_cell = current_fe_ref.n_dofs_per_cell();

            current_dofs_indices.resize(n_dofs_curr_cell);
            cell->get_dof_indices(current_dofs_indices);
    /*
            double *row = new double[pod_basis.NumGlobalCols()];
            int *global_indices = new int[pod_basis.NumGlobalCols()];
            int numE;
            int row_num = current_dofs_indices[0];
            pod_basis.ExtractGlobalRowCopy(row_num, pod_basis.NumGlobalCols(), numE, row, global_indices);
            int neighbour_dofs_curr_cell = 0;
            for (int i = 0; i < numE; i++){
                neighbour_dofs_curr_cell +=1;
                neighbour_dofs_indices.resize(neighbour_dofs_curr_cell);
                neighbour_dofs_indices[neighbour_dofs_curr_cell-1] = global_indices[i];
            }
            delete[] row;
            delete[] global_indices;
            */
            // Create L_e matrix and transposed L_e matrixfor current cell
            Epetra_Map LeRowMap(n_dofs_curr_cell, 0, epetra_comm);
            Epetra_CrsMatrix L_e(Epetra_DataAccess::Copy, LeRowMap, N);
            const double posOne = 1.0;

            for(int i = 0; i < n_dofs_curr_cell; i++){
                const int col = current_dofs_indices[i];
                L_e.InsertGlobalValues(i, 1, &posOne , &col);
            }
            L_e.FillComplete(basis_rowmap, LeRowMap);

            // Find contribution of element to the Jacobian
            Epetra_CrsMatrix V_L_e_T(Epetra_DataAccess::Copy, LeRowMap, number_modes);
            Epetra_CrsMatrix V_e_m(Epetra_DataAccess::Copy, basis_rowmap, number_modes);
            EpetraExt::MatrixMatrix::Multiply(L_e, false, pod_basis, false, V_L_e_T, true);
            EpetraExt::MatrixMatrix::Multiply(L_e, true, V_L_e_T, false, V_e_m, true);

            // Add the contribution of the element to the hyper-reduced Jacobian with scaling from the weights
            double scaling = 1.0;
            if (trial_basis) scaling = ECSW_weights[cell->active_cell_index()];
            EpetraExt::MatrixMatrix::Add(V_e_m, false, scaling, hyper_reduced_basis, 1.0);
        }
    }
    hyper_reduced_basis.FillComplete(basis_domainmap, basis_rowmap);
    std::ofstream file("HR_Pod_basis.txt");
    hyper_reduced_basis.Print(file);
    return std::make_shared<Epetra_CrsMatrix>(hyper_reduced_basis);
}

template <int dim, typename real, int n_rk_stages, typename MeshType>
std::shared_ptr<Epetra_Vector> HyperReducedPODGalerkinRungeKuttaODESolver<dim, real, n_rk_stages, MeshType>::generate_hyper_reduced_residual(
    Epetra_Vector epetra_right_hand_side,
    const Epetra_CrsMatrix &test_basis)
{
     /* Refer to Equation (10) in:
    https://onlinelibrary.wiley.com/doi/10.1002/nme.6603 (includes definitions of matrices used below such as L_e and L_e_PLUS)
    Create empty Hyper-reduced residual Epetra structure */
    Epetra_MpiComm epetra_comm(MPI_COMM_WORLD);
    Epetra_Map test_basis_colmap = test_basis.ColMap();
    Epetra_Vector hyper_reduced_residual(test_basis_colmap);
    int N = test_basis.NumGlobalRows();
    const unsigned int max_dofs_per_cell = this->dg->dof_handler.get_fe_collection().max_dofs_per_cell();
    std::vector<dealii::types::global_dof_index> current_dofs_indices(max_dofs_per_cell);

    // Loop through elements
    for (const auto &cell : this->dg->dof_handler.active_cell_iterators())
    {
        // Add the contributions of an element if the weight from the NNLS is non-zero
        if (ECSW_weights[cell->active_cell_index()] != 0){
            const int fe_index_curr_cell = cell->active_fe_index();
            const dealii::FESystem<dim,dim> &current_fe_ref = this->dg->fe_collection[fe_index_curr_cell];
            const int n_dofs_curr_cell = current_fe_ref.n_dofs_per_cell();

            current_dofs_indices.resize(n_dofs_curr_cell);
            cell->get_dof_indices(current_dofs_indices);

            // Create L_e matrix for current cell
            Epetra_Map LeRowMap(n_dofs_curr_cell, 0, epetra_comm);
            Epetra_Map LeColMap(N, 0, epetra_comm);
            Epetra_CrsMatrix L_e(Epetra_DataAccess::Copy, LeRowMap, N);
            double posOne = 1.0;

            for(int i = 0; i < n_dofs_curr_cell; i++){
                const int col = current_dofs_indices[i];
                L_e.InsertGlobalValues(i, 1, &posOne , &col);
            }
            L_e.FillComplete(LeColMap, LeRowMap);

            // Find contribution of the current element in the global dimensions
            Epetra_Vector local_r(LeRowMap);
            Epetra_Vector global_r_e(LeColMap);
            L_e.Multiply(false, epetra_right_hand_side, local_r);
            L_e.Multiply(true, local_r, global_r_e);

            // Find reduced representation of residual and scale by weight
            Epetra_Vector reduced_rhs_e(test_basis_colmap);
            test_basis.Multiply(true, global_r_e, reduced_rhs_e);
            reduced_rhs_e.Scale(ECSW_weights[cell->active_cell_index()]);
            double *reduced_rhs_array = new double[reduced_rhs_e.GlobalLength()];

            // Add to hyper-reduced representation of the residual
            reduced_rhs_e.ExtractCopy(reduced_rhs_array);
            for (int k = 0; k < reduced_rhs_e.GlobalLength(); ++k){
                hyper_reduced_residual.SumIntoGlobalValues(1, &reduced_rhs_array[k], &k);
            }
            delete[] reduced_rhs_array;
        }
    }
    return std::make_shared<Epetra_Vector>(hyper_reduced_residual);
}

template <int dim, typename real, int n_rk_stages, typename MeshType>
std::shared_ptr<Epetra_CrsMatrix> HyperReducedPODGalerkinRungeKuttaODESolver<dim, real, n_rk_stages, MeshType>::generate_reduced_lhs(
    const Epetra_CrsMatrix &system_matrix,
    const Epetra_CrsMatrix &test_basis,
    const Epetra_CrsMatrix &trial_basis)
{
    if (test_basis.RowMap().SameAs(system_matrix.RowMap()) && test_basis.NumGlobalRows() == system_matrix.NumGlobalRows()){
        Epetra_CrsMatrix epetra_reduced_lhs(Epetra_DataAccess::Copy, test_basis.DomainMap(), test_basis.NumGlobalCols());
        Epetra_CrsMatrix epetra_reduced_lhs_tmp(Epetra_DataAccess::Copy, system_matrix.RowMap(), test_basis.NumGlobalCols());
        if (EpetraExt::MatrixMatrix::Multiply(system_matrix, false, test_basis, false, epetra_reduced_lhs_tmp) != 0){
            std::cerr << "Error in first Matrix Multiplication" << std::endl;
            return nullptr;
        };
        if (EpetraExt::MatrixMatrix::Multiply(trial_basis, true, epetra_reduced_lhs_tmp, false, epetra_reduced_lhs) != 0){
            std::cerr << "Error in second Matrix Multiplication" << std::endl;
            return nullptr;
        };
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
    int HyperReducedPODGalerkinRungeKuttaODESolver<dim,real,n_rk_stages,MeshType>::multiply(Epetra_CrsMatrix &epetra_matrix,
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
void HyperReducedPODGalerkinRungeKuttaODESolver<dim,real,n_rk_stages,MeshType>::epetra_to_dealii(Epetra_Vector &epetra_vector,
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
    template class HyperReducedPODGalerkinRungeKuttaODESolver<PHILIP_DIM, double,1, dealii::Triangulation<PHILIP_DIM> >;
    template class HyperReducedPODGalerkinRungeKuttaODESolver<PHILIP_DIM, double,2, dealii::Triangulation<PHILIP_DIM> >;
    template class HyperReducedPODGalerkinRungeKuttaODESolver<PHILIP_DIM, double,3, dealii::Triangulation<PHILIP_DIM> >;
    template class HyperReducedPODGalerkinRungeKuttaODESolver<PHILIP_DIM, double,4, dealii::Triangulation<PHILIP_DIM> >;
    template class HyperReducedPODGalerkinRungeKuttaODESolver<PHILIP_DIM, double,1, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
    template class HyperReducedPODGalerkinRungeKuttaODESolver<PHILIP_DIM, double,2, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
    template class HyperReducedPODGalerkinRungeKuttaODESolver<PHILIP_DIM, double,3, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
    template class HyperReducedPODGalerkinRungeKuttaODESolver<PHILIP_DIM, double,4, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
#if PHILIP_DIM != 1
    template class HyperReducedPODGalerkinRungeKuttaODESolver<PHILIP_DIM, double,1, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class HyperReducedPODGalerkinRungeKuttaODESolver<PHILIP_DIM, double,2, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class HyperReducedPODGalerkinRungeKuttaODESolver<PHILIP_DIM, double,3, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class HyperReducedPODGalerkinRungeKuttaODESolver<PHILIP_DIM, double,4, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
#endif
}