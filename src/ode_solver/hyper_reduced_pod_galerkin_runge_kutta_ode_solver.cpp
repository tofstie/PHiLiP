#include "hyper_reduced_pod_galerkin_runge_kutta_ode_solver.h"
#include <EpetraExt_MatrixMatrix.h>
#include <Epetra_LinearProblem.h>
#include <Amesos_Lapack.h>
#include <Epetra_SerialComm.h>
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
    std::ofstream rk_stage_file("rk_stage_file_stage_" + std::to_string(istage) + ".txt");
    this->rk_stage[istage].print(rk_stage_file);
    this->dg->solution = this->rk_stage[istage];
}

template <int dim, typename real, int n_rk_stages, typename MeshType>
void HyperReducedPODGalerkinRungeKuttaODESolver<dim, real, n_rk_stages, MeshType>::calculate_stage_derivative (int istage, real dt) {
    this->dg->set_current_time(this->current_time + this->butcher_tableau->get_c(istage)*dt);
    if(this->all_parameters->reduced_order_param.entropy_variables_in_snapshots){
        dealii::TrilinosWrappers::SparseMatrix pod_basis;
        pod_basis.reinit(*epetra_test_basis);
        this->dg->calculate_global_entropy();
        this->dg->calculate_ROM_projected_entropy(pod_basis);
    }
    this->dg->assemble_residual(); //RHS : du/dt = RHS = F(u_n + dt* sum(a_ij*V*k_j) + dt * a_ii * u^(istage)))
    Epetra_Vector epetra_right_hand_side(Epetra_DataAccess::View, epetra_test_basis->RowMap(), this->dg->right_hand_side.begin());
    std::ofstream epetra_right_hand_side_file("epetra_right_hand_side"+std::to_string(istage)+".txt");
    epetra_right_hand_side.Print(epetra_right_hand_side_file);
    std::shared_ptr<Epetra_Vector> hyper_reduced_rhs = generate_hyper_reduced_residual(epetra_right_hand_side, *epetra_test_basis);
    hyper_reduced_rhs->Scale(1.0);
    std::ofstream hyper_reduced_rhs_file("hyper_reduced_rhs"+std::to_string(istage)+".txt");
    hyper_reduced_rhs->Print(hyper_reduced_rhs_file);
    if(this->all_parameters->use_inverse_mass_on_the_fly){
        assert(1 == 0 && "Not Implemented: use_inverse_mass_on_the_fly=true && ode_solver_type=pod_galerkin_rk_solver\n Please set use_inverse_mass_on_the_fly=false and try again");
    } else{
        // Creating Reduced RHS
        dealii::LinearAlgebra::distributed::Vector<double> dealii_reduced_stage_i;

        Epetra_Vector epetra_reduced_rhs(*hyper_reduced_rhs); // Flip to range map?
        //int rank = dealii::Utilities::MPI::this_mpi_process(this->dg->solution.get_mpi_communicator());
        //std::ofstream dealii_rhs("rhs_dealii_"+ std::to_string(rank)+ ".txt");
        //print_dealii(dealii_rhs,rhs);
        //std::ofstream rhs_file("rhs_file_"+ std::to_string(rank)+ ".txt");
        //epetra_rhs.Print(rhs_file);
        //Epetra_Vector epetra_reduced_rhs(epetra_test_basis->DomainMap());
        //epetra_trial_basis->Multiply(true,epetra_rhs,epetra_reduced_rhs);
        std::ofstream epetra_reduced_right_hand_side_file("epetra_reduced_right_hand_side"+std::to_string(istage)+".txt");
        epetra_reduced_rhs.Print(epetra_reduced_right_hand_side_file);
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
        std::ofstream reduced_stages_file("reduced_stages"+std::to_string(istage)+".txt");
        this->reduced_rk_stage[istage].print(reduced_stages_file);
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
    std::ofstream new_solution_file("new_solution_file.txt");
    this->solution_update.print(new_solution_file);
}

template <int dim, typename real, int n_rk_stages, typename MeshType>
void HyperReducedPODGalerkinRungeKuttaODESolver<dim, real, n_rk_stages, MeshType>::allocate_runge_kutta_system() {
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
    std::ofstream test_basis_file("test_basis_file.txt");
    std::ofstream trail_basis_file("trail_basis_file.txt");
    epetra_test_basis->Print(test_basis_file);
    epetra_trial_basis->Print(trail_basis_file);
    epetra_reduced_lhs = generate_reduced_lhs(epetra_mass_matrix,*epetra_test_basis,*epetra_test_basis);
    if(this->all_parameters->reduced_order_param.entropy_variables_in_snapshots){
        this->dg->calculate_projection_matrix(*epetra_reduced_lhs,*epetra_trial_basis);
        this->dg->set_galerkin_basis(epetra_test_basis);
    }
    std::ofstream lhs_file("lhs_file.txt");
    epetra_reduced_lhs->Print(lhs_file);
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
                                                                                                                                       const bool trial_basis)
{
    if (!trial_basis) return std::make_shared<Epetra_CrsMatrix>(pod_basis);
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

            // Find contribution of element to the JacobianThe root is known to exist due to the
            Epetra_CrsMatrix V_L_e_T(Epetra_DataAccess::Copy, LeRowMap, number_modes);
            Epetra_CrsMatrix V_e_m(Epetra_DataAccess::Copy, basis_rowmap, number_modes);
            EpetraExt::MatrixMatrix::Multiply(L_e, false, pod_basis, false, V_L_e_T, true);
            EpetraExt::MatrixMatrix::Multiply(L_e, true, V_L_e_T, false, V_e_m, true);
            // Add the contribution of the element to the hyper-reduced Jacobian with scaling from the weights
            double scaling = 1.0;
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
    const Epetra_CrsMatrix &/*system_matrix*/,
    const Epetra_CrsMatrix &test_basis,
    const Epetra_CrsMatrix &trial_basis)
{
    Epetra_CrsMatrix lhs_matrix(Epetra_DataAccess::Copy,test_basis.DomainMap(),test_basis.NumGlobalCols());
    // Setup for Local Mass Matrix
    using FR_enum = Parameters::AllParameters::Flux_Reconstruction;
    const FR_enum FR_Type = this->all_parameters->flux_reconstruction_type;
    using FR_Aux_enum = Parameters::AllParameters::Flux_Reconstruction_Aux;
    const FR_Aux_enum FR_Type_Aux = this->all_parameters->flux_reconstruction_aux_type;
    const int nstate = this->all_parameters->nstate;
    std::vector<dealii::types::global_dof_index> dofs_indices;
    const unsigned int init_grid_degree = this->dg->high_order_grid->fe_system.tensor_degree();
    OPERATOR::mapping_shape_functions<dim,2*dim,real> mapping_basis(1, this->dg->max_degree, init_grid_degree);//first set at max degree
    OPERATOR::basis_functions<dim,2*dim,real> basis(1, this->dg->max_degree, init_grid_degree);
    OPERATOR::local_mass<dim,2*dim,real> reference_mass_matrix(1, this->dg->max_degree, init_grid_degree);//first set at max degree
    OPERATOR::local_Flux_Reconstruction_operator<dim,2*dim,real> reference_FR(1, this->dg->max_degree, init_grid_degree, FR_Type);
    OPERATOR::local_Flux_Reconstruction_operator_aux<dim,2*dim,real> reference_FR_aux(1, this->dg->max_degree, init_grid_degree, FR_Type_Aux);
    OPERATOR::derivative_p<dim,2*dim,real> deriv_p(1, this->dg->max_degree, init_grid_degree);
    const int N_FOM_dim = test_basis.NumGlobalRows(); // Length of solution vector
    auto first_cell = this->dg->dof_handler.begin_active();
    const bool Cartesian_first_element = (first_cell->manifold_id() == dealii::numbers::flat_manifold_id);

    this->dg->reinit_operators_for_mass_matrix(Cartesian_first_element, this->dg->max_degree, init_grid_degree, mapping_basis, basis, reference_mass_matrix, reference_FR, reference_FR_aux, deriv_p);

    //Loop over cells and set the matrices.
    auto metric_cell = this->dg->high_order_grid->dof_handler_grid.begin_active();
    for (auto cell = this->dg->dof_handler.begin_active(); cell!=this->dg->dof_handler.end(); ++cell, ++metric_cell) {

        if (!cell->is_locally_owned()) continue;
        if (ECSW_weights[cell->active_cell_index()] == 0) continue;

        this->dg->global_mass_matrix.reinit(this->dg->locally_owned_dofs, this->dg->mass_sparsity_pattern);
        const bool Cartesian_element = (cell->manifold_id() == dealii::numbers::flat_manifold_id);

        const unsigned int fe_index_curr_cell = cell->active_fe_index();
        const unsigned int curr_grid_degree   = this->dg->high_order_grid->fe_system.tensor_degree();//in the future the metric cell's should store a local grid degree. currently high_order_grid dof_handler_grid doesn't have that capability

        //Check if need to recompute the 1D basis for the current degree (if different than previous cell)
        //That is, if the poly_degree, manifold type, or grid degree is different than previous reference operator
        if((fe_index_curr_cell != mapping_basis.current_degree) ||
           (curr_grid_degree != mapping_basis.current_grid_degree))
        {
            this->dg->reinit_operators_for_mass_matrix(Cartesian_element, fe_index_curr_cell, curr_grid_degree, mapping_basis, basis, reference_mass_matrix, reference_FR, reference_FR_aux, deriv_p);

            mapping_basis.current_degree = fe_index_curr_cell;
            basis.current_degree = fe_index_curr_cell;
            reference_mass_matrix.current_degree = fe_index_curr_cell;
            reference_FR.current_degree = fe_index_curr_cell;
            reference_FR_aux.current_degree = fe_index_curr_cell;
            deriv_p.current_degree = fe_index_curr_cell;
        }

        // Current reference element related to this physical cell
        const unsigned int n_dofs_cell = this->dg->fe_collection[fe_index_curr_cell].n_dofs_per_cell();
        const int n_dofs_cell_int = this->dg->fe_collection[fe_index_curr_cell].n_dofs_per_cell();
        const unsigned int n_quad_pts  = this->dg->volume_quadrature_collection[fe_index_curr_cell].size();

        //setup metric cell
        const dealii::FESystem<dim> &fe_metric = this->dg->high_order_grid->fe_system;
        const unsigned int n_metric_dofs = this->dg->high_order_grid->fe_system.dofs_per_cell;
        const unsigned int n_grid_nodes  = n_metric_dofs/dim;
        std::vector<dealii::types::global_dof_index> metric_dof_indices(n_metric_dofs);
        metric_cell->get_dof_indices (metric_dof_indices);
        // get mapping_support points
        std::array<std::vector<real>,dim> mapping_support_points;
        for(int idim=0; idim<dim; idim++){
            mapping_support_points[idim].resize(n_metric_dofs/dim);
        }
        const std::vector<unsigned int > &index_renumbering = dealii::FETools::hierarchic_to_lexicographic_numbering<dim>(curr_grid_degree);
        for (unsigned int idof = 0; idof< n_metric_dofs; ++idof) {
            const real val = (this->dg->high_order_grid->volume_nodes[metric_dof_indices[idof]]);
            const unsigned int istate = fe_metric.system_to_component_index(idof).first;
            const unsigned int ishape = fe_metric.system_to_component_index(idof).second;
            const unsigned int igrid_node = index_renumbering[ishape];
            mapping_support_points[istate][igrid_node] = val;
        }

        //get determinant of Jacobian
        OPERATOR::metric_operators<real, dim, 2*dim> metric_oper(nstate, fe_index_curr_cell, curr_grid_degree);
        metric_oper.build_determinant_volume_metric_Jacobian(
                        n_quad_pts, n_grid_nodes,
                        mapping_support_points,
                        mapping_basis);

        //Get dofs indices to set local matrices in global.
        dofs_indices.resize(n_dofs_cell);
        cell->get_dof_indices (dofs_indices);
        //Compute local matrices and set them in the global system.
        this->dg->evaluate_local_metric_dependent_mass_matrix_and_set_in_global_mass_matrix(
            Cartesian_element,
            false,
            fe_index_curr_cell,
            curr_grid_degree,
            n_quad_pts,
            n_dofs_cell,
            dofs_indices,
            metric_oper,
            basis,
            reference_mass_matrix,
            reference_FR,
            reference_FR_aux,
            deriv_p);
        // Set up the Le matrix
        const Epetra_SerialComm sComm;
        Epetra_Map LeRowMap(n_dofs_cell_int, 0, sComm);
        Epetra_Map LeTRowMap(N_FOM_dim, 0, sComm);
        Epetra_CrsMatrix L_e(Epetra_DataAccess::Copy, LeRowMap, LeTRowMap, 1);
        Epetra_CrsMatrix L_e_T(Epetra_DataAccess::Copy, LeTRowMap, n_dofs_cell_int);
        double posOne = 1.0;

        for(int i = 0; i < n_dofs_cell_int; i++){
            const int col = dofs_indices[i];
            L_e.InsertGlobalValues(i, 1, &posOne , &col);
            L_e_T.InsertGlobalValues(col, 1, &posOne , &i);
        }
        L_e.FillComplete(LeTRowMap, LeRowMap);
        L_e_T.FillComplete(LeRowMap, LeTRowMap);
        Epetra_CrsMatrix epetra_mass_matrix = this->dg->global_mass_matrix.trilinos_matrix();
        // Preform the Pre-mult of LeM
        Epetra_CrsMatrix J_L_e_T(Epetra_DataAccess::Copy, epetra_mass_matrix.RowMap(), n_dofs_cell_int);
        Epetra_CrsMatrix J_e_m(Epetra_DataAccess::Copy, LeRowMap, n_dofs_cell_int);
        EpetraExt::MatrixMatrix::Multiply(epetra_mass_matrix, false, L_e_T, false, J_L_e_T, true);
        EpetraExt::MatrixMatrix::Multiply(L_e, false, J_L_e_T, false, J_e_m, true);
        // Preform post-mult of LeMLe
        Epetra_CrsMatrix M_temp(Epetra_DataAccess::Copy, LeRowMap, N_FOM_dim);
        Epetra_CrsMatrix M_global_e(Epetra_DataAccess::Copy, LeTRowMap, N_FOM_dim);
        EpetraExt::MatrixMatrix::Multiply(J_e_m, false, L_e, false, M_temp, true);
        EpetraExt::MatrixMatrix::Multiply(L_e_T, false, M_temp, false, M_global_e, true);
        std::ofstream mass_element_le_file("mass_element.txt");
        M_global_e.Print(mass_element_le_file);
        if (test_basis.RowMap().SameAs(M_global_e.RowMap()) && test_basis.NumGlobalRows() == M_global_e.NumGlobalRows()){
            Epetra_CrsMatrix epetra_reduced_lhs(Epetra_DataAccess::Copy, test_basis.DomainMap(), test_basis.NumGlobalCols());
            Epetra_CrsMatrix epetra_reduced_lhs_tmp(Epetra_DataAccess::Copy, M_global_e.RowMap(), test_basis.NumGlobalCols());
            if (EpetraExt::MatrixMatrix::Multiply(M_global_e, false, test_basis, false, epetra_reduced_lhs_tmp) != 0){
                std::cerr << "Error in first Matrix Multiplication" << std::endl;
                return nullptr;
            };
            if (EpetraExt::MatrixMatrix::Multiply(trial_basis, true, epetra_reduced_lhs_tmp, false, epetra_reduced_lhs) != 0){
                std::cerr << "Error in second Matrix Multiplication" << std::endl;
                return nullptr;
            };
            if (EpetraExt::MatrixMatrix::Add(epetra_reduced_lhs,false,1.0,lhs_matrix,1.0) != 0) {
                std::cerr << "Error in third Matrix Add" << std::endl;
                return nullptr;
            }
        } else {
            if(!(test_basis.RowMap().SameAs(epetra_mass_matrix.RowMap()))){
                std::cerr << "Error: Inconsistent maps" << std::endl;
            } else {
                std::cerr << "Error: Inconsistent row sizes" << std::endl
                << "System: " << std::to_string(epetra_mass_matrix.NumGlobalRows()) << std::endl
                << "Test: " << std::to_string(test_basis.NumGlobalRows()) << std::endl;
            }
        }
    }
    //end of cell loop
    lhs_matrix.FillComplete(test_basis.DomainMap(),test_basis.DomainMap());
    std::ofstream lhs_file("lhs_matrix.txt");
    lhs_matrix.Print(lhs_file);
    this->dg->evaluate_mass_matrices(false);
    return std::make_shared<Epetra_CrsMatrix>(lhs_matrix);
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