#include "hyper_reduced_pod_galerkin_runge_kutta_ode_solver.h"
#include <EpetraExt_MatrixMatrix.h>
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
                                                                                , epetra_pod_basis(pod->getPODBasis()->trilinos_matrix())
                                                                                , epetra_test_basis(nullptr)
{}

template <int dim, typename real, int n_rk_stages, typename MeshType>
void HyperReducedPODGalerkinRungeKuttaODESolver<dim, real, n_rk_stages, MeshType>::calculate_stage_solution (int istage, real dt, const bool pseudotime) {
    this->rk_stage[istage] = 0.0;
    this->reduced_rk_stage[istage] = 0.0;
    for(int j = 0; j < istage; ++j){
        if(this->butcher_tableau->get_a(istage,j) != 0){
            dealii::LinearAlgebra::distributed::Vector<double> dealii_rk_stage_j;
            multiply(*epetra_test_basis,this->reduced_rk_stage[j],dealii_rk_stage_j,solution_index,false);
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
    this->dg->assemble_residual(); //RHS : du/dt = RHS = F(u_n + dt* sum(a_ij*V*k_j) + dt * a_ii * u^(istage)))
    Epetra_Vector epetra_right_hand_side(Epetra_DataAccess::View, epetra_test_basis->RowMap(), this->dg->right_hand_side.begin());
    std::shared_ptr<Epetra_Vector> hyper_reduced_rhs = generate_hyper_reduced_residual(epetra_right_hand_side, epetra_test_basis);
    hyper_reduced_rhs->Scale(-1.0);
    return;
}

template <int dim, typename real, int n_rk_stages, typename MeshType>
void HyperReducedPODGalerkinRungeKuttaODESolver<dim, real, n_rk_stages, MeshType>::sum_stages (real dt, const bool pseudotime) {
    dealii::LinearAlgebra::distributed::Vector<double> reduced_sum;
    reduced_sum.reinit(this->reduced_rk_stage[0]);
    for (int istage = 0; istage < n_rk_stages; ++istage){
        reduced_sum.add(dt* this->butcher_tableau->get_b(istage),this->reduced_rk_stage[istage]);
    }
    // Convert Reduced order step to Full order step
    dealii::LinearAlgebra::distributed::Vector<double> dealii_update;
    multiply(*epetra_test_basis,reduced_sum,dealii_update,solution_index,false);
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

    epetra_test_basis = generate_test_basis(epetra_pod_basis);
    epetra_reduced_lhs = generate_reduced_lhs(epetra_mass_matrix,epetra_test_basis.get());
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
std::shared_ptr<Epetra_CrsMatrix> HyperReducedPODGalerkinRungeKuttaODESolver<dim, real, n_rk_stages, MeshType>::generate_test_basis(const Epetra_CrsMatrix &pod_basis) {
    Epetra_MpiComm epetra_comm(MPI_COMM_WORLD);
    Epetra_Map basis_rowmap = pod_basis.RowMap();
    Epetra_CrsMatrix hyper_reduced_basis(Epetra_DataAccess::Copy, basis_rowmap,pod_basis.NumGlobalCols());
    const int N = pod_basis.NumGlobalRows();
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
            // Create L_e matrix and transposed L_e matrixfor current cell
            Epetra_Map LeRowMap(n_dofs_curr_cell, 0, epetra_comm);
            Epetra_CrsMatrix L_e(Epetra_DataAccess::Copy, LeRowMap, N);
            Epetra_CrsMatrix L_e_T(Epetra_DataAccess::Copy, basis_rowmap, n_dofs_curr_cell);
            Epetra_Map LePLUSRowMap(neighbour_dofs_curr_cell, 0, epetra_comm);
            Epetra_CrsMatrix L_e_PLUS(Epetra_DataAccess::Copy, LePLUSRowMap, N);
            const double posOne = 1.0;

            for(int i = 0; i < n_dofs_curr_cell; i++){
                const int col = current_dofs_indices[i];
                L_e.InsertGlobalValues(i, 1, &posOne , &col);
                L_e_T.InsertGlobalValues(col, 1, &posOne , &i);
            }
            L_e.FillComplete(basis_rowmap, LeRowMap);
            L_e_T.FillComplete(LeRowMap, basis_rowmap);

            for(int i = 0; i < neighbour_dofs_curr_cell; i++){
                const int col = neighbour_dofs_indices[i];
                L_e_PLUS.InsertGlobalValues(i, 1, &posOne , &col);
            }
            L_e_PLUS.FillComplete(basis_rowmap, LePLUSRowMap);

            // Find contribution of element to the Jacobian
            Epetra_CrsMatrix V_L_e_T(Epetra_DataAccess::Copy, basis_rowmap, neighbour_dofs_curr_cell);
            Epetra_CrsMatrix V_e_m(Epetra_DataAccess::Copy, LeRowMap, neighbour_dofs_curr_cell);
            EpetraExt::MatrixMatrix::Multiply(pod_basis, false, L_e_PLUS, true, V_L_e_T, true);
            EpetraExt::MatrixMatrix::Multiply(L_e, false, V_L_e_T, false, V_e_m, true);

            // Jacobian for this element in the global dimensions
            Epetra_CrsMatrix V_temp(Epetra_DataAccess::Copy, LeRowMap, N);
            Epetra_CrsMatrix V_global_e(Epetra_DataAccess::Copy, basis_rowmap, N);
            EpetraExt::MatrixMatrix::Multiply(V_e_m, false, L_e_PLUS, false, V_temp, true);
            EpetraExt::MatrixMatrix::Multiply(L_e_T, false, V_temp, false, V_global_e, true);

            // Add the contribution of the element to the hyper-reduced Jacobian with scaling from the weights
            double scaling = ECSW_weights[cell->active_cell_index()];
            EpetraExt::MatrixMatrix::Add(V_global_e, false, scaling, hyper_reduced_basis, 1.0);
        }
    }
    hyper_reduced_basis.FillComplete();
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
    const Epetra_CrsMatrix &test_basis)
{
    if (test_basis.RowMap().SameAs(system_matrix.RowMap()) && test_basis.NumGlobalRows() == system_matrix.NumGlobalRows()){
        Epetra_CrsMatrix epetra_reduced_lhs(Epetra_DataAccess::Copy, test_basis.DomainMap(), test_basis.NumGlobalCols());
        Epetra_CrsMatrix epetra_reduced_lhs_tmp(Epetra_DataAccess::Copy, system_matrix.RowMap(), test_basis.NumGlobalCols());
        if (EpetraExt::MatrixMatrix::Multiply(system_matrix, false, test_basis, false, epetra_reduced_lhs_tmp) != 0){
            std::cerr << "Error in first Matrix Multiplication" << std::endl;
            return nullptr;
        };
        if (EpetraExt::MatrixMatrix::Multiply(test_basis, true, epetra_reduced_lhs_tmp, false, epetra_reduced_lhs) != 0){
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
                                                                    const dealii::IndexSet &index_set,
                                                                    const bool transpose // Transpose needs to be used with care of maps
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
void HyperReducedPODGalerkinRungeKuttaODESolver<dim,real,n_rk_stages,MeshType>::epetra_to_dealii(Epetra_Vector &epetra_vector,
                                                                             dealii::LinearAlgebra::distributed::Vector<double> &dealii_vector,
                                                                             const dealii::IndexSet &index_set)
{
    const Epetra_BlockMap &epetra_map = epetra_vector.Map();
    dealii_vector.reinit(index_set,this->mpi_communicator);
    for(int i = 0; i < epetra_map.NumMyElements();++i){
        int global_idx = epetra_map.GID(i);
        if(dealii_vector.in_local_range(global_idx)){
            dealii_vector[global_idx] = epetra_vector[i];
        }
    }
    dealii_vector.compress(dealii::VectorOperation::insert);

}

}