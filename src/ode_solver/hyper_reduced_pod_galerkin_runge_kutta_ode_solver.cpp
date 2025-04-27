#include "hyper_reduced_pod_galerkin_runge_kutta_ode_solver.h"
#include <EpetraExt_MatrixMatrix.h>
#include <Epetra_LinearProblem.h>
#include <Amesos_Lapack.h>
#include <Epetra_SerialComm.h>
#include "linear_solver/helper_functions.h"
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
        pod_basis.reinit(*epetra_trial_basis);
        this->dg->calculate_global_entropy(true);
        this->dg->calculate_ROM_projected_entropy(pod_basis);
        this->dg->assemble_hyper_reduced_residual(*Qtx,*Qty,*Qtz,*BEtx);
    } else {
        this->dg->assemble_residual();
    }
     //RHS : du/dt = RHS = F(u_n + dt* sum(a_ij*V*k_j) + dt * a_ii * u^(istage)))
    std::ofstream rhs_file("rhs_before_hyper_"+ std::to_string(istage) +".txt");
    for(unsigned int i = 0 ; i < this->dg->right_hand_side.size(); i++){
        if (this->dg->right_hand_side.in_local_range(i)){
            rhs_file << this->dg->right_hand_side[i] << '\n';
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    rhs_file.close();
    Epetra_Vector epetra_right_hand_side(Epetra_DataAccess::View, epetra_trial_basis->RowMap(), this->dg->right_hand_side.begin());
    std::ofstream epetra_right_hand_side_file("epetra_right_hand_side"+std::to_string(istage)+".txt");
    epetra_right_hand_side.Print(epetra_right_hand_side_file);
    std::shared_ptr<Epetra_Vector> hyper_reduced_rhs = generate_hyper_reduced_residual(epetra_right_hand_side, *epetra_trial_basis);
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
    std::ofstream solution_file("past_solution.txt");
    for(unsigned int i = 0 ; i < this->solution_update.size(); i++){
        if (this->solution_update.in_local_range(i)){
            solution_file << this->solution_update[i] << '\n';
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    solution_file.close();
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
    for(unsigned int i = 0 ; i < this->solution_update.size(); i++){
        if (this->solution_update.in_local_range(i)){
            new_solution_file << this->solution_update[i] << '\n';
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    new_solution_file.close();
}

template <int dim, typename real, int n_rk_stages, typename MeshType>
void HyperReducedPODGalerkinRungeKuttaODESolver<dim, real, n_rk_stages, MeshType>::allocate_runge_kutta_system() {
    // Setting up butcher tableau
    this->butcher_tableau->set_tableau();

    this->butcher_tableau_aii_is_zero.resize(n_rk_stages);
    std::fill(this->butcher_tableau_aii_is_zero.begin(),
              this->butcher_tableau_aii_is_zero.end(),
              false);
    for (int istage=0; istage<n_rk_stages; ++istage) {
        if (this->butcher_tableau->get_a(istage,istage)==0.0)     this->butcher_tableau_aii_is_zero[istage] = true;

    }
    // Initialize solution update
    this->solution_update.reinit(this->dg->right_hand_side);

    // Create distributions
    const Epetra_Map reduced_map = epetra_pod_basis.DomainMap();
    reduced_index = dealii::IndexSet(reduced_map);
    solution_index = this->dg->solution.locally_owned_elements();
    this->reduced_rk_stage.resize(n_rk_stages);
    for (int istage=0; istage<n_rk_stages; ++istage){
        this->reduced_rk_stage[istage].reinit(reduced_index, this->mpi_communicator); // Add IndexSet
    }
    // Store weights into DG (FIX FOR MULTICORE LATER)
    dealii::Vector<double> weights_dealii(ECSW_weights.GlobalLength());
    for(unsigned int i = 0; i < weights_dealii.size(); ++i) {
        weights_dealii[i] = ECSW_weights[i];
    }
    this->dg->reduced_mesh_weights = weights_dealii;
    //this->dg->reduced_mesh_weights = 0.5;
    // Initialize the Mass Matrix
    Epetra_CrsMatrix epetra_mass_matrix(this->dg->global_mass_matrix.trilinos_matrix());
    std::ofstream global_mass_matrix_file("global_mass_matrix_file.txt");
    epetra_mass_matrix.Print(global_mass_matrix_file);
    Epetra_CrsMatrix epetra_quad_mass_matrix(this->dg->global_quad_mass_matrix.trilinos_matrix());
    this->dg->evaluate_mass_matrices(true);
    // Generate the Test and Trail Basis
    epetra_test_basis = generate_test_basis(epetra_pod_basis, false);
    epetra_trial_basis = generate_test_basis(epetra_pod_basis, true);
    Epetra_CrsMatrix quad_basis = pod->getTestBasis()->trilinos_matrix();
    std::shared_ptr<Epetra_CrsMatrix> epetra_quad_basis = std::make_shared<Epetra_CrsMatrix>(quad_basis);
    std::ofstream test_basis_file("test_basis_file.txt");
    std::ofstream trail_basis_file("trail_basis_file.txt");
    epetra_test_basis->Print(test_basis_file);
    epetra_trial_basis->Print(trail_basis_file);
    // Generate the LHS


    // If using ESROM, create projection operator
    if(this->all_parameters->reduced_order_param.entropy_variables_in_snapshots){

        this->dg->set_galerkin_basis(epetra_trial_basis,false);
        std::cout << "Setting Vt" << std::endl;
        this->dg->set_galerkin_basis(epetra_quad_basis,true);
        //this->pod->setPODBasis(epetra_trial_basis);
    }



    // Creation of Qtx,Qty,Qtz
    const int global_size = this->dg->solution.size();
    Epetra_MpiComm comm(MPI_COMM_WORLD);
    Epetra_Map global_map(global_size/(dim+2),0,comm); // FIX THIS LATER 📢📢📢, THE NUMBER SHOULD BE NSTATE
    Epetra_Map domain_map = global_map;
    // Construct Q
    Epetra_CrsMatrix Qx(Epetra_DataAccess::Copy,global_map,epetra_mass_matrix.ColMap().MaxElementSize());
    Epetra_CrsMatrix Qy(Epetra_DataAccess::Copy,global_map,epetra_mass_matrix.ColMap().MaxElementSize());
    Epetra_CrsMatrix Qz(Epetra_DataAccess::Copy,global_map,epetra_mass_matrix.ColMap().MaxElementSize());
    this->dg->construct_global_Q(Qx,Qy,Qz,true);
    Eigen::MatrixXd Qx_eig = epetra_to_eig_matrix(Qx);
    std::ofstream file("(Q-Qt)x_eig.txt");
    const static Eigen::IOFormat CSVFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
    if (file.is_open()){
        file << Qx_eig.format(CSVFormat);
    }
    file.close();
    // Eigen::MatrixXd Qy_eig = epetra_to_eig_matrix(Qy);
    // std::ofstream yfile("(Q-Qt)y_eig.txt");
    // if (yfile.is_open()){
    //     yfile << Qy_eig.format(CSVFormat);
    // }
    // yfile.close();
    // Construct Pi_t
    std::cout << "Constructing Pi_t" << std::endl;
    this->dg->test_projection_matrix.resize(dim);
    this->dg->evaluate_hyper_mass_matrices(false,true);
    epetra_mass_matrix = this->dg->global_mass_matrix.trilinos_matrix();
    std::ofstream mass_dof_new_weight_file("mass_dof_new_weight.txt");
    epetra_mass_matrix.Print(mass_dof_new_weight_file);
    epetra_reduced_lhs = generate_reduced_lhs(epetra_mass_matrix,*epetra_trial_basis,*epetra_trial_basis);
    std::ofstream lhs_file("lhs_file.txt");
    epetra_reduced_lhs->Print(lhs_file);
    dealii::TrilinosWrappers::SparseMatrix pod_basis;
    pod_basis.reinit(*epetra_trial_basis);
    this->dg->calculate_projection_matrix(*epetra_reduced_lhs,*epetra_trial_basis);//(pod_basis);
    //this->dg->right_hand_side.reinit(this->dg->dof_handler.n_dofs());
    for(int idim = 0; idim < dim; ++idim) {
        std::ofstream vt_file("vt_file" + std::to_string(idim) + ".txt");
        std::shared_ptr<Epetra_CrsMatrix> hyper_reduced_vt = generate_hyper_test_basis(*this->dg->galerkin_test_basis[idim]);
        hyper_reduced_vt->Print(vt_file);
        std::shared_ptr<Epetra_CrsMatrix> test_lhs = generate_hyper_reduced_lhs(epetra_quad_mass_matrix,*hyper_reduced_vt,*hyper_reduced_vt);
        this->dg->set_test_projection_matrix(test_lhs,hyper_reduced_vt,idim);
        if (idim == 0) {
            Qtx = std::make_shared<Epetra_CrsMatrix>(this->dg->calculate_hyper_reduced_Q(Qx,*this->dg->galerkin_test_basis[idim],idim));
            BEtx = std::make_shared<Epetra_CrsMatrix>(this->dg->calculate_hyper_reduced_Bx(*this->dg->galerkin_test_basis[idim],idim));
            //Qtx = std::make_shared<Epetra_CrsMatrix>(Qx);
        } else if (idim == 1) {
            Qty = std::make_shared<Epetra_CrsMatrix>(this->dg->calculate_hyper_reduced_Q(Qy,*this->dg->galerkin_test_basis[idim],idim));
            //Qty = std::make_shared<Epetra_CrsMatrix>(Qy);
        } else {
            Qtz = std::make_shared<Epetra_CrsMatrix>(this->dg->calculate_hyper_reduced_Q(Qz,*this->dg->galerkin_test_basis[idim],idim));
        }
        this->dg->galerkin_test_basis[idim] = nullptr;
        this->dg->test_projection_matrix[idim] = nullptr;
    }
    this->dg->boundary_term = std::make_shared<Epetra_Vector>(BEtx->RowMap());
    Eigen::MatrixXd Qtx_eig = epetra_to_eig_matrix(*Qtx);
    std::ofstream tfile("Qtx_eig.txt");
    if (tfile.is_open()){
        tfile << Qtx_eig.format(CSVFormat);
    }
    tfile.close();
    // Eigen::MatrixXd Qty_eig = epetra_to_eig_matrix(*Qty);
    // std::ofstream Qtyfile("Qty_eig.txt");
    // if (Qtyfile.is_open()){
    //     Qtyfile << Qty_eig.format(CSVFormat);
    // }
    // Qtyfile.close();
    std::cout << "Construction Qt" << std::endl;
    // Construct Qt

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
    Epetra_Map basis_rowmap = pod_basis.RowMap();
    Epetra_Map basis_domainmap = pod_basis.DomainMap();
    Epetra_CrsMatrix hyper_reduced_basis(Epetra_DataAccess::Copy, basis_rowmap,basis_domainmap, pod_basis.NumGlobalCols());
    const int length = pod_basis.NumGlobalCols();
    int NumEntries;
    std::vector<int> indices(length);
    std::vector<double> values(length);
    for (auto current_cell = this->dg->dof_handler.begin_active(); current_cell != this->dg->dof_handler.end(); ++current_cell) {
        if (!current_cell->is_locally_owned()) continue;
        const int i_fele = current_cell->active_fe_index();
        const unsigned int poly_degree = i_fele;
        const dealii::FESystem<dim,dim> &current_fe_ref = this->dg->fe_collection[i_fele];
        const unsigned int n_dofs_curr_cell = current_fe_ref.n_dofs_per_cell();
        const unsigned int n_quad_pts = this->dg->volume_quadrature_collection[poly_degree].size();
        std::vector<dealii::types::global_dof_index> current_dofs_indices;
        current_dofs_indices.resize(n_dofs_curr_cell);
        current_cell->get_dof_indices (current_dofs_indices);
        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
            for (int istate = 0; istate < this->dg->nstate; ++istate) {
                int current_row = current_dofs_indices[iquad+istate*n_quad_pts];
                if (this->dg->reduced_mesh_weights[this->dg->dofs_to_quad[current_row]] == 0) continue;
                pod_basis.ExtractGlobalRowCopy(current_row,length,NumEntries,values.data(),indices.data());
                if(NumEntries != 0) {
                    hyper_reduced_basis.InsertGlobalValues(current_row,NumEntries,values.data(),indices.data());
                }
            }
        }
    }
    hyper_reduced_basis.FillComplete(basis_domainmap, basis_rowmap);
    std::ofstream file("HR_Pod_basis.txt");
    hyper_reduced_basis.Print(file);
    return std::make_shared<Epetra_CrsMatrix>(hyper_reduced_basis);
}
template <int dim, typename real, int n_rk_stages, typename MeshType>
std::shared_ptr<Epetra_CrsMatrix> HyperReducedPODGalerkinRungeKuttaODESolver<dim, real, n_rk_stages, MeshType>::generate_hyper_test_basis(
    const Epetra_CrsMatrix &pod_basis) {
    Epetra_MpiComm epetra_comm(MPI_COMM_WORLD);
    Epetra_Map basis_rowmap = pod_basis.RowMap();
    Epetra_Map basis_domainmap = pod_basis.DomainMap();
    Epetra_CrsMatrix hyper_reduced_basis(Epetra_DataAccess::Copy, basis_rowmap,basis_domainmap, pod_basis.NumGlobalCols());
    const int length = pod_basis.NumGlobalCols();
    int NumEntries;
    std::vector<int> indices(length);
    std::vector<double> values(length);
    for (auto current_cell = this->dg->dof_handler.begin_active(); current_cell != this->dg->dof_handler.end(); ++current_cell) {
        if (!current_cell->is_locally_owned()) continue;
        const int i_fele = current_cell->active_fe_index();
        const unsigned int poly_degree = i_fele;
        const dealii::FESystem<dim,dim> &current_fe_ref = this->dg->fe_collection[i_fele];
        const unsigned int n_dofs_curr_cell = current_fe_ref.n_dofs_per_cell();
        const unsigned int n_quad_pts = this->dg->volume_quadrature_collection[poly_degree].size();
        std::vector<dealii::types::global_dof_index> current_dofs_indices;
        current_dofs_indices.resize(n_dofs_curr_cell);
        current_cell->get_dof_indices (current_dofs_indices);
        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
            if (this->dg->reduced_mesh_weights[this->dg->dofs_to_quad[current_dofs_indices[iquad]]] == 0) continue;
            int current_row = this->dg->dofs_to_quad[current_dofs_indices[iquad]];
            pod_basis.ExtractGlobalRowCopy(current_row,length,NumEntries,values.data(),indices.data());
            if(NumEntries != 0) {
                hyper_reduced_basis.InsertGlobalValues(current_row,NumEntries,values.data(),indices.data());
            }
        }
    }
    // for (int i = 0; i < pod_basis.NumGlobalRows(); i++) {
    //     if(ECSW_weights[i] == 0) continue;
    //     pod_basis.ExtractGlobalRowCopy(i,length,NumEntries,values.data(),indices.data());
    //     if(NumEntries != 0) {
    //         hyper_reduced_basis.InsertGlobalValues(i,NumEntries,values.data(),indices.data());
    //     }
    // }
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
    Epetra_Map test_basis_colmap = test_basis.ColMap();
     /*Epetra_Map test_basis_rowmap = test_basis.RowMap();
     Epetra_Vector weights_vector(test_basis_rowmap);
     for (const auto &cell : this->dg->dof_handler.active_cell_iterators()) {
         const int active_cell_index = cell->active_cell_index();
         const int fe_index_curr_cell = cell->active_fe_index();
         const dealii::FESystem<dim,dim> &current_fe_ref = this->dg->fe_collection[fe_index_curr_cell];
         const int n_dofs_curr_cell = current_fe_ref.n_dofs_per_cell();
         const int n_quad_pts = n_dofs_curr_cell/this->dg->nstate;

         for(int i_quad = 0; i_quad < n_quad_pts; i_quad++) {
             const int dof_row = this->dg->quad_to_dof[i_quad+active_cell_index*n_quad_pts];
             for(int istate = 0; istate < this->dg->nstate; istate++) {
                 const int dof_row_istate = dof_row + (istate)*n_quad_pts;
                 weights_vector[dof_row_istate] = ECSW_weights[i_quad+active_cell_index*n_quad_pts];
             }
         }
     }
     epetra_right_hand_side.Scale(1.,weights_vector);*/

    Epetra_Vector hyper_reduced_residual(test_basis_colmap);
    test_basis.Multiply(true,epetra_right_hand_side,hyper_reduced_residual);
    if(this->dg->number_global_boundaries != 0) {
        Epetra_MpiComm comm( MPI_COMM_WORLD );
        Epetra_Map boundary_map((int)this->dg->number_global_boundaries,0,comm);
        Epetra_Vector boundary_rhs(Epetra_DataAccess::View, boundary_map,this->dg->BExFB_term.begin());
        Epetra_CrsMatrix Vb(Epetra_DataAccess::Copy,boundary_map,test_basis.NumGlobalCols());
        const int n_quad_pts = this->dg->volume_quadrature_collection[this->dg->all_parameters->flow_solver_param.poly_degree].size();
        const int last_row_istate_zero = test_basis.NumGlobalRows()-1-n_quad_pts*(this->dg->nstate-1);
        const int length = test_basis.NumGlobalCols();
        std::vector<double> V_row(length);
        std::vector<int> V_indices(length);
        int NumEntries;
        for (int istate = 0; istate < this->dg->nstate; istate++) {
            test_basis.ExtractGlobalRowCopy(0+istate*n_quad_pts,length,NumEntries,V_row.data(),V_indices.data());
            Vb.InsertGlobalValues(0+istate,NumEntries,V_row.data(),V_indices.data());
            test_basis.ExtractGlobalRowCopy(last_row_istate_zero+istate*n_quad_pts,length,NumEntries,V_row.data(),V_indices.data());
            Vb.InsertGlobalValues(1*this->dg->nstate+istate,NumEntries,V_row.data(),V_indices.data());
        }
        Vb.FillComplete(test_basis.DomainMap(),boundary_map);
        Epetra_Vector hyper_reduced_boundary_residual(test_basis_colmap);
        Epetra_Vector hyper_reduced_numerical_boundary_flux(test_basis_colmap);
        Vb.Multiply(true,boundary_rhs,hyper_reduced_boundary_residual);
        Vb.Multiply(true,*this->dg->boundary_term,hyper_reduced_numerical_boundary_flux);
        hyper_reduced_residual.Update(1.,hyper_reduced_boundary_residual,-1.,hyper_reduced_numerical_boundary_flux,1.);
    }
    //  /* Refer to Equation (10) in:
    // https://onlinelibrary.wiley.com/doi/10.1002/nme.6603 (includes definitions of matrices used below such as L_e and L_e_PLUS)
    // Create empty Hyper-reduced residual Epetra structure */
    // Epetra_MpiComm epetra_comm(MPI_COMM_WORLD);
    // Epetra_Map test_basis_colmap = test_basis.ColMap();
    // Epetra_Vector hyper_reduced_residual(test_basis_colmap);
    // int N = test_basis.NumGlobalRows();
    // const unsigned int max_dofs_per_cell = this->dg->dof_handler.get_fe_collection().max_dofs_per_cell();
    // std::vector<dealii::types::global_dof_index> current_dofs_indices(max_dofs_per_cell);
    //
    // // Loop through elements
    // for (const auto &cell : this->dg->dof_handler.active_cell_iterators())
    // {
    //     // Add the contributions of an element if the weight from the NNLS is non-zero
    //     if (ECSW_weights[cell->active_cell_index()] != 0){
    //         const int fe_index_curr_cell = cell->active_fe_index();
    //         const dealii::FESystem<dim,dim> &current_fe_ref = this->dg->fe_collection[fe_index_curr_cell];
    //         const int n_dofs_curr_cell = current_fe_ref.n_dofs_per_cell();
    //
    //         current_dofs_indices.resize(n_dofs_curr_cell);
    //         cell->get_dof_indices(current_dofs_indices);
    //
    //         // Create L_e matrix for current cell
    //         Epetra_Map LeRowMap(n_dofs_curr_cell, 0, epetra_comm);
    //         Epetra_Map LeColMap(N, 0, epetra_comm);
    //         Epetra_CrsMatrix L_e(Epetra_DataAccess::Copy, LeRowMap, N);
    //         double posOne = 1.0;
    //
    //         for(int i = 0; i < n_dofs_curr_cell; i++){
    //             const int col = current_dofs_indices[i];
    //             L_e.InsertGlobalValues(i, 1, &posOne , &col);
    //         }
    //         L_e.FillComplete(LeColMap, LeRowMap);
    //
    //         // Find contribution of the current element in the global dimensions
    //         Epetra_Vector local_r(LeRowMap);
    //         Epetra_Vector global_r_e(LeColMap);
    //         L_e.Multiply(false, epetra_right_hand_side, local_r);
    //         L_e.Multiply(true, local_r, global_r_e);
    //
    //         // Find reduced representation of residual and scale by weight
    //         Epetra_Vector reduced_rhs_e(test_basis_colmap);
    //         test_basis.Multiply(true, global_r_e, reduced_rhs_e);
    //         reduced_rhs_e.Scale(ECSW_weights[cell->active_cell_index()]);
    //         double *reduced_rhs_array = new double[reduced_rhs_e.GlobalLength()];
    //
    //         // Add to hyper-reduced representation of the residual
    //         reduced_rhs_e.ExtractCopy(reduced_rhs_array);
    //         for (int k = 0; k < reduced_rhs_e.GlobalLength(); ++k){
    //             hyper_reduced_residual.SumIntoGlobalValues(1, &reduced_rhs_array[k], &k);
    //         }
    //         delete[] reduced_rhs_array;
    //     }
    // }
    return std::make_shared<Epetra_Vector>(hyper_reduced_residual);
}

template <int dim, typename real, int n_rk_stages, typename MeshType>
std::shared_ptr<Epetra_CrsMatrix> HyperReducedPODGalerkinRungeKuttaODESolver<dim, real, n_rk_stages, MeshType>::generate_reduced_lhs(
    const Epetra_CrsMatrix &/*system_matrix*/,
    const Epetra_CrsMatrix &test_basis,
    const Epetra_CrsMatrix &trial_basis)
{
     // Setup for Local Mass Matrix
    using FR_enum = Parameters::AllParameters::Flux_Reconstruction;
    const FR_enum FR_Type = this->all_parameters->flux_reconstruction_type;
    using FR_Aux_enum = Parameters::AllParameters::Flux_Reconstruction_Aux;
    const FR_Aux_enum FR_Type_Aux = this->all_parameters->flux_reconstruction_aux_type;
    std::vector<dealii::types::global_dof_index> dofs_indices;
    const unsigned int init_grid_degree = this->dg->high_order_grid->fe_system.tensor_degree();
    OPERATOR::mapping_shape_functions<dim,2*dim,real> mapping_basis(1, this->dg->max_degree, init_grid_degree);//first set at max degree
    OPERATOR::basis_functions<dim,2*dim,real> basis(1, this->dg->max_degree, init_grid_degree);
    OPERATOR::local_mass<dim,2*dim,real> reference_mass_matrix(1, this->dg->max_degree, init_grid_degree);//first set at max degree
    OPERATOR::local_Flux_Reconstruction_operator<dim,2*dim,real> reference_FR(1, this->dg->max_degree, init_grid_degree, FR_Type);
    OPERATOR::local_Flux_Reconstruction_operator_aux<dim,2*dim,real> reference_FR_aux(1, this->dg->max_degree, init_grid_degree, FR_Type_Aux);
    OPERATOR::derivative_p<dim,2*dim,real> deriv_p(1, this->dg->max_degree, init_grid_degree);
    auto first_cell = this->dg->dof_handler.begin_active();
    const bool Cartesian_first_element = (first_cell->manifold_id() == dealii::numbers::flat_manifold_id);

    this->dg->reinit_operators_for_mass_matrix(Cartesian_first_element, this->dg->max_degree, init_grid_degree, mapping_basis, basis, reference_mass_matrix, reference_FR, reference_FR_aux, deriv_p);
    Epetra_CrsMatrix empty_quad_matrix(Epetra_DataAccess::Copy, test_basis.RowMap(), test_basis.NumGlobalRows());

    //Loop over cells and set the matrices.

    Epetra_CrsMatrix M_global_e = this->dg->global_mass_matrix.trilinos_matrix();
    Epetra_CrsMatrix lhs_matrix(Epetra_DataAccess::Copy, test_basis.DomainMap(), test_basis.NumGlobalCols());
    if (test_basis.RowMap().SameAs(M_global_e.RowMap()) && test_basis.NumGlobalRows() == M_global_e.NumGlobalRows()){
        Epetra_CrsMatrix epetra_reduced_lhs_tmp(Epetra_DataAccess::Copy, M_global_e.RowMap(), test_basis.NumGlobalCols());
        if (EpetraExt::MatrixMatrix::Multiply(M_global_e, false, test_basis, false, epetra_reduced_lhs_tmp) != 0){
            std::cerr << "Error in first Matrix Multiplication" << std::endl;
            return nullptr;
        }
        if (EpetraExt::MatrixMatrix::Multiply(trial_basis, true, epetra_reduced_lhs_tmp, false, lhs_matrix) != 0){
            std::cerr << "Error in second Matrix Multiplication" << std::endl;
            return nullptr;
        }
    } else {
        if(!(test_basis.RowMap().SameAs(M_global_e.RowMap()))){
            std::cerr << "Error: Inconsistent maps" << std::endl;
        } else {
            std::cerr << "Error: Inconsistent row sizes" << std::endl
            << "System: " << std::to_string(M_global_e.NumGlobalRows()) << std::endl
            << "Test: " << std::to_string(test_basis.NumGlobalRows()) << std::endl;
        }
    }
    //end of cell loop
    lhs_matrix.FillComplete(test_basis.DomainMap(),test_basis.DomainMap());
    std::ofstream lhs_file("lhs_matrix.txt");
    lhs_matrix.Print(lhs_file);
    return std::make_shared<Epetra_CrsMatrix>(lhs_matrix);
}

template <int dim, typename real, int n_rk_stages, typename MeshType>
std::shared_ptr<Epetra_CrsMatrix> HyperReducedPODGalerkinRungeKuttaODESolver<dim, real, n_rk_stages, MeshType>::generate_hyper_reduced_lhs(
    const Epetra_CrsMatrix &/*system_matrix*/,
    const Epetra_CrsMatrix &test_basis,
    const Epetra_CrsMatrix &trial_basis)
{
    // Setup for Local Mass Matrix
    using FR_enum = Parameters::AllParameters::Flux_Reconstruction;
    const FR_enum FR_Type = this->all_parameters->flux_reconstruction_type;
    using FR_Aux_enum = Parameters::AllParameters::Flux_Reconstruction_Aux;
    const FR_Aux_enum FR_Type_Aux = this->all_parameters->flux_reconstruction_aux_type;
    std::vector<dealii::types::global_dof_index> dofs_indices;
    const unsigned int init_grid_degree = this->dg->high_order_grid->fe_system.tensor_degree();
    OPERATOR::mapping_shape_functions<dim,2*dim,real> mapping_basis(1, this->dg->max_degree, init_grid_degree);//first set at max degree
    OPERATOR::basis_functions<dim,2*dim,real> basis(1, this->dg->max_degree, init_grid_degree);
    OPERATOR::local_mass<dim,2*dim,real> reference_mass_matrix(1, this->dg->max_degree, init_grid_degree);//first set at max degree
    OPERATOR::local_Flux_Reconstruction_operator<dim,2*dim,real> reference_FR(1, this->dg->max_degree, init_grid_degree, FR_Type);
    OPERATOR::local_Flux_Reconstruction_operator_aux<dim,2*dim,real> reference_FR_aux(1, this->dg->max_degree, init_grid_degree, FR_Type_Aux);
    OPERATOR::derivative_p<dim,2*dim,real> deriv_p(1, this->dg->max_degree, init_grid_degree);
    auto first_cell = this->dg->dof_handler.begin_active();
    const bool Cartesian_first_element = (first_cell->manifold_id() == dealii::numbers::flat_manifold_id);

    this->dg->reinit_operators_for_mass_matrix(Cartesian_first_element, this->dg->max_degree, init_grid_degree, mapping_basis, basis, reference_mass_matrix, reference_FR, reference_FR_aux, deriv_p);
    Epetra_CrsMatrix empty_quad_matrix(Epetra_DataAccess::Copy, test_basis.RowMap(), test_basis.NumGlobalRows());

    //Loop over cells and set the matrices.

    Epetra_CrsMatrix M_global_e = this->dg->global_quad_mass_matrix.trilinos_matrix();
    Epetra_CrsMatrix lhs_matrix(Epetra_DataAccess::Copy, test_basis.DomainMap(), test_basis.NumGlobalCols());
    if (test_basis.RowMap().SameAs(M_global_e.RowMap()) && test_basis.NumGlobalRows() == M_global_e.NumGlobalRows()){
        Epetra_CrsMatrix epetra_reduced_lhs_tmp(Epetra_DataAccess::Copy, M_global_e.RowMap(), test_basis.NumGlobalCols());
        if (EpetraExt::MatrixMatrix::Multiply(M_global_e, false, test_basis, false, epetra_reduced_lhs_tmp) != 0){
            std::cerr << "Error in first Matrix Multiplication" << std::endl;
            return nullptr;
        }
        if (EpetraExt::MatrixMatrix::Multiply(trial_basis, true, epetra_reduced_lhs_tmp, false, lhs_matrix) != 0){
            std::cerr << "Error in second Matrix Multiplication" << std::endl;
            return nullptr;
        }
    } else {
        if(!(test_basis.RowMap().SameAs(M_global_e.RowMap()))){
            std::cerr << "Error: Inconsistent maps" << std::endl;
        } else {
            std::cerr << "Error: Inconsistent row sizes" << std::endl
            << "System: " << std::to_string(M_global_e.NumGlobalRows()) << std::endl
            << "Test: " << std::to_string(test_basis.NumGlobalRows()) << std::endl;
        }
    }
    //end of cell loop
    lhs_matrix.FillComplete(test_basis.DomainMap(),test_basis.DomainMap());
    std::ofstream lhs_file("lhs_matrix.txt");
    lhs_matrix.Print(lhs_file);
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