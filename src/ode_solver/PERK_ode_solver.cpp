#include "PERK_ode_solver.h"


namespace PHiLiP {
namespace ODE {

template <int dim, typename real, int n_rk_stages, typename MeshType> 
PERKODESolver<dim,real,n_rk_stages, MeshType>::PERKODESolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input,
        std::shared_ptr<PERKTableauBase<dim,real,MeshType>> rk_tableau_input,
        std::shared_ptr<EmptyRRKBase<dim,real,MeshType>> RRK_object_input)
        : RungeKuttaBase<dim,real,n_rk_stages,MeshType>(dg_input, RRK_object_input)
        , butcher_tableau(rk_tableau_input)
{}

template<int dim, typename real, int n_rk_stages, typename MeshType>
void PERKODESolver<dim,real,n_rk_stages, MeshType>::calculate_stage_solution (int istage, real dt, const bool pseudotime)
{
    stage_solution = 0;
    for (std::size_t k = 0; k < this->group_ID.size(); ++k){ // calculate stage solutions corresponding to tableaus
        if (this->calc_stage[k][istage]==true){
            for (int j = 0; j < istage; ++j){
                if (this->butcher_tableau->get_a(istage,j, k+1) != 0){
                    stage_solution.add(dt * this->butcher_tableau->get_a(istage,j, k+1), this->rk_stage_k[k][j]); 
                }
            } //sum(a_ij *k_j), explicit part
            
            if(pseudotime) {
                const double CFL = dt;
                this->dg->time_scale_solution_update(this->rk_stage_k[k][istage], CFL);
            }
        }
    } 
    stage_solution.add(1.0, this->solution_update);

    this->dg->solution = stage_solution;
}

template<int dim, typename real, int n_rk_stages, typename MeshType>
void PERKODESolver<dim,real,n_rk_stages,MeshType>::calculate_stage_derivative (int istage, real dt)
{
     //set the DG current time for unsteady source terms
     this->dg->set_current_time(this->current_time + this->butcher_tableau->get_c(istage)*dt);

    for (size_t k = 0; k < this->group_ID.size(); ++k){
        if (this->calc_stage[k][istage]==true){
            this->dg->right_hand_side*=0;
            this->dg->assemble_residual(false, false, false, 0.0, this->group_ID[k]); //RHS : du/dt = RHS = F(u_n + dt* sum(a_ij*k_j) + dt * a_ii * u^(istage)))
            if(this->all_parameters->use_inverse_mass_on_the_fly){
                this->dg->apply_inverse_global_mass_matrix(this->dg->right_hand_side, this->rk_stage_k[k][istage]); //rk_stage[istage] = IMM*RHS = F(u_n + dt*sum(a_ij*k_j))
            } else{
                this->dg->global_inverse_mass_matrix.vmult(this->rk_stage_k[k][istage], this->dg->right_hand_side); //rk_stage[istage] = IMM*RHS = F(u_n + dt*sum(a_ij*k_j))
            }
         }
    }
}


template<int dim, typename real, int n_rk_stages, typename MeshType>
void PERKODESolver<dim,real,n_rk_stages,MeshType>::sum_stages (real dt, const bool pseudotime)
{
    //assemble solution from stages
    for (size_t k = 0; k < this->group_ID.size(); ++k){
            for (int istage = 0; istage < n_rk_stages; ++istage){
                if (pseudotime){
                    std::cout << "not implemented for pseudotime" << std::endl;
                    std::abort();
                } else {
                    if (this->calc_stage[k][istage]==true){
                        this->solution_update.add(dt* this->butcher_tableau->get_b(istage),this->rk_stage_k[k][istage]);
                    }
                }
            }
        }

}        


template<int dim, typename real, int n_rk_stages, typename MeshType>
void PERKODESolver<dim,real,n_rk_stages,MeshType>::apply_limiter ()
{
    // Apply limiter at every RK stage
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
}

template<int dim, typename real, int n_rk_stages, typename MeshType>
real PERKODESolver<dim,real,n_rk_stages,MeshType>::adjust_time_step (real dt)
{
    // Calculates relaxation parameter and modify the time step size as dt*=relaxation_parameter.
    // if not using RRK, the relaxation parameter will be set to 1, such that dt is not modified.
    this->relaxation_parameter_RRK_solver = this->relaxation_runge_kutta->update_relaxation_parameter(dt, this->dg, this->rk_stage, this->solution_update);
    dt *= this->relaxation_parameter_RRK_solver;
    this->modified_time_step = dt;
    return dt;
}

template <int dim, typename real, int n_rk_stages, typename MeshType> 
void PERKODESolver<dim,real,n_rk_stages,MeshType>::allocate_runge_kutta_system ()
{

    this->butcher_tableau->set_tableau();
    
    this->butcher_tableau_aii_is_zero.resize(n_rk_stages);
    std::fill(this->butcher_tableau_aii_is_zero.begin(),
              this->butcher_tableau_aii_is_zero.end(),
              false); 
    for (int istage=0; istage<n_rk_stages; ++istage) {
        if (this->butcher_tableau->get_a(istage,istage, 1)==0.0)     this->butcher_tableau_aii_is_zero[istage] = true;
    
    }
    if(this->all_parameters->use_inverse_mass_on_the_fly == false) {
        this->pcout << " evaluating inverse mass matrix..." << std::flush;
        this->dg->evaluate_mass_matrices(true); // creates and stores global inverse mass matrix
        //RRK needs both mass matrix and inverse mass matrix
        using ODEEnum = Parameters::ODESolverParam::ODESolverEnum;
        ODEEnum ode_type = this->ode_param.ode_solver_type;
        if (ode_type == ODEEnum::rrk_explicit_solver){
            this->dg->evaluate_mass_matrices(false); // creates and stores global mass matrix
        }
    }

    // store whether to calculate stage
    this->calc_stage.resize(this->group_ID.size());
    for (size_t k = 0; k < this->group_ID.size(); ++k) {
        this->calc_stage[k].resize(n_rk_stages);
        for (int j = 0; j < n_rk_stages; ++j) {
            bool calcStage = false;
            for (int i = 0; i < n_rk_stages; ++i) {
                if (this->butcher_tableau->get_a(i, j, k+1) != 0 || this->butcher_tableau->get_b(j) != 0) {
                    calcStage = true;
                    break;
                }
            }
            this->calc_stage[k][j] = calcStage;
        }
   }
   stage_solution.reinit(this->dg->solution);
}

template <int dim, typename real, int n_rk_stages, typename MeshType>
void PERKODESolver<dim, real, n_rk_stages, MeshType>::partition_scheme()
{
    // Allocate the variables for the partitioning
    using PartitionEnum = Parameters::ODESolverParam::PartitionTypeEnum;
    this->dg->assemble_residual();
    const std::size_t n_groups = this->group_ID.size();
    const unsigned int n_cells = this->dg->triangulation->n_active_cells();

    std::vector<dealii::LinearAlgebra::distributed::Vector<int>> local_locations_to_evaluate;
    std::vector<std::vector<int>> locations_rhs_vector;
    std::vector<std::vector<std::vector<int>>> all_cores_locations_to_evaluate;

    local_locations_to_evaluate.resize(n_groups);
    locations_rhs_vector.resize(n_groups);
    all_cores_locations_to_evaluate.resize(n_groups);

    for (std::size_t i = 0; i < n_groups; ++i) {
        local_locations_to_evaluate[i].reinit(this->dg->triangulation->n_active_cells());
        local_locations_to_evaluate[i] = 0;
    }
    for (std::size_t i = 0; i < n_groups; ++i) {
        locations_rhs_vector[i].resize(this->dg->triangulation->n_active_cells());
    }
    // Partitioning Cells
    if (this->ode_param.partition_type == PartitionEnum::cell_size)
    {
        cell_size_partition(n_groups, local_locations_to_evaluate);
    }
    else if (this->ode_param.partition_type == PartitionEnum::cell_number)
    {
        cell_number_partition(n_groups, local_locations_to_evaluate);
    }
    else
    {
        this->pcout << "Please specify a partition type when using PERK schemes" << std::endl;
        std::abort();
    }

    std::vector<unsigned int> indices(n_cells);
    std::iota(indices.begin(), indices.end(), 0);

    for (std::size_t i = 0; i < n_groups; ++i){
        // Copy dealii vector to std vector for all_gather
        for (std::size_t t = 0; t < n_cells; ++t){
            locations_rhs_vector[i][t] = local_locations_to_evaluate[i][t];
        }
        all_cores_locations_to_evaluate[i] = dealii::Utilities::MPI::all_gather(MPI_COMM_WORLD, locations_rhs_vector[i]);
        // Copy the data from other cores into the cells. The data has to be the same on every core
        const std::size_t n_ranks = all_cores_locations_to_evaluate[i].size();
        for (std::size_t idx = 0; idx < n_ranks; ++idx){
            if (idx != static_cast<std::size_t>(this->mpi_rank))
                local_locations_to_evaluate[i].add(indices, all_cores_locations_to_evaluate[i][idx]);
        }
        local_locations_to_evaluate[i].compress(dealii::VectorOperation::insert);
        local_locations_to_evaluate[i].update_ghost_values();
        this->dg->set_list_of_cell_group_IDs(local_locations_to_evaluate[i], this->group_ID[i]);
    }

}
template <int dim, typename real, int n_rk_stages, typename MeshType>
void PERKODESolver<dim, real, n_rk_stages, MeshType>::cell_size_partition(
    const std::size_t n_groups,
    std::vector<dealii::LinearAlgebra::distributed::Vector<int>> &local_locations_to_evaluate
)
{
    const double local_max = this->dg->cell_volume.linfty_norm();
    const double max_cell_volume = dealii::Utilities::MPI::max(local_max, this->mpi_communicator);

    for (typename dealii::DoFHandler<dim>::active_cell_iterator cell = this->dg->dof_handler.begin_active(); cell != this->dg->dof_handler.end(); ++cell) {
        if (!cell->is_locally_owned())
            continue;
        const double vol = this->dg->cell_volume[cell->active_cell_index()];
        for (std::size_t i = 0; i < n_groups; ++i) {
            if (vol >= 0.05 * max_cell_volume && i == 0) {
                local_locations_to_evaluate[i](cell->active_cell_index()) = 1;
            } else if (i == 1 && vol >= 0.005 * max_cell_volume && vol < 0.05 * max_cell_volume) {
                local_locations_to_evaluate[i](cell->active_cell_index()) = 1;
            } else if (i == 2 && vol >= 0.0001 * max_cell_volume && vol < 0.005 * max_cell_volume) {
                local_locations_to_evaluate[i](cell->active_cell_index()) = 1;
            } else if (i == 3 && vol >= 0.000005 * max_cell_volume && vol < 0.0001 * max_cell_volume) {
                local_locations_to_evaluate[i](cell->active_cell_index()) = 1;
            } else if (i == 4 && vol >= 0.0000006 * max_cell_volume && vol < 0.000005 * max_cell_volume) {
                local_locations_to_evaluate[i](cell->active_cell_index()) = 1;
            } else if (i == 5 && vol < 0.0000006 * max_cell_volume) {
                local_locations_to_evaluate[i](cell->active_cell_index()) = 1;
            }
        }
    }
}
template <int dim, typename real, int n_rk_stages, typename MeshType>
void PERKODESolver<dim, real, n_rk_stages, MeshType>::cell_number_partition(
const std::size_t n_groups,
std::vector<dealii::LinearAlgebra::distributed::Vector<int>> &local_locations_to_evaluate )
{

    const int evaluate_until_this_index = local_locations_to_evaluate.size() / 2;
    const int index_remainder = local_locations_to_evaluate.size() % 2;
    int curr_idx = 0;
    for (std::size_t group_idx = 0; group_idx < n_groups; ++group_idx)
    {
        for (int i = curr_idx; i < curr_idx + evaluate_until_this_index; ++i){
            if (local_locations_to_evaluate[group_idx].in_local_range(i))
                local_locations_to_evaluate[group_idx](i) = 1;
        }
        curr_idx += evaluate_until_this_index;
    }
    for (int i = 0; i < index_remainder; ++i)
    {
        if (local_locations_to_evaluate[n_groups - 1].in_local_range(i+curr_idx))
            local_locations_to_evaluate[n_groups - 1](i+curr_idx);
    }
}
template class PERKODESolver<PHILIP_DIM, double,10, dealii::Triangulation<PHILIP_DIM> >;
template class PERKODESolver<PHILIP_DIM, double,10, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
template class PERKODESolver<PHILIP_DIM, double,16, dealii::Triangulation<PHILIP_DIM> >;
template class PERKODESolver<PHILIP_DIM, double,16, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
#if PHILIP_DIM != 1
    template class PERKODESolver<PHILIP_DIM, double,10, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class PERKODESolver<PHILIP_DIM, double,16, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
#endif


} // ODESolver namespace
} // PHiLiP namespace