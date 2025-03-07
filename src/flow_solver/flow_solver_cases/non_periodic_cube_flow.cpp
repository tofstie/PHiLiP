#include "non_periodic_cube_flow.h"
#include "mesh/grids/non_periodic_cube.h"
#include <deal.II/grid/grid_generator.h>
#include "physics/physics_factory.h"

namespace PHiLiP {
namespace FlowSolver {

template <int dim, int nstate>
NonPeriodicCubeFlow<dim, nstate>::NonPeriodicCubeFlow(const PHiLiP::Parameters::AllParameters *const parameters_input)
    : CubeFlow_UniformGrid<dim, nstate>(parameters_input)
    , unsteady_data_table_filename_with_extension(this->all_param.flow_solver_param.unsteady_data_table_filename+".txt")
    , euler_physics (Physics::Euler<dim, dim+2, double>(
        &this->all_param,
        this->all_param.euler_param.ref_length,
        this->all_param.euler_param.gamma_gas,
        this->all_param.euler_param.mach_inf,
        this->all_param.euler_param.angle_of_attack,
        this->all_param.euler_param.side_slip_angle))
{
    //create the Physics object
    this->pde_physics = std::dynamic_pointer_cast<Physics::PhysicsBase<dim,nstate,double>>(
                Physics::PhysicsFactory<dim,nstate,double>::create_Physics(parameters_input));
}

template <int dim, int nstate>
std::shared_ptr<Triangulation> NonPeriodicCubeFlow<dim,nstate>::generate_grid() const
{
    std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation> (
    #if PHILIP_DIM!=1
                this->mpi_communicator
    #endif
        );

    bool use_number_mesh_refinements = false;
    if(this->all_param.flow_solver_param.number_of_mesh_refinements>0)
        use_number_mesh_refinements = true;
    
    const unsigned int number_of_refinements = use_number_mesh_refinements ? this->all_param.flow_solver_param.number_of_mesh_refinements 
                                                                           : log2(this->all_param.flow_solver_param.number_of_grid_elements_per_dimension);

    const double domain_left = this->all_param.flow_solver_param.grid_left_bound;
    const double domain_right = this->all_param.flow_solver_param.grid_right_bound;
    const bool colorize = true;
    
    int left_boundary_id = 9999;
    using flow_case_enum = Parameters::FlowSolverParam::FlowCaseType;
    flow_case_enum flow_case_type = this->all_param.flow_solver_param.flow_case_type;

    if (flow_case_type == flow_case_enum::sod_shock_tube
        || flow_case_type == flow_case_enum::leblanc_shock_tube) {
        left_boundary_id = 1001; 
    } else if (flow_case_type == flow_case_enum::shu_osher_problem) {
        left_boundary_id = 1004;
    } else if (flow_case_type == flow_case_enum::reflective_shock_tube) {
        left_boundary_id = 1001; // wall bc // Maybe moe this into if statement above, talk to Sruthi
    }


    Grids::non_periodic_cube<dim>(*grid, domain_left, domain_right, colorize, left_boundary_id);
    grid->refine_global(number_of_refinements);

    return grid;
}

template <int dim, int nstate>
void NonPeriodicCubeFlow<dim,nstate>::display_additional_flow_case_specific_parameters() const
{
    this->pcout << "- - Courant-Friedrichs-Lewy number: " << this->all_param.flow_solver_param.courant_friedrichs_lewy_number << std::endl;
}

template<int dim, int nstate>
void NonPeriodicCubeFlow<dim, nstate>::check_positivity_density(DGBase<dim, double>& dg)
{

    //create 1D solution polynomial basis functions and corresponding projection operator
    //to interpolate the solution to the quadrature nodes, and to project it back to the
    //modal coefficients.
    const unsigned int init_grid_degree = dg.max_grid_degree;
    const unsigned int poly_degree = this->all_param.flow_solver_param.poly_degree;
    //Constructor for the operators
    OPERATOR::basis_functions<dim, 2 * dim, double> soln_basis(1, poly_degree, init_grid_degree);
    OPERATOR::vol_projection_operator<dim, 2 * dim, double> soln_basis_projection_oper(1, dg.max_degree, init_grid_degree);


    // Build the oneD operator to perform interpolation/projection
    soln_basis.build_1D_volume_operator(dg.oneD_fe_collection_1state[poly_degree], dg.oneD_quadrature_collection[poly_degree]);
    soln_basis_projection_oper.build_1D_volume_operator(dg.oneD_fe_collection_1state[poly_degree], dg.oneD_quadrature_collection[poly_degree]);

    for (auto soln_cell = dg.dof_handler.begin_active(); soln_cell != dg.dof_handler.end(); ++soln_cell) {
        if (!soln_cell->is_locally_owned()) continue;


        std::vector<dealii::types::global_dof_index> current_dofs_indices;
        // Current reference element related to this physical cell
        const int i_fele = soln_cell->active_fe_index();
        const dealii::FESystem<dim, dim>& current_fe_ref = dg.fe_collection[i_fele];
        const int poly_degree = current_fe_ref.tensor_degree();

        const unsigned int n_dofs_curr_cell = current_fe_ref.n_dofs_per_cell();

        // Obtain the mapping from local dof indices to global dof indices
        current_dofs_indices.resize(n_dofs_curr_cell);
        soln_cell->get_dof_indices(current_dofs_indices);

        // Extract the local solution dofs in the cell from the global solution dofs
        std::array<std::vector<double>, nstate> soln_coeff;
        const unsigned int n_shape_fns = n_dofs_curr_cell / nstate;

        for (unsigned int istate = 0; istate < nstate; ++istate) {
            soln_coeff[istate].resize(n_shape_fns);
        }

        // Allocate solution dofs and set local max and min
        for (unsigned int idof = 0; idof < n_dofs_curr_cell; ++idof) {
            const unsigned int istate = dg.fe_collection[poly_degree].system_to_component_index(idof).first;
            const unsigned int ishape = dg.fe_collection[poly_degree].system_to_component_index(idof).second;
            soln_coeff[istate][ishape] = dg.solution[current_dofs_indices[idof]];
        }

        const unsigned int n_quad_pts = dg.volume_quadrature_collection[poly_degree].size();

        std::array<std::vector<double>, nstate> soln_at_q;

        // Interpolate solution dofs to quadrature pts.
        for (int istate = 0; istate < nstate; istate++) {
            soln_at_q[istate].resize(n_quad_pts);
            soln_basis.matrix_vector_mult_1D(soln_coeff[istate], soln_at_q[istate], soln_basis.oneD_vol_operator);
        }

        for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {
            // Verify that positivity of density is preserved
            if (soln_at_q[0][iquad] < 0 || (isnan(soln_at_q[0][iquad])) ) {
                std::cout << "Error: Density is negative or NaN - Aborting... " << std::endl << std::flush;
                const double current_time = 1;
                throw current_time;
            }
        }
    }
}

template <int dim, int nstate>
void NonPeriodicCubeFlow<dim, nstate>::compute_unsteady_data_and_write_to_table(
    const unsigned int current_iteration,
    const double current_time,
    const std::shared_ptr <DGBase<dim, double>> dg,
    const std::shared_ptr <dealii::TableHandler> unsteady_data_table)
{
    this->check_positivity_density(*dg);
    // All discrete proofs use solution nodes, therefore it is best to report
    // entropy on the solution nodes rather than by overintegrating.
    if (this->mpi_rank == 0) {

        unsteady_data_table->add_value("iteration", current_iteration);
        // Add values to data table
        if constexpr (nstate == dim+2) {
            const double current_numerical_entropy = this->compute_integrated_quantities(*dg, IntegratedQuantityEnum::numerical_entropy, 0); //do not overintegrate
            if (current_iteration==0) this->previous_numerical_entropy = current_numerical_entropy;
            const double entropy = current_numerical_entropy - previous_numerical_entropy;
            this->previous_numerical_entropy = current_numerical_entropy;
            const double kinetic_energy = this->compute_integrated_quantities(*dg, IntegratedQuantityEnum::kinetic_energy);
            this->add_value_to_data_table(current_time, "time", unsteady_data_table);
            this->add_value_to_data_table(entropy,"entropy",unsteady_data_table);
            unsteady_data_table->set_scientific("entropy", false);
            this->add_value_to_data_table(entropy/initial_entropy,"U/Uo",unsteady_data_table);
            unsteady_data_table->set_scientific("U/Uo", false);
            this->add_value_to_data_table(kinetic_energy,"kinetic_energy",unsteady_data_table);
            unsteady_data_table->set_scientific("kinetic_energy", false);
        }
        // Write to file
        std::ofstream unsteady_data_table_file(this->unsteady_data_table_filename_with_extension);
        unsteady_data_table->write_text(unsteady_data_table_file);
    }

    if (current_iteration % this->all_param.ode_solver_param.print_iteration_modulo == 0) {
        // Print to console
        this->pcout << "    Iter: " << current_iteration
            << "    Time: " << current_time;

        this->pcout << std::endl;
    }
}

template<int dim, int nstate>
double NonPeriodicCubeFlow<dim, nstate>::compute_integrated_quantities(DGBase<dim, double> &dg, IntegratedQuantityEnum quantity, const int overintegrate) const
{
#if PHILIP_DIM == 2
    double integrated_quantity = 0;
    if(quantity == IntegratedQuantityEnum::kinetic_energy) {
        const unsigned int poly_degree = dg.max_degree;
        integrated_quantity += poly_degree;
    }
    return integrated_quantity + overintegrate;
#else
    // Check that poly_degree is uniform everywhere
    if (dg.get_max_fe_degree() != dg.get_min_fe_degree()) {
        // Note: This function may have issues with nonuniform p. Should test in debug mode if developing in the future.
        this->pcout << "ERROR: compute_integrated_quantities() is untested for nonuniform p. Aborting..." << std::endl;
        std::abort();
    }

    double integrated_quantity = 0.0;

    const unsigned int grid_degree = dg.high_order_grid->fe_system.tensor_degree();
    const unsigned int poly_degree = dg.max_degree;

    // Set the quadrature of size dim and 1D for sum-factorization.
    dealii::Quadrature<1> quad_1D;
    std::vector<double> quad_weights_;
    unsigned int n_quad_pts_;
    if (overintegrate > 0) {
        dealii::QGauss<dim> quad_extra(dg.max_degree+1+overintegrate);
        dealii::QGauss<1> quad_extra_1D(dg.max_degree+1+overintegrate);
        quad_1D = quad_extra_1D;
        quad_weights_ = quad_extra.get_weights();
        n_quad_pts_ = quad_extra.size();
    } else {
        quad_1D = dg.oneD_quadrature_collection[poly_degree];
        quad_weights_ = dg.volume_quadrature_collection[poly_degree].get_weights();
        n_quad_pts_ = dg.volume_quadrature_collection[poly_degree].size();
    }
    const std::vector<double> &quad_weights = quad_weights_;
    const unsigned int n_quad_pts = n_quad_pts_;

    // Construct the basis functions and mapping shape functions.
    OPERATOR::basis_functions<dim,2*dim,double> soln_basis(1, poly_degree, grid_degree);
    OPERATOR::mapping_shape_functions<dim,2*dim,double> mapping_basis(1, poly_degree, grid_degree);
    // Build basis function volume operator and gradient operator from 1D finite element for 1 state.
    soln_basis.build_1D_volume_operator(dg.oneD_fe_collection_1state[poly_degree], quad_1D);
    soln_basis.build_1D_gradient_operator(dg.oneD_fe_collection_1state[poly_degree], quad_1D);
    // Build mapping shape functions operators using the oneD high_ordeR_grid finite element
    mapping_basis.build_1D_shape_functions_at_grid_nodes(dg.high_order_grid->oneD_fe_system, dg.high_order_grid->oneD_grid_nodes);
    mapping_basis.build_1D_shape_functions_at_flux_nodes(dg.high_order_grid->oneD_fe_system, quad_1D, dg.oneD_face_quadrature);
    // If in the future we need the physical quadrature node location, turn these flags to true and the constructor will
    // automatically compute it for you. Currently set to false as to not compute extra unused terms.
    const bool store_vol_flux_nodes = false;//currently doesn't need the volume physical nodal position
    const bool store_surf_flux_nodes = false;//currently doesn't need the surface physical nodal position

    const unsigned int n_dofs = dg.fe_collection[poly_degree].n_dofs_per_cell();
    const unsigned int n_shape_fns = n_dofs / nstate;
    std::vector<dealii::types::global_dof_index> dofs_indices (n_dofs);
    auto metric_cell = dg.high_order_grid->dof_handler_grid.begin_active();
    // Changed for loop to update metric_cell.
    for (auto cell = dg.dof_handler.begin_active(); cell!= dg.dof_handler.end(); ++cell, ++metric_cell) {
        if (!cell->is_locally_owned()) continue;
        //if (dg.reduced_mesh_weights[cell->active_cell_index()] == 0) continue;
        cell->get_dof_indices (dofs_indices);
        // We first need to extract the mapping support points (grid nodes) from high_order_grid.
        const dealii::FESystem<dim> &fe_metric = dg.high_order_grid->fe_system;
        const unsigned int n_metric_dofs = fe_metric.dofs_per_cell;
        const unsigned int n_grid_nodes  = n_metric_dofs / dim;
        std::vector<dealii::types::global_dof_index> metric_dof_indices(n_metric_dofs);
        metric_cell->get_dof_indices (metric_dof_indices);
        std::array<std::vector<double>,dim> mapping_support_points;
        for(int idim=0; idim<dim; idim++){
            mapping_support_points[idim].resize(n_grid_nodes);
        }
        // Get the mapping support points (physical grid nodes) from high_order_grid.
        // Store it in such a way we can use sum-factorization on it with the mapping basis functions.
        const std::vector<unsigned int > &index_renumbering = dealii::FETools::hierarchic_to_lexicographic_numbering<dim>(grid_degree);
        for (unsigned int idof = 0; idof< n_metric_dofs; ++idof) {
            const double val = (dg.high_order_grid->volume_nodes[metric_dof_indices[idof]]);
            const unsigned int istate = fe_metric.system_to_component_index(idof).first;
            const unsigned int ishape = fe_metric.system_to_component_index(idof).second;
            const unsigned int igrid_node = index_renumbering[ishape];
            mapping_support_points[istate][igrid_node] = val;
        }
        // Construct the metric operators.
        OPERATOR::metric_operators<double, dim, 2*dim> metric_oper(nstate, poly_degree, grid_degree, store_vol_flux_nodes, store_surf_flux_nodes);
        // Build the metric terms to compute the gradient and volume node positions.
        // This functions will compute the determinant of the metric Jacobian and metric cofactor matrix.
        // If flags store_vol_flux_nodes and store_surf_flux_nodes set as true it will also compute the physical quadrature positions.
        metric_oper.build_volume_metric_operators(
            n_quad_pts, n_grid_nodes,
            mapping_support_points,
            mapping_basis,
            dg.all_parameters->use_invariant_curl_form);

        // Fetch the modal soln coefficients
        // We immediately separate them by state as to be able to use sum-factorization
        // in the interpolation operator. If we left it by n_dofs_cell, then the matrix-vector
        // mult would sum the states at the quadrature point.
        // That is why the basis functions are based off the 1state oneD fe_collection.
        std::array<std::vector<double>,nstate> soln_coeff;
        for (unsigned int idof = 0; idof < n_dofs; ++idof) {
            const unsigned int istate = dg.fe_collection[poly_degree].system_to_component_index(idof).first;
            const unsigned int ishape = dg.fe_collection[poly_degree].system_to_component_index(idof).second;
            if(ishape == 0){
                soln_coeff[istate].resize(n_shape_fns);
            }

            soln_coeff[istate][ishape] = dg.solution(dofs_indices[idof]);
        }
        // Interpolate each state to the quadrature points using sum-factorization
        // with the basis functions in each reference direction.
        std::array<std::vector<double>,nstate> soln_at_q_vect;
        std::array<dealii::Tensor<1,dim,std::vector<double>>,nstate> soln_grad_at_q_vect;
        for(int istate=0; istate<nstate; istate++){
            soln_at_q_vect[istate].resize(n_quad_pts);
            // Interpolate soln coeff to volume cubature nodes.
            soln_basis.matrix_vector_mult_1D(soln_coeff[istate], soln_at_q_vect[istate],
                                             soln_basis.oneD_vol_operator);
            // We need to first compute the reference gradient of the solution, then transform that to a physical gradient.
            dealii::Tensor<1,dim,std::vector<double>> ref_gradient_basis_fns_times_soln;
            for(int idim=0; idim<dim; idim++){
                ref_gradient_basis_fns_times_soln[idim].resize(n_quad_pts);
                soln_grad_at_q_vect[istate][idim].resize(n_quad_pts);
            }
            // Apply gradient of reference basis functions on the solution at volume cubature nodes.
            soln_basis.gradient_matrix_vector_mult_1D(soln_coeff[istate], ref_gradient_basis_fns_times_soln,
                                                      soln_basis.oneD_vol_operator,
                                                      soln_basis.oneD_grad_operator);
            // Transform the reference gradient into a physical gradient operator.
            for(int idim=0; idim<dim; idim++){
                for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                    for(int jdim=0; jdim<dim; jdim++){
                        //transform into the physical gradient
                        soln_grad_at_q_vect[istate][idim][iquad] += metric_oper.metric_cofactor_vol[idim][jdim][iquad]
                                                                  * ref_gradient_basis_fns_times_soln[jdim][iquad]
                                                                  / metric_oper.det_Jac_vol[iquad];
                    }
                }
            }
        }

        // Loop over quadrature nodes, compute quantities to be integrated, and integrate them.
        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

            std::array<double,nstate> soln_at_q;
            std::array<dealii::Tensor<1,dim,double>,nstate> soln_grad_at_q;
            // Extract solution and gradient in a way that the physics ca n use them.
            for(int istate=0; istate<nstate; istate++){
                soln_at_q[istate] = soln_at_q_vect[istate][iquad];
                for(int idim=0; idim<dim; idim++){
                    soln_grad_at_q[istate][idim] = soln_grad_at_q_vect[istate][idim][iquad];
                }
            }

            //#####################################################################
            // Compute integrated quantities here
            //#####################################################################
            if (quantity == IntegratedQuantityEnum::kinetic_energy) {
                const double KE_integrand = this->euler_physics.compute_kinetic_energy_from_conservative_solution(soln_at_q);
                integrated_quantity += KE_integrand * quad_weights[iquad] * metric_oper.det_Jac_vol[iquad];
            } else if (quantity == IntegratedQuantityEnum::numerical_entropy) {
                const double quadrature_entropy = this->euler_physics.compute_numerical_entropy_function(soln_at_q);
                //Using std::cout because of cell->is_locally_owned check
                if (isnan(quadrature_entropy)){
                    std::cout << "WARNING: NaN entropy detected at a node!"  << std::endl;}
                integrated_quantity += quadrature_entropy * quad_weights[iquad] * metric_oper.det_Jac_vol[iquad] ; // Include Reduced Mesh Weights?
            } else if (quantity == IntegratedQuantityEnum::max_wave_speed) {
                const double local_wave_speed = this->euler_physics.max_convective_eigenvalue(soln_at_q);
                if(local_wave_speed > integrated_quantity) integrated_quantity = local_wave_speed;
            } else {
                std::cout << "Integrated quantity is not correctly defined." << std::endl;
            }
            //#####################################################################
        }
    }

    //MPI
    if (quantity == IntegratedQuantityEnum::max_wave_speed) {
        integrated_quantity = dealii::Utilities::MPI::max(integrated_quantity, this->mpi_communicator);
    } else {
        integrated_quantity = dealii::Utilities::MPI::sum(integrated_quantity, this->mpi_communicator);
    }

    return integrated_quantity;
#endif
}


template <int dim, int nstate>
double NonPeriodicCubeFlow<dim, nstate>::compute_entropy(
        const std::shared_ptr <DGBase<dim, double>> dg
) const {
    return compute_integrated_quantities(*dg, IntegratedQuantityEnum::numerical_entropy, 0);
}
#if PHILIP_DIM==2
    template class NonPeriodicCubeFlow<PHILIP_DIM, 1>;
#else
    template class NonPeriodicCubeFlow <PHILIP_DIM,PHILIP_DIM+2>;
#endif
} // FlowSolver namespace
} // PHiLiP namespace