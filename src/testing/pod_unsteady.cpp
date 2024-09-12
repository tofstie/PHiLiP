#include "pod_unsteady.h"
#include <iostream>
#include <filesystem>
#include "functional/functional.h"
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/vector_operation.h>
#include "reduced_order/reduced_order_solution.h"
#include "flow_solver/flow_solver.h"
#include "flow_solver/flow_solver_factory.h"
#include "physics/euler.h"
#include <cmath>
#include "operators/operators.h"
#include "reduced_order/rbf_interpolation.h"
#include "ROL_Algorithm.hpp"
#include "ROL_LineSearchStep.hpp"
#include "ROL_StatusTest.hpp"
#include "ROL_Stream.hpp"
#include "ROL_Bounds.hpp"
#include "reduced_order/halton.h"
#include "reduced_order/min_max_scaler.h"
#include <deal.II/base/timer.h>
#include "reduced_order/assemble_greedy_residual.h"
#include "tests.h"

// FOR TESTING TYPES
#include <typeinfo>
#include <limits>



namespace PHiLiP {
namespace Tests {

template<int dim, int nstate>
PODUnsteady<dim,nstate>::PODUnsteady(
    const PHiLiP::Parameters::AllParameters *const parameters_input,
    const dealii::ParameterHandler &parameter_handler_input)
    : TestsBase::TestsBase(parameters_input)
    , parameter_handler(parameter_handler_input)
    , flow_solver(FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(all_parameters, parameter_handler))
    , all_param(*parameters_input)
    , flow_solver_param(all_param.flow_solver_param)
    , ode_param(all_param.ode_solver_param)
    , do_output_solution_at_fixed_times(ode_param.output_solution_at_fixed_times)
    , number_of_fixed_times_to_output_solution(ode_param.number_of_fixed_times_to_output_solution)
    , output_solution_at_exact_fixed_times(ode_param.output_solution_at_exact_fixed_times)
    , final_time(flow_solver_param.final_time)
    , output_snapshot_every_x_timesteps(all_param.reduced_order_param.output_snapshot_every_x_timesteps)
    {
        //const bool compute_dRdW = true;
        //flow_solver->dg->assemble_residual(compute_dRdW);
        if(all_parameters->reduced_order_param.path_to_search == "."){
            //const bool compute_dRdW = true;
            //flow_solver->dg->assemble_residual(compute_dRdW);
            std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> system_matrix = std::make_shared<dealii::TrilinosWrappers::SparseMatrix>();
            system_matrix->copy_from(flow_solver->dg->system_matrix);
            current_pod = std::make_shared<ProperOrthogonalDecomposition::OnlinePOD<dim>>(system_matrix);
        } else {
            offline_pod = flow_solver->ode_solver->pod;
        }
        if(false){ // Am I doing hyper reduction?!
            auto ode_solver_type = ode_param.ode_solver_type;
            HyperReduction::AssembleGreedyRes<dim,nstate> hyper_reduction(&all_param, parameter_handler, flow_solver->dg, offline_pod, ode_solver_type);
            hyper_reduction.build_weights();
            hyper_reduction.build_initial_target();
            hyper_reduction.build_problem();
        }
    }

template <int dim, int nstate>
int PODUnsteady<dim,nstate>
::run_test ()     
//    : all_param(*parameters_input)
//    , flow_solver_param(all_param.flow_solver_param)
//    , ode_param(all_param.ode_solver_param)
//    , do_output_solution_at_fixed_times(ode_param.output_solution_at_fixed_times)
//    , number_of_fixed_times_to_output_solution(ode_param.number_of_fixed_times_to_output_solution)
//    , output_solution_at_exact_fixed_times(ode_param.output_solution_at_exact_fixed_times):
const {
    int number_of_timesteps = 0;
    int iteration = 0;
    if(all_parameters->reduced_order_param.entropy_varibles_in_snapshots){
        std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> pod_basis = offline_pod->getPODBasis();
        flow_solver->dg->calculate_projection_matrix(*pod_basis);
    }
    dealii::LinearAlgebra::distributed::Vector<double> entropy_snapshots(flow_solver->dg->solution);
    dealii::LinearAlgebra::distributed::Vector<double> conservative_snapshots(flow_solver->dg->solution);
    dealii::QGauss<dim> quad_extra(flow_solver->dg->max_degree);
    //const unsigned int n_quad_pts  = flow_solver->dg->volume_quadrature_collection[flow_solver_param.poly_degree].size();
    //const unsigned int n_dofs_cell = flow_solver->dg->fe_collection[flow_solver_param.poly_degree].dofs_per_cell;
    //const unsigned int n_shape_fns = n_dofs_cell / nstate; 
    
    const Parameters::AllParameters::Flux_Reconstruction FR_Type = all_param.flux_reconstruction_type;
    // Build Operator Basis
    OPERATOR::basis_functions<dim,2*dim,double> soln_basis(1, flow_solver->dg->max_degree, flow_solver->dg->high_order_grid->fe_system.tensor_degree());
    OPERATOR::vol_projection_operator_FR<dim,2*dim,double> soln_basis_projection_oper(1,flow_solver->dg->max_degree, flow_solver->dg->high_order_grid->fe_system.tensor_degree(),
                                                                                   FR_Type, true);
    // Build Volume Operators
    soln_basis.build_1D_volume_operator(flow_solver->dg->oneD_fe_collection_1state[flow_solver->dg->max_degree], flow_solver->dg->oneD_quadrature_collection[flow_solver->dg->max_degree]);
    soln_basis.build_1D_gradient_operator(flow_solver->dg->oneD_fe_collection_1state[flow_solver->dg->max_degree], flow_solver->dg->oneD_quadrature_collection[flow_solver->dg->max_degree]);
    soln_basis.build_1D_surface_operator(flow_solver->dg->oneD_fe_collection_1state[flow_solver->dg->max_degree], flow_solver->dg->oneD_face_quadrature);
    soln_basis.build_1D_surface_gradient_operator(flow_solver->dg->oneD_fe_collection_1state[flow_solver->dg->max_degree], flow_solver->dg->oneD_face_quadrature);

    soln_basis_projection_oper.build_1D_volume_operator(flow_solver->dg->oneD_fe_collection_1state[flow_solver->dg->max_degree], flow_solver->dg->oneD_quadrature_collection[flow_solver->dg->max_degree]);
    // Build Physics Object
    Physics::Euler<dim,nstate,double> euler_physics_double
    = Physics::Euler<dim, nstate, double>(
            this->all_parameters,
            all_param.euler_param.ref_length,
            all_param.euler_param.gamma_gas,
            all_param.euler_param.mach_inf,
            all_param.euler_param.angle_of_attack,
            all_param.euler_param.side_slip_angle);
    dealii::Table<1,double> output_solution_fixed_times;
    flow_solver->dg->evaluate_mass_matrices();
    // Full Order flow solver (For L2 Errors)
    
    Parameters::AllParameters FOM_param = *(TestsBase::all_parameters);
    using ODESolverEnum = Parameters::ODESolverParam::ODESolverEnum;
    using OutputEnum = Parameters::OutputEnum;
    dealii::ParameterHandler dummy_handler;
    FOM_param.ode_solver_param.ode_solver_type = ODESolverEnum::runge_kutta_solver;
    FOM_param.ode_solver_param.ode_output = OutputEnum::quiet;
    FOM_param.reduced_order_param.entropy_varibles_in_snapshots = false;
    const Parameters::AllParameters FOM_param_const = FOM_param;
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> FOM_flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&FOM_param_const, dummy_handler);
    FOM_flow_solver->dg->allocate_system(true, false, false);
    // Outputting locations order
    dealii::LinearAlgebra::distributed::Vector<double> location_x;
    dealii::LinearAlgebra::distributed::Vector<double> location_y;
    flow_solver->dg->location2D(location_x,location_y);
    dealii::LinearAlgebra::ReadWriteVector<double> read_x(location_x.size());
    dealii::LinearAlgebra::ReadWriteVector<double> read_y(location_y.size());
    read_x.import(location_x, dealii::VectorOperation::values::insert);
    read_y.import(location_y, dealii::VectorOperation::values::insert);
    // Converting to dealii LAPACK
    dealii::LAPACKFullMatrix<double> dealiiLocation;
    dealiiLocation.reinit(location_x.size(),2);
    for(unsigned int row = 0;row < location_x.size(); row++){
        dealiiLocation.set(row,0,read_x(row));
        dealiiLocation.set(row,1,read_y(row));
    }
    // Saving to Files
    std::ofstream location_file("location.txt");
    unsigned int precision = 16;
    dealiiLocation.print_formatted(location_file, precision, true,0,"0"); // Added fix to 0?
    location_file.close();
    // For outputting solution at fixed times
    if(do_output_solution_at_fixed_times && (number_of_fixed_times_to_output_solution > 0)) {
        output_solution_fixed_times.reinit(number_of_fixed_times_to_output_solution);
        
        // Get output_solution_fixed_times from string
        const std::string output_solution_fixed_times_string = ode_param.output_solution_fixed_times_string;
        std::string line = output_solution_fixed_times_string;
        std::string::size_type sz1;
        output_solution_fixed_times[0] = std::stod(line,&sz1);
        for(unsigned int i=1; i<number_of_fixed_times_to_output_solution; ++i) {
            line = line.substr(sz1);
            sz1 = 0;
            output_solution_fixed_times[i] = std::stod(line,&sz1);
        }
    }


    try{
        // Index of current desired fixed time to output solution
        unsigned int index_of_current_desired_fixed_time_to_output_solution = 0;
        pcout << "Running Flow Solver..." << std::endl;
            // RESTART FILES
        // determine index_of_current_desired_fixed_time_to_output_solution if restarting solution
        if(flow_solver_param.restart_computation_from_file == true) {
            // use current_time to determine if restarting the computation from a non-zero initial time
            for(unsigned int i=0; i<number_of_fixed_times_to_output_solution; ++i) {
                if(flow_solver->ode_solver->current_time < output_solution_fixed_times[i]) {
                    index_of_current_desired_fixed_time_to_output_solution = i;
                    break;
                }
            }
        }
        //----------------------------------------------------
        // dealii::TableHandler and data at initial time
        //----------------------------------------------------
        std::shared_ptr<dealii::TableHandler> unsteady_data_table = std::make_shared<dealii::TableHandler>();
        std::shared_ptr<dealii::TableHandler> L2error_data_table = std::make_shared<dealii::TableHandler>();
        // no restart:
        if(flow_solver_param.restart_computation_from_file == true) {
            pcout << "Initializing data table from corresponding restart file... " << std::flush;
            const std::string restart_filename_without_extension = flow_solver->get_restart_filename_without_extension(flow_solver_param.restart_file_index);
            const std::string restart_unsteady_data_table_filename = flow_solver_param.unsteady_data_table_filename+std::string("-")+restart_filename_without_extension+std::string(".txt");
            flow_solver->initialize_data_table_from_file(flow_solver_param.restart_files_directory_name + std::string("/") + restart_unsteady_data_table_filename,unsteady_data_table);
            pcout << "done." << std::endl;
        } else {
            pcout << "Writing unsteady data computed at initial time... " << std::endl;
            flow_solver->flow_solver_case->compute_unsteady_data_and_write_to_table(flow_solver->ode_solver, flow_solver->dg, unsteady_data_table);
            pcout << "done." << std::endl;
        }
        // Output Initial VTK
        if(flow_solver_param.restart_computation_from_file == false) {
            if (ode_param.output_solution_every_x_steps > 0) {
                pcout << "  ... Writing vtk solution file at initial time ..." << std::endl;
                flow_solver->dg->output_results_vtk(flow_solver->ode_solver->current_iteration);
            } else if (ode_param.output_solution_every_dt_time_intervals > 0.0) {
                pcout << "  ... Writing vtk solution file at initial time ..." << std::endl;
                flow_solver->dg->output_results_vtk(flow_solver->ode_solver->current_iteration);
                flow_solver->ode_solver->current_desired_time_for_output_solution_every_dt_time_intervals += ode_param.output_solution_every_dt_time_intervals;
            } else if (do_output_solution_at_fixed_times && (number_of_fixed_times_to_output_solution > 0)) {
                pcout << "  ... Writing vtk solution file at initial time ..." << std::endl;
                flow_solver->dg->output_results_vtk(flow_solver->ode_solver->current_iteration);
            }
        }
        // Time step
        double time_step = 0.0;
        if(flow_solver_param.adaptive_time_step == true) {
            pcout << "Setting initial adaptive time step... " << std::flush;
            time_step = flow_solver->flow_solver_case->get_adaptive_time_step_initial(flow_solver->dg);
        } else {
            pcout << "Setting constant time step... " << std::flush;
            time_step = flow_solver->flow_solver_case->get_constant_time_step(flow_solver->dg);
        }
        #if PHILIP_DIM>1
            double current_desired_time_for_output_restart_files_every_dt_time_intervals = flow_solver->ode_solver->current_time; // when used, same as the initial time
        #endif
        if(flow_solver_param.restart_computation_from_file == true) {
            const double restart_time_step = ode_param.initial_time_step;
            if(std::abs(time_step-restart_time_step) > 1E-13) {
                pcout << "WARNING: Computed initial time step does not match value in restart parameter file within the tolerance. "
                        << "Diff is: " << std::abs(time_step-restart_time_step) << std::endl;
            }
        }
        flow_solver->flow_solver_case->set_time_step(time_step);
        if(all_parameters->reduced_order_param.path_to_search == "."){
             current_pod->addSnapshot(flow_solver->dg->solution);
        } else {
            CalculateL2Error(L2error_data_table,*FOM_flow_solver,euler_physics_double,iteration);
        }
        double next_time_step = time_step;
        pcout << "Advancing solution in time... " << std::endl;
        pcout << "Timer starting." << std::endl;
        dealii::Timer timer(this->mpi_communicator,false);
        timer.start();
        while(flow_solver->ode_solver->current_time < final_time)
        {
            number_of_timesteps++;
            time_step = next_time_step; // update time step
            // check if we need to decrease the time step
            if((flow_solver->ode_solver->current_time+time_step) > final_time && flow_solver_param.end_exactly_at_final_time) {
                // decrease time step to finish exactly at specified final time
                time_step = final_time - flow_solver->ode_solver->current_time;
            } else if (output_solution_at_exact_fixed_times && (do_output_solution_at_fixed_times && (number_of_fixed_times_to_output_solution > 0))) { // change this to some parameter
                const double next_time = flow_solver->ode_solver->current_time + time_step;
                const double desired_time = output_solution_fixed_times[index_of_current_desired_fixed_time_to_output_solution];
                // Check if current time is an output time
                const bool is_output_time = ((flow_solver->ode_solver->current_time<desired_time) && (next_time>desired_time));
                if(is_output_time) time_step = desired_time - flow_solver->ode_solver->current_time;
            }

            // update time step in flow_solver->flow_solver_case
            flow_solver->flow_solver_case->set_time_step(time_step);
            // advance solution
            //Online
            flow_solver->ode_solver->step_in_time(time_step,false); // pseudotime==false

            // Compute the unsteady quantities, write to the dealii table, and output to file
            flow_solver->flow_solver_case->compute_unsteady_data_and_write_to_table(flow_solver->ode_solver, flow_solver->dg, unsteady_data_table);
            // update next time step
            if(flow_solver_param.adaptive_time_step == true) {
                next_time_step = flow_solver->flow_solver_case->get_adaptive_time_step(flow_solver->dg);
            } else {
                next_time_step = flow_solver->flow_solver_case->get_constant_time_step(flow_solver->dg);
            }


#if PHILIP_DIM>1
            if(flow_solver_param.output_restart_files == true) {
                // Output restart files
                if(flow_solver_param.output_restart_files_every_dt_time_intervals > 0.0) {
                    const bool is_output_time = ((flow_solver->ode_solver->current_time <= current_desired_time_for_output_restart_files_every_dt_time_intervals) && 
                                                 ((flow_solver->ode_solver->current_time + next_time_step) > current_desired_time_for_output_restart_files_every_dt_time_intervals));
                    if (is_output_time) {
                        const unsigned int file_number = int(round(current_desired_time_for_output_restart_files_every_dt_time_intervals / flow_solver_param.output_restart_files_every_dt_time_intervals));
                        flow_solver->output_restart_files(file_number, next_time_step, unsteady_data_table);
                        current_desired_time_for_output_restart_files_every_dt_time_intervals += flow_solver_param.output_restart_files_every_dt_time_intervals;
                    }
                } else /*if (flow_solver_param.output_restart_files_every_x_steps > 0)*/ {
                    const bool is_output_iteration = (flow_solver->ode_solver->current_iteration % flow_solver_param.output_restart_files_every_x_steps == 0);
                    if (is_output_iteration) {
                        const unsigned int file_number = flow_solver->ode_solver->current_iteration / flow_solver_param.output_restart_files_every_x_steps;
                        flow_solver->output_restart_files(file_number, next_time_step, unsteady_data_table);
                    }
                }
            }
#endif
            // Output vtk solution files for post-processing in Paraview
            if (ode_param.output_solution_every_x_steps > 0) {
                const bool is_output_iteration = (flow_solver->ode_solver->current_iteration % ode_param.output_solution_every_x_steps == 0);
                if (is_output_iteration) {
                    pcout << "  ... Writing vtk solution file ..." << std::endl;
                    const unsigned int file_number = flow_solver->ode_solver->current_iteration / ode_param.output_solution_every_x_steps;
                    flow_solver->dg->output_results_vtk(file_number,flow_solver->ode_solver->current_time);
                }
            } else if(ode_param.output_solution_every_dt_time_intervals > 0.0) {
                const bool is_output_time = ((flow_solver->ode_solver->current_time <= flow_solver->ode_solver->current_desired_time_for_output_solution_every_dt_time_intervals) && 
                                             ((flow_solver->ode_solver->current_time + next_time_step) > flow_solver->ode_solver->current_desired_time_for_output_solution_every_dt_time_intervals));
                if (is_output_time) {
                    pcout << "  ... Writing vtk solution file ..." << std::endl;
                    const unsigned int file_number = int(round(flow_solver->ode_solver->current_desired_time_for_output_solution_every_dt_time_intervals / ode_param.output_solution_every_dt_time_intervals));
                    flow_solver->dg->output_results_vtk(file_number,flow_solver->ode_solver->current_time);
                    flow_solver->ode_solver->current_desired_time_for_output_solution_every_dt_time_intervals += ode_param.output_solution_every_dt_time_intervals;
                }
            } else if (do_output_solution_at_fixed_times && (number_of_fixed_times_to_output_solution > 0)) {
                const double next_time = flow_solver->ode_solver->current_time + next_time_step;
                const double desired_time = output_solution_fixed_times[index_of_current_desired_fixed_time_to_output_solution];
                // Check if current time is an output time
                bool is_output_time = false; // default initialization
                if(output_solution_at_exact_fixed_times) {
                    is_output_time = flow_solver->ode_solver->current_time == desired_time;
                } else {
                    is_output_time = ((flow_solver->ode_solver->current_time<=desired_time) && (next_time>desired_time));
                }
                if(is_output_time) {
                    pcout << "  ... Writing vtk solution file ..." << std::endl;
                    const int file_number = index_of_current_desired_fixed_time_to_output_solution+1; // +1 because initial time is 0
                    flow_solver->dg->output_results_vtk(file_number,flow_solver->ode_solver->current_time);
                    
                    // Update index s.t. it never goes out of bounds
                    if(index_of_current_desired_fixed_time_to_output_solution 
                        < (number_of_fixed_times_to_output_solution-1)) {
                        index_of_current_desired_fixed_time_to_output_solution += 1;
                    }
                }
            }
            // Outputing Snapshots
            if(number_of_timesteps == output_snapshot_every_x_timesteps){
                number_of_timesteps = 0;
                iteration++;
               if(all_parameters->reduced_order_param.path_to_search == "."){
                    current_pod->addSnapshot(flow_solver->dg->solution);
                    pcout << "Outputed Snapshot Data" << std::endl;
                } else { // 
                    CalculateL2Error(L2error_data_table,*FOM_flow_solver,euler_physics_double,iteration);
                }
                
            }
        } // closing while loop
        timer.stop();
        if(all_parameters->reduced_order_param.path_to_search == "."){
            outputSnapshotData(iteration);
        }
        pcout << "Timer stopped. " << std::endl;
        const double max_wall_time = dealii::Utilities::MPI::max(timer.wall_time(), this->mpi_communicator);
        pcout << "Elapsed wall time (mpi max): " << max_wall_time << " seconds." << std::endl;
        pcout << "Elapsed CPU time: " << timer.cpu_time() << " seconds." << std::endl;
        
    }catch(double end) {
        pcout << "Simulation did not reach the end time. Crashed at t = " << end <<std::endl;
    }
    return 0;
};

template <int dim, int nstate>
void PODUnsteady<dim,nstate>
::outputSnapshotData(int iteration) const {
    std::unique_ptr<dealii::TableHandler> snapshot_table = std::make_unique<dealii::TableHandler>();
    std::ofstream solution_out_file("solution_snapshots_iteration_" +  std::to_string(iteration) + ".txt");
    unsigned int precision = 16;
    current_pod->dealiiSnapshotMatrix.print_formatted(solution_out_file, precision, true,0,"0"); // Added fix to 0?
    solution_out_file.close();
};


// 📢 Stick to naming convention
template <int dim, int nstate>
std::array<std::vector<double>,4> PODUnsteady<dim,nstate>
::compute_quantities(DGBase<dim, double> &dg, Physics::Euler<dim,dim+2,double> euler_physics) const {

    // Code here a block to compute soln at q for the snapshot and dg solution
    std::array<std::vector<double>,4> l2_values;
    //std::fill(l2_values.begin(), l2_values.end(), 0.0);
    // Overintegrate the error to make sure there is not integration error in the error estimate
    int overintegrate = 0;
    // Set the quadrature of size dim and 1D for sum-factorization.
    dealii::QGauss<dim> quad_extra(dg.max_degree+1+overintegrate);
    dealii::QGauss<1> quad_extra_1D(dg.max_degree+1+overintegrate);

    const unsigned int n_quad_pts = quad_extra.size();
    const unsigned int grid_degree = dg.high_order_grid->fe_system.tensor_degree();
    const unsigned int poly_degree = dg.max_degree;

    OPERATOR::basis_functions<dim,2*dim,double> soln_basis(1, poly_degree, grid_degree); 
    OPERATOR::mapping_shape_functions<dim,2*dim,double> mapping_basis(1, poly_degree, grid_degree);

    soln_basis.build_1D_volume_operator(dg.oneD_fe_collection_1state[poly_degree], quad_extra_1D);
    soln_basis.build_1D_gradient_operator(dg.oneD_fe_collection_1state[poly_degree], quad_extra_1D);
    mapping_basis.build_1D_shape_functions_at_grid_nodes(dg.high_order_grid->oneD_fe_system, dg.high_order_grid->oneD_grid_nodes);
    mapping_basis.build_1D_shape_functions_at_flux_nodes(dg.high_order_grid->oneD_fe_system, quad_extra_1D, dg.oneD_face_quadrature);

    const bool store_vol_flux_nodes = false;//currently doesn't need the volume physical nodal position
    const bool store_surf_flux_nodes = false;//currently doesn't need the surface physical nodal position

    const unsigned int n_dofs = dg.fe_collection[poly_degree].n_dofs_per_cell();
    const unsigned int n_shape_fns = n_dofs / nstate;
    std::vector<dealii::types::global_dof_index> dofs_indices (n_dofs);
    auto metric_cell = dg.high_order_grid->dof_handler_grid.begin_active();
    for (auto cell = dg.dof_handler.begin_active(); cell!= dg.dof_handler.end(); ++cell, ++metric_cell) {
        if (!cell->is_locally_owned()) continue;
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
            // Apply gradient of reference basis functions on the solution at volume cubature nodes.}
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

            double const pressure = euler_physics.compute_pressure(soln_at_q);
            double entropy = euler_physics.compute_entropy_measure(soln_at_q);
            double KE = euler_physics.compute_kinetic_energy_from_conservative_solution(soln_at_q);
            l2_values[0].push_back(soln_at_q[0]);
            l2_values[1].push_back(pressure);
            l2_values[2].push_back(KE);
            l2_values[3].push_back(entropy);

        }
    }
    return l2_values;
}

template <int dim, int nstate>
std::array<double,2> PODUnsteady<dim, nstate>
::integrate_quantities(DGBase<dim, double> &dg, Physics::Euler<dim,dim+2,double> euler_physics) const{ 
    std::array<double,2> integral_values;
    std::fill(integral_values.begin(), integral_values.end(), 0.0);
    std::array<double,2> integrated_values;
    std::fill(integrated_values.begin(), integrated_values.end(), 0.0);
        // Code here a block to compute soln at q for the snapshot and dg solution
    std::array<std::vector<double>,4> l2_values;
    //std::fill(l2_values.begin(), l2_values.end(), 0.0);
    // Overintegrate the error to make sure there is not integration error in the error estimate
    int overintegrate = 0;

    const unsigned int poly_degree = dg.max_degree;
    const unsigned int grid_degree = dg.high_order_grid->fe_system.tensor_degree();

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
    
    //const double domain_size = pow(dg.all_parameters->flow_solver_param.grid_right_bound - dg.all_parameters->flow_solver_param.grid_left_bound,dim);
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
    for (auto cell = dg.dof_handler.begin_active(); cell!= dg.dof_handler.end(); ++cell, ++metric_cell) {
        if (!cell->is_locally_owned()) continue;
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
            // Apply gradient of reference basis functions on the solution at volume cubature nodes.}
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


            //double const pressure = euler_physics.compute_pressure(soln_at_q);
            integral_values[0] += euler_physics.compute_kinetic_energy_from_conservative_solution(soln_at_q)* quad_weights[iquad] * metric_oper.det_Jac_vol[iquad];;
            integral_values[1] += euler_physics.compute_numerical_entropy_function(soln_at_q)* quad_weights[iquad] * metric_oper.det_Jac_vol[iquad];
        }
    }
    for(int i_quantity=0; i_quantity<2; ++i_quantity) {
        integrated_values[i_quantity] = dealii::Utilities::MPI::sum(integral_values[i_quantity], this->mpi_communicator);
    }
    
    return integrated_values;
}


template <int dim, int nstate>
void PODUnsteady<dim, nstate>
::CalculateL2Error(std::shared_ptr <dealii::TableHandler> L2error_data_table,
                   FlowSolver::FlowSolver<dim,nstate> FOM_flow_solver,
                   Physics::Euler<dim,dim+2,double> euler_physics_double,
                   int iteration) const{
    const double current_time = flow_solver->ode_solver->current_time;
    Eigen::MatrixXd snapshotMatrix = offline_pod->getSnapshotMatrix();
    dealii::LinearAlgebra::distributed::Vector<double> FOM_solution(flow_solver->dg->solution);
    for(int m = 0; m < snapshotMatrix.rows();m++){
        if(FOM_solution.in_local_range(m)){
            FOM_solution(m) = snapshotMatrix(m,iteration);
        }
    }
    FOM_flow_solver.dg->solution = FOM_solution;

    std::array<std::vector<double>,4> FOM_quantities;
    FOM_quantities = compute_quantities(*FOM_flow_solver.dg,euler_physics_double);
    std::array<std::vector<double>,4> ROM_quantities;
    ROM_quantities = compute_quantities(*flow_solver->dg,euler_physics_double);
    std::array<double,2> FOM_integrated_values {{0,0}};
    std::array<double,2> ROM_integrated_values {{0,0}};

    FOM_integrated_values = integrate_quantities(*FOM_flow_solver.dg,euler_physics_double);
    ROM_integrated_values = integrate_quantities(*flow_solver->dg,euler_physics_double);
    double inner_product_ROM;
    // Assembling ROM vh^T*RHS
    if(all_parameters->reduced_order_param.entropy_varibles_in_snapshots){
         std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> pod_basis = offline_pod->getPODBasis();
        flow_solver->dg->calculate_global_entropy();
        flow_solver->dg->calculate_ROM_projected_entropy(*pod_basis);
        flow_solver->dg->assemble_residual();
        dealii::LinearAlgebra::distributed::Vector<double> inner_product_vector_ROM;
        inner_product_vector_ROM.reinit(flow_solver->dg->projected_entropy);
        inner_product_ROM = inner_product_vector_ROM.add_and_dot(1.0,flow_solver->dg->projected_entropy,flow_solver->dg->right_hand_side);

    } else {
        std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> pod_basis = offline_pod->getPODBasis();
        flow_solver->dg->calculate_global_entropy();
        //flow_solver->dg->calculate_ROM_projected_entropy(*pod_basis);
        flow_solver->dg->assemble_residual();
        dealii::LinearAlgebra::distributed::Vector<double> inner_product_vector_ROM;
        inner_product_vector_ROM.reinit(flow_solver->dg->global_entropy);
        inner_product_ROM = inner_product_vector_ROM.add_and_dot(1.0,flow_solver->dg->global_entropy,flow_solver->dg->right_hand_side);
    }
    // Assembling FOM vh^T*RHS
    FOM_flow_solver.dg->calculate_global_entropy();
    FOM_flow_solver.dg->assemble_residual();
    dealii::LinearAlgebra::distributed::Vector<double> inner_product_vector_FOM;
    inner_product_vector_FOM.reinit(FOM_flow_solver.dg->global_entropy);
    double inner_product_FOM = inner_product_vector_FOM.add_and_dot(1.0,FOM_flow_solver.dg->global_entropy,FOM_flow_solver.dg->right_hand_side);



    std::array<double,4> L2_error {{0,0,0,0}};
    std::array<double,4> FOM_sum {{0,0,0,0}};

    for(unsigned int iquad = 0; iquad < FOM_quantities[0].size(); iquad++ ){
        for(int i = 0; i < 4; i++){
            L2_error[i] += std::pow((FOM_quantities[i][iquad]-ROM_quantities[i][iquad]),2);
            FOM_sum[i] += std::pow(FOM_quantities[i][iquad],2);
        }
    }
    for (int i = 0; i < 4; i++){
        L2_error[i] /= FOM_sum[i];
        L2_error[i] = std::sqrt(L2_error[i]);
    }
    if(iteration == 0) {
        this->rom_previous_entropy = ROM_integrated_values[1];
        this->fom_previous_entropy = FOM_integrated_values[1];
    }
    double ROM_entropy = ROM_integrated_values[1] - this->rom_previous_entropy;
    double FOM_entropy = FOM_integrated_values[1] - this->fom_previous_entropy;
    this->rom_previous_entropy = ROM_integrated_values[1];
    this->fom_previous_entropy = FOM_integrated_values[1];
    ROM_integrated_values[1] = ROM_entropy;
    FOM_integrated_values[1] = FOM_entropy;
    std::array<std::string,4> L2_labels {{"density_l2","pressure_l2","tke_l2","entropy_l2"}};
    std::array<std::string,2> FOM_labels{{"FOM_integrated_tke", "FOM_integrated_entropy"}};
    std::array<std::string,2> ROM_labels{{"ROM_integrated_tke", "ROM_integrated_entropy"}};
    if(this->mpi_rank==0) {
        L2error_data_table->add_value("time",current_time);
        L2error_data_table->set_precision("time", 16);
        L2error_data_table->set_scientific("time", true);
        for(int i = 0; i < 4; i++){
            L2error_data_table->add_value(L2_labels[i],L2_error[i]);
            L2error_data_table->set_precision(L2_labels[i], 16);
            L2error_data_table->set_scientific(L2_labels[i], true);
        }
        for(int i = 0; i < 2; i++){
            L2error_data_table->add_value(FOM_labels[i],FOM_integrated_values[i]);
            L2error_data_table->set_precision(FOM_labels[i], 16);
            L2error_data_table->set_scientific(FOM_labels[i], true);
        }
        for(int i =0; i < 2; i++){
            L2error_data_table->add_value(ROM_labels[i],ROM_integrated_values[i]);
            L2error_data_table->set_precision(ROM_labels[i], 16);
            L2error_data_table->set_scientific(ROM_labels[i], true);
        }
        L2error_data_table->add_value("vh^T*RHS_FOM",inner_product_FOM);
        L2error_data_table->set_precision("vh^T*RHS_FOM", 16);
        L2error_data_table->set_scientific("vh^T*RHS_FOM", true);
        L2error_data_table->add_value("vh^T*RHS_ROM",inner_product_ROM);
        L2error_data_table->set_precision("vh^T*RHS_ROM", 16);
        L2error_data_table->set_scientific("vh^T*RHS_ROM", true);
        
        std::ofstream l2_error_table_file("rom_l2_error.txt");
        L2error_data_table->write_text(l2_error_table_file);
    }
}

template class PODUnsteady<PHILIP_DIM,PHILIP_DIM+2>;

}
}