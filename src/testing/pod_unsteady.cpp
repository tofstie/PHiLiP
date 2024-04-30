#include "pod_unsteady.h"
#include <iostream>
#include <filesystem>
#include "functional/functional.h"
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/vector_operation.h>
#include "reduced_order/reduced_order_solution.h"
#include "flow_solver/flow_solver.h"
#include "flow_solver/flow_solver_factory.h"
#include <cmath>
#include "reduced_order/rbf_interpolation.h"
#include "ROL_Algorithm.hpp"
#include "ROL_LineSearchStep.hpp"
#include "ROL_StatusTest.hpp"
#include "ROL_Stream.hpp"
#include "ROL_Bounds.hpp"
#include "reduced_order/halton.h"
#include "reduced_order/min_max_scaler.h"
#include <deal.II/base/timer.h>
#include "tests.h"

namespace PHiLiP {
namespace Tests {

template<int dim, int nstate>
PODUnsteady<dim,nstate>::PODUnsteady(
    const PHiLiP::Parameters::AllParameters *const parameters_input,
    const dealii::ParameterHandler &parameter_handler_input)
    : TestsBase::TestsBase(parameters_input)
    , parameter_handler(parameter_handler_input)
    {
        pcout << "Initializing POD Test" << std::endl;
        flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(all_parameters, parameter_handler);
        pcout << "Flow solver made" << std::endl;
        //const bool compute_dRdW = true;
        //flow_solver->dg->assemble_residual(compute_dRdW);
        pcout << "Assembled Residual" << std::endl;
        std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> system_matrix = std::make_shared<dealii::TrilinosWrappers::SparseMatrix>();
        system_matrix->copy_from(flow_solver->dg->system_matrix);
        pcout << "System matrix made" << std::endl;
        current_pod = std::make_shared<ProperOrthogonalDecomposition::OnlinePOD<dim>>(system_matrix);
        pcout << "Online POD made" << std::endl;
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

    auto all_param = *this->all_parameters;
    auto flow_solver_param = all_param.flow_solver_param;
    auto ode_param = all_param.ode_solver_param;
    auto do_output_solution_at_fixed_times = ode_param.output_solution_at_fixed_times;
    auto number_of_fixed_times_to_output_solution = ode_param.number_of_fixed_times_to_output_solution;
    auto output_solution_at_exact_fixed_times = ode_param.output_solution_at_exact_fixed_times;
    auto final_time = flow_solver_param.final_time;

    int output_snapshot_every_x_timesteps = 100;
    int number_of_timesteps = 0;
    int iteration = 0;
    dealii::Table<1,double> output_solution_fixed_times;
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
        //----------------------------------------------------
        // dealii::TableHandler and data at initial time
        //----------------------------------------------------
        std::shared_ptr<dealii::TableHandler> unsteady_data_table = std::make_shared<dealii::TableHandler>();
        
        // no restart:
        pcout << "Writing unsteady data computed at initial time... " << std::endl;
        flow_solver->flow_solver_case->compute_unsteady_data_and_write_to_table(flow_solver->ode_solver, flow_solver->dg, unsteady_data_table);
        pcout << "done." << std::endl;
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
        flow_solver->flow_solver_case->set_time_step(time_step);

        double next_time_step = time_step;
        pcout << "Advancing solution in time... " << std::endl;
        pcout << "Timer starting. Test file " << std::endl;
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
            flow_solver->ode_solver->step_in_time(time_step,false); // pseudotime==false

            // Compute the unsteady quantities, write to the dealii table, and output to file
            flow_solver->flow_solver_case->compute_unsteady_data_and_write_to_table(flow_solver->ode_solver, flow_solver->dg, unsteady_data_table);
            // update next time step
            if(flow_solver_param.adaptive_time_step == true) {
                next_time_step = flow_solver->flow_solver_case->get_adaptive_time_step(flow_solver->dg);
            } else {
                next_time_step = flow_solver->flow_solver_case->get_constant_time_step(flow_solver->dg);
            }
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
            if(number_of_timesteps == output_snapshot_every_x_timesteps){
                number_of_timesteps = 0;
                current_pod->addSnapshot(flow_solver->dg->solution);
                outputSnapshotData(iteration);
                iteration++;
                pcout << "Outputed Snapshot Data" << std::endl;
            }
        } // closing while loop
        timer.stop();
        
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
    current_pod->dealiiSnapshotMatrix.print_formatted(solution_out_file, precision);
    solution_out_file.close();
};
#if PHILIP_DIM==1
    template class PODUnsteady<PHILIP_DIM,PHILIP_DIM+2>;
#endif
}
}