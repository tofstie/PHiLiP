#include <stdlib.h>
#include <iostream>

#include <deal.II/base/convergence_table.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/numerics/vector_tools.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/fe/mapping_q.h>


#include "vortex_shedding.h"

#include "physics/initial_conditions/initial_condition_function.h"
#include "physics/euler.h"
#include "physics/manufactured_solution.h"
#include "dg/dg_factory.hpp"
#include "ode_solver/ode_solver_factory.h"
#include "flow_solver/flow_solver_factory.h"
#include "flow_solver/flow_solver_cases/periodic_entropy_tests.h"

namespace PHiLiP{
namespace Tests {

dealii::Point<2> center_of_circle(0.0,0.0);
const double inner_radius = 1, outer_radius = inner_radius*20;

void cylinder (dealii::parallel::distributed::Triangulation<2> & tria,
                    const unsigned int n_cells_circle,
                    const unsigned int n_cells_radial)
{
    const bool colorize = true;


    dealii::GridGenerator::hyper_shell(tria,center_of_circle,inner_radius,outer_radius,n_cells_circle + n_cells_radial,colorize);
    tria.set_all_manifold_ids(0);
    tria.set_manifold(0, dealii::SphericalManifold<2>(center_of_circle));
    for (auto cell = tria.begin_active(); cell != tria.end(); ++cell) {
        //if (!cell->is_locally_owned()) continue;
        for (unsigned int face=0; face<dealii::GeometryInfo<2>::faces_per_cell; ++face) {
            if (cell->face(face)->at_boundary()) {
                unsigned int current_id = cell->face(face)->boundary_id();
                if (current_id == 0) {
                    cell->face(face)->set_boundary_id (1004); // x_left, Farfield
                } else if (current_id == 1) {
                    cell->face(face)->set_boundary_id (1001); // x_right, Symmetry/Wall
                } else if (current_id == 2) {
                    cell->face(face)->set_boundary_id (1001); // y_bottom, Symmetry/Wall
                } else if (current_id == 3) {
                    cell->face(face)->set_boundary_id (1001); // y_top, Wall
                } else {
                    std::abort();
                }
            }
        }
    }

}

template <int dim, int nstate>
VortexShedding<dim,nstate>::VortexShedding(
    const PHiLiP::Parameters::AllParameters *const parameters_input,
    const dealii::ParameterHandler &parameter_handler_input)
    :
    TestsBase::TestsBase(parameters_input), 
    parameter_handler(parameter_handler_input)
{}

template <int dim, int nstate>
int VortexShedding<dim,nstate>
::run_test () const {
    pcout << "Starting Test" << std::endl;
    const double final_time = this->all_parameters->flow_solver_param.final_time;
    const double initial_time_step = this->all_parameters->ode_solver_param.initial_time_step;
    const int n_steps = floor(final_time/initial_time_step);
    pcout << "Checking Warning" << std::endl;
    if (n_steps * initial_time_step != final_time){
        pcout << "WARNING: final_time is not evenly divisible by initial_time_step!" << std::endl
              << "Remainder is " << fmod(final_time, initial_time_step)
              << ". Consider modifying parameters." << std::endl;
    }

    // Initialize flow_solver
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(this->all_parameters, parameter_handler);
    try{
        static_cast<void>(flow_solver->run());
    }
    catch(double end) {
        pcout << "Simulation did not reach the end time. Crashed at t = " << end <<std::endl;
    }
    return 0;
}   
#if PHILIP_DIM==2
    template class VortexShedding<PHILIP_DIM,PHILIP_DIM+2>;
#endif
}
}