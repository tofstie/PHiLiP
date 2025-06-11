#include "unsteady_dof_and_quad_ESROM.h"
#include "flow_solver/flow_solver.h"
#include "flow_solver/flow_solver_factory.h"
#include "flow_solver/flow_solver_cases/periodic_entropy_tests.h"
#include "ode_solver/ode_solver_factory.h"
#include "reduced_order/pod_basis_online.h"

namespace PHiLiP {
namespace Tests {

template<int dim, int nstate>
UnsteadyDofAndQuadESROM<dim,nstate>::UnsteadyDofAndQuadESROM(const Parameters::AllParameters *const parameters_input,
                                                     const dealii::ParameterHandler &parameter_handler_input)
      : TestsBase::TestsBase(parameters_input)
      , parameter_handler(parameter_handler_input)
{}

template<int dim, int nstate>
int UnsteadyDofAndQuadESROM<dim,nstate>::run_test() const {
    pcout << "Starting unsteady reduced-order test..." << std::endl;
    int testfail = 0;

    // Create FlowSolverCase
    std::unique_ptr<FlowSolver::PeriodicEntropyTests<dim, nstate>> flow_solver_case = std::make_unique<FlowSolver::PeriodicEntropyTests<dim,nstate>>(all_parameters);
    // Creating FOM and Solve
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver_full_order = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(all_parameters, parameter_handler);
    flow_solver_full_order->run();


    // DOF ROM SV (MATLAB Script)

    // QUAD ROM SV (MATLAB Script)

    // DOF ESROM
    Parameters::AllParameters dof_param = *(TestsBase::all_parameters);
    dof_param.ode_solver_param.ode_solver_type = Parameters::ODESolverParam::ODESolverEnum::pod_galerkin_runge_kutta_solver;
    dof_param.ode_solver_param.allocate_matrix_dRdW = true;
    dof_param.flow_solver_param.unsteady_data_table_filename = "dof_ESROM_"+dof_param.flow_solver_param.unsteady_data_table_filename;
    dof_param.reduced_order_param.entropy_variables_in_snapshots = true;
    dof_param.reduced_order_param.quadrature_POD = false;
    dof_param.reduced_order_param.number_modes *= nstate;
    const Parameters::AllParameters dof_param_const = dof_param;
    // Create ROM and Solve
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver_dof = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&dof_param_const, parameter_handler);
    try {
        static_cast<void>(flow_solver_dof->run());
    } catch (double end) {
        this->pcout << "ROM Failed at t = " << flow_solver_dof->ode_solver->current_time << std::endl;
    }

    // QUAD ESROM
    Parameters::AllParameters quad_param = *(TestsBase::all_parameters);
    quad_param.ode_solver_param.ode_solver_type = Parameters::ODESolverParam::ODESolverEnum::pod_galerkin_runge_kutta_solver;
    quad_param.ode_solver_param.allocate_matrix_dRdW = true;
    quad_param.flow_solver_param.unsteady_data_table_filename = "quad_ESROM_"+quad_param.flow_solver_param.unsteady_data_table_filename;
    quad_param.reduced_order_param.entropy_variables_in_snapshots = true;
    quad_param.reduced_order_param.quadrature_POD = true;
    const Parameters::AllParameters quad_param_const = quad_param;

    // Create ROM and Solve
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver_quad = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&quad_param_const, parameter_handler);

    try {
        static_cast<void>(flow_solver_quad->run());
    } catch (double end) {
        this->pcout << "ROM Failed at t = " << flow_solver_quad->ode_solver->current_time << std::endl;
    }

    dealii::LinearAlgebra::distributed::Vector<double> full_order_solution(flow_solver_full_order->dg->solution);
    dealii::LinearAlgebra::distributed::Vector<double> quad_solution(flow_solver_quad->dg->solution);
    dealii::LinearAlgebra::distributed::Vector<double> dof_solution(flow_solver_dof->dg->solution);


    const double quad_solution_error = ((quad_solution-=full_order_solution).l2_norm()/full_order_solution.l2_norm());
    const double dof_galerkin_solution_error = ((dof_solution-=full_order_solution).l2_norm()/full_order_solution.l2_norm());

    pcout << "Quad solution error: " << quad_solution_error << std::endl;
    pcout << "DOF Galerkin solution error: " << dof_galerkin_solution_error << std::endl;

    // Change Parameters to ROM
    return testfail;
}
template class UnsteadyDofAndQuadESROM<PHILIP_DIM, PHILIP_DIM+2>;
}
}