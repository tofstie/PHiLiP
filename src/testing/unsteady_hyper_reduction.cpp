//
// Created by tyson on 19/11/24.
//

#include "unsteady_hyper_reduction.h"
#include "flow_solver/flow_solver.h"
#include "flow_solver/flow_solver_factory.h"
#include "flow_solver/flow_solver_cases/periodic_entropy_tests.h"
#include "reduced_order/assemble_ECSW_residual.h"
#include "reduced_order/assemble_ECSW_jacobian.h"
#include "linear_solver/NNLS_solver.h"
#include "linear_solver/helper_functions.h"

namespace PHiLiP {
namespace Tests {


template<int dim, int nstate>
UnsteadyHyperReduction<dim, nstate>::UnsteadyHyperReduction(const Parameters::AllParameters *const parameters_input,
                                                            const dealii::ParameterHandler &parameter_handler_input)
                                                            : TestsBase::TestsBase(parameters_input)
                                                            , parameter_handler(parameter_handler_input)
{}

template<int dim, int nstate>
int UnsteadyHyperReduction<dim, nstate>::run_test() const {
    int test_fail = 0;
    std::cout << "Starting Unsteady Hyper Reduction Test" << std::endl;
    // Create FlowSolverCase
    std::unique_ptr<FlowSolver::PeriodicEntropyTests<dim, nstate>> flow_solver_case = std::make_unique<FlowSolver::PeriodicEntropyTests<dim,nstate>>(all_parameters);
    // Creating FOM and Solve
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver_full_order = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(all_parameters, parameter_handler);
    const double initial_FOM_entropy = flow_solver_case->compute_entropy(flow_solver_full_order->dg);
    //flow_solver_full_order->run();
    const double final_FOM_entropy = flow_solver_case->compute_entropy(flow_solver_full_order->dg);
    const double FOM_entropy_diff = final_FOM_entropy - initial_FOM_entropy;

    // Create HROM Parameters
    Parameters::AllParameters HROM_param = *(TestsBase::all_parameters);
    HROM_param.ode_solver_param.ode_solver_type = Parameters::ODESolverParam::ODESolverEnum::hyper_reduced_galerkin_runge_kutta_solver;
    HROM_param.ode_solver_param.allocate_matrix_dRdW = true;
    HROM_param.reduced_order_param.entropy_variables_in_snapshots = true;
    //HROM_param.reduced_order_param.entropy_variables_in_snapshots = true;
    HROM_param.flow_solver_param.unsteady_data_table_filename = "HROM_" + HROM_param.flow_solver_param.unsteady_data_table_filename;
    const Parameters::AllParameters HROM_param_const = HROM_param;

    // Create HROM Flow Solver
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> HROM_flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&HROM_param_const, parameter_handler);
    const double initial_HROM_entropy = flow_solver_case->compute_entropy(HROM_flow_solver->dg);
    HROM_flow_solver->run();
    const double final_HROM_entropy = flow_solver_case->compute_entropy(HROM_flow_solver->dg);
    const double HROM_entropy_diff = final_HROM_entropy - initial_HROM_entropy;

    std::cout << "FOM Entropy: " << FOM_entropy_diff << std::endl;
    std::cout << "HROM Entropy: " << HROM_entropy_diff << std::endl;

    return test_fail;
}


template class UnsteadyHyperReduction<PHILIP_DIM, PHILIP_DIM+2>;
}
}