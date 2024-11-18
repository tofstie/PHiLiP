#include "unsteady_reduced_order.h"
#include "flow_solver/flow_solver.h"
#include "flow_solver/flow_solver_factory.h"
#include "flow_solver/flow_solver_cases/periodic_entropy_tests.h"
#include "ode_solver/ode_solver_factory.h"
#include "reduced_order/pod_basis_online.h"
namespace PHiLiP {
namespace Tests {

template<int dim, int nstate>
UnsteadyReducedOrder<dim,nstate>::UnsteadyReducedOrder(const Parameters::AllParameters *const parameters_input,
                                                       const dealii::ParameterHandler &parameter_handler_input)
        : TestsBase::TestsBase(parameters_input)
        , parameter_handler(parameter_handler_input)
{}

template<int dim, int nstate>
int UnsteadyReducedOrder<dim,nstate>::run_test() const 
{
    pcout << "Starting unsteady reduced-order test..." << std::endl;
    int testfail = 0;

    // Create FlowSolverCase
    std::unique_ptr<FlowSolver::PeriodicEntropyTests<dim, nstate>> flow_solver_case = std::make_unique<FlowSolver::PeriodicEntropyTests<dim,nstate>>(all_parameters);
    // Creating FOM and Solve
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver_full_order = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(all_parameters, parameter_handler);
    const double initial_FOM_entropy = flow_solver_case->compute_entropy(flow_solver_full_order->dg);
    std::cout << initial_FOM_entropy << std::endl;
    flow_solver_full_order->run();
    const double end_FOM_entropy = flow_solver_case->compute_entropy(flow_solver_full_order->dg);
    // Change Parameters to ROM
    Parameters::AllParameters ROM_param = *(TestsBase::all_parameters);
    ROM_param.ode_solver_param.ode_solver_type = Parameters::ODESolverParam::ODESolverEnum::pod_galerkin_runge_kutta_solver;
    ROM_param.ode_solver_param.allocate_matrix_dRdW = true;
    const Parameters::AllParameters ROM_param_const = ROM_param;

    // Create ROM and Solve
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver_galerkin = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&ROM_param_const, parameter_handler);
    const int modes = flow_solver_galerkin->ode_solver->pod->getPODBasis()->n();
    //flow_solver_galerkin->run();

    // Change Parameters to Entropy-Stable ROM
    ROM_param.reduced_order_param.entropy_varibles_in_snapshots = true;
    const Parameters::AllParameters Entropy_ROM_param_const = ROM_param;
    // Create ROM and Solve
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver_entropy_galerkin = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&Entropy_ROM_param_const, parameter_handler);
    dealii::LinearAlgebra::distributed::Vector<double> entropy_intial_solution(flow_solver_entropy_galerkin->dg->solution);
    const double initial_entropy = flow_solver_case->compute_entropy(flow_solver_entropy_galerkin->dg);
    flow_solver_entropy_galerkin->run();

    dealii::LinearAlgebra::distributed::Vector<double> full_order_solution(flow_solver_full_order->dg->solution);
    dealii::LinearAlgebra::distributed::Vector<double> galerkin_solution(flow_solver_galerkin->dg->solution);
    dealii::LinearAlgebra::distributed::Vector<double> entropy_galerkin_solution(flow_solver_entropy_galerkin->dg->solution);

    const double FOM_change_in_entropy = end_FOM_entropy-initial_FOM_entropy;

    const double end_entropy_cons_entropy = flow_solver_case->compute_entropy(flow_solver_entropy_galerkin->dg);
    const double change_in_entropy = end_entropy_cons_entropy-initial_entropy;

    const double galerkin_solution_error = ((galerkin_solution-=full_order_solution).l2_norm()/full_order_solution.l2_norm());
    const double entropy_galerkin_solution_error = ((entropy_galerkin_solution-=full_order_solution).l2_norm()/full_order_solution.l2_norm());
    
    pcout << "Galerkin solution error: " << galerkin_solution_error << std::endl;
    pcout << "Entropy Galerkin solution error: " << entropy_galerkin_solution_error << std::endl;
    pcout << "FOM Change in Entropy: " << FOM_change_in_entropy << std::endl;
    pcout << "Entropy Galerkin Change in Entropy: " << change_in_entropy << std::endl;

    if (std::abs(galerkin_solution_error) > 2.5E-5) testfail = 1;
    // Hard coding expected_modes based on past test results
    if (constexpr int expected_modes = 30; modes != expected_modes) testfail = 1;
    return testfail;
}

template class UnsteadyReducedOrder<PHILIP_DIM, PHILIP_DIM+2>;


} // Tests namespace
} // PHiLiP namespace