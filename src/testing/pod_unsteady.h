#ifndef __POD_UNSTEADY__
#define __POD_UNSTEADY__

#include <deal.II/numerics/vector_tools.h>
#include "parameters/all_parameters.h"
#include "reduced_order/pod_basis_base.h"
#include "reduced_order/pod_basis_online.h"
#include "reduced_order/pod_basis_offline.h"
#include "reduced_order/rom_test_location.h"
#include <eigen/Eigen/Dense>
#include "reduced_order/nearest_neighbors.h"
#include "tests.h"
#include "flow_solver/flow_solver.h"
#include "flow_solver/flow_solver_factory.h"

namespace PHiLiP {
namespace Tests {

using DealiiVector = dealii::LinearAlgebra::distributed::Vector<double>;
using Eigen::MatrixXd;
using Eigen::RowVectorXd;
using Eigen::VectorXd;
template <int dim, int nstate>
class PODUnsteady: public TestsBase
{
public:
    /// Constructor
    PODUnsteady(const PHiLiP::Parameters::AllParameters *const parameters_input,
                     const dealii::ParameterHandler &parameter_handler_input);
    /// Parameter handler for storing the .prm file being ran
    const dealii::ParameterHandler &parameter_handler;

    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver;

    const Parameters::AllParameters all_param; ///< All parameters
    const Parameters::FlowSolverParam flow_solver_param; ///< Flow solver parameters
    const Parameters::ODESolverParam ode_param; ///< ODE solver parameters

    const bool do_output_solution_at_fixed_times; ///< Flag for outputting solution at fixed times
    const unsigned int number_of_fixed_times_to_output_solution; ///< Number of fixed times to output the solution
    const bool output_solution_at_exact_fixed_times;///< Flag for outputting the solution at exact fixed times by decreasing the time step on the fly
    const double final_time; ///< Final time of solution
    const int output_snapshot_every_x_timesteps;
    /// Most up to date POD basis
    std::shared_ptr<ProperOrthogonalDecomposition::OnlinePOD<dim>> current_pod;
    std::shared_ptr<ProperOrthogonalDecomposition::PODBase<dim>> offline_pod;
    
    /// Number of time steps for every snapshots
    mutable int snapshots_every_x_steps;

    /// Run Test
    int run_test() const override;

    /// Output Snapshots
    void outputSnapshotData(int iteration) const;
    /// Output L2 Error
    void CalculateL2Error (std::shared_ptr <dealii::TableHandler> L2error_data_table,
                   FlowSolver::FlowSolver<dim,nstate> FOM_flow_solver,
                   Physics::Euler<dim,dim+2,double> euler_physics_double,
                   int iteration) const;
    std::array<std::vector<double>,4> compute_quantities(DGBase<dim, double> &dg, Physics::Euler<dim,dim+2,double> euler_physics) const;
    std::array<double,2> integrate_quantities(DGBase<dim, double> &dg, Physics::Euler<dim,dim+2,double> euler_physics) const;
    //void hyperReduction(double tol = 10) const;
    /// Compute Hyper-reduction points
    //std::tuple<int,int> computeHyperReduction(dealii::TrilinosWrappers::SparseMatrix V, dealii::LinearAlgebra::distributed::Vector w) const;
};
}
}
#endif