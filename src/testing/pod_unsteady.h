#ifndef __POD_UNSTEADY__
#define __POD_UNSTEADY__

#include <deal.II/numerics/vector_tools.h>
#include "parameters/all_parameters.h"
#include "reduced_order/pod_basis_online.h"
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

    /// Most up to date POD basis
    std::shared_ptr<ProperOrthogonalDecomposition::OnlinePOD<dim>> current_pod;
    

    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver;
    /// Number of time steps for every snapshots
    mutable int snapshots_every_x_steps;

    /// Run Test
    int run_test() const override;

    /// Output Snapshots
    void outputSnapshotData(int iteration) const;

    /// Compute Hyper-reduction points
    //std::tuple<int,int> computeHyperReduction(dealii::TrilinosWrappers::SparseMatrix V, dealii::LinearAlgebra::distributed::Vector w) const;
};
}
}
#endif