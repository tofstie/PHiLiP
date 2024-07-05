#ifndef __HYPER_REDUCED_ENTROPY_CONSERVING__
#define __HYPER_REDUCED_ENTROPY_CONSERVING__

#include <deal.II/numerics/vector_tools.h>
#include "parameters/all_parameters.h"
#include "pod_basis_online.h"
#include "rom_test_location.h"
#include <eigen/Eigen/Dense>
#include "nearest_neighbors.h"
#include "adaptive_sampling_base.h"

namespace PHiLiP {
template<int dim, int nstate>
class HyperreducedEntropyConserving{
    HyperreducedEntropyConserving(const PHiLiP::Parameters::AllParameters *const parameters_input,
                                                const dealii::ParameterHandler &parameter_handler_input);

    /// Ptr vector of Weights
    mutable std::shared_ptr<Epetra_Vector> ptr_weights;
}
}
#endif