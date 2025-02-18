#ifndef HYPER_REDUCTION_DG_H
#define HYPER_REDUCTION_DG_H
#include "tests.h"
#include "parameters/all_parameters.h"

namespace PHiLiP {
namespace Tests {

/// Unsteady POD reduced order test, verifies consistency of solution and implementation of threshold function
template <int dim, int nstate>
class HyperReductionDG: public TestsBase
{
public:
    /// Constructor.
    HyperReductionDG(const Parameters::AllParameters *const parameters_input,
                 const dealii::ParameterHandler &parameter_handler_input);

    /// Run Unsteady POD reduced order
    int run_test () const override;

    /// Dummy parameter handler because flowsolver requires it
    const dealii::ParameterHandler &parameter_handler;
};
} // End of Tests namespace
} // End of PHiLiP namespace
#endif //HYPER_REDUCTION_DG_H
