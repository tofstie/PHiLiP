//
// Created by tyson on 19/11/24.
//

#ifndef __UNSTEADY_HYPER_REDUCTION_H__
#define __UNSTEADY_HYPER_REDUCTION_H__

#include "tests.h"
#include "parameters/all_parameters.h"

namespace PHiLiP {
namespace Tests {
template<int dim, int nstate>
class UnsteadyHyperReduction: public TestsBase {
public:
    UnsteadyHyperReduction(const Parameters::AllParameters *const parameters_input,
                 const dealii::ParameterHandler &parameter_handler_input);
    /// Build three models and evaluate error measures
    int run_test () const override;

    /// Dummy parameter handler because flowsolver requires it
    const dealii::ParameterHandler &parameter_handler;
};

}
}

#endif //__UNSTEADY_HYPER_REDUCTION_H__
