#ifndef __UNSTEADY_DOF_AND_QUAD_ESROM_H__
#define __UNSTEADY_DOF_AND_QUAD_ESROM_H__

#include "tests.h"
#include "parameters/all_parameters.h"

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
class UnsteadyDofAndQuadESROM: public TestsBase
{
public:
    UnsteadyDofAndQuadESROM(const Parameters::AllParameters *const parameters_input,
                 const dealii::ParameterHandler &parameter_handler_input);

    /// Run Unsteady POD reduced order
    int run_test () const override;

    /// Dummy parameter handler because flowsolver requires it
    const dealii::ParameterHandler &parameter_handler;
};
}
}

#endif //__UNSTEADY_DOF_AND_QUAD_ESROM_H__
