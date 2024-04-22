#ifndef __VORTEX_SHEDDING_H__
#define __VORTEX_SHEDDING_H__

#include <deal.II/grid/manifold_lib.h>

#include "dg/dg_base.hpp"
#include "parameters/all_parameters.h"
#include "physics/physics.h"
#include "tests.h"

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
class VortexShedding: public TestsBase
{
public:
    explicit VortexShedding(const Parameters::AllParameters *const parameters_input,
    const dealii::ParameterHandler &parameter_handler_input);
    const dealii::ParameterHandler &parameter_handler;
    int run_test() const override;
};
}
} 
#endif