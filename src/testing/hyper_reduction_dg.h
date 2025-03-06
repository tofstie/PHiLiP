#ifndef HYPER_REDUCTION_DG_H
#define HYPER_REDUCTION_DG_H
#include "tests.h"
#include "parameters/all_parameters.h"
#include "Epetra_CrsMatrix.h"
#include "dg/dg_base.hpp"

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

    std::shared_ptr<Epetra_CrsMatrix> generate_reduced_lhs(std::shared_ptr< DGBase<dim, double> > dg, const Epetra_CrsMatrix &test_basis, const Epetra_CrsMatrix &trial_basis) const;
    std::shared_ptr<Epetra_CrsMatrix> generate_test_basis(std::shared_ptr< DGBase<dim, double> > dg, const Epetra_CrsMatrix &pod_basis) const;

    /// Dummy parameter handler because flowsolver requires it
    const dealii::ParameterHandler &parameter_handler;
};
} // End of Tests namespace
} // End of PHiLiP namespace
#endif //HYPER_REDUCTION_DG_H
