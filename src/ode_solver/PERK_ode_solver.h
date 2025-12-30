#ifndef __PERK_ODESOLVER__
#define __PERK_ODESOLVER__

#include "JFNK_solver/JFNK_solver.h"
#include "dg/dg_base.hpp"
#include "runge_kutta_base.h"
#include "runge_kutta_methods/PERK_tableau_base.h"
#include "relaxation_runge_kutta/empty_RRK_base.h"

namespace PHiLiP {
namespace ODE {

/** Reference for the Paired Explicit Runge-Kutta (PERK) schemes is found in
* Vermeire, Brian. (2023). Paired explicit Runge-Kutta schemes for stiff systems of equations.
* Journal of Computational Physics, 393, 465-483. https://doi.org/10.1016/j.jcp.2023.112159
*
* In this class, we allocate the PERK scheme and the partitioning for the specific problem
*/
#if PHILIP_DIM==1
template <int dim, typename real, int n_rk_stages, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, int n_rk_stages, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class PERKODESolver: public RungeKuttaBase <dim, real, n_rk_stages, MeshType>
{
public:
    /// Default constructor that will set the constants.
    PERKODESolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input,
            std::shared_ptr<PERKTableauBase<dim,real,MeshType>> rk_tableau_input,
            std::shared_ptr<EmptyRRKBase<dim,real,MeshType>> RRK_object_input); ///< Constructor.

    /// Destructor
    virtual ~PERKODESolver() = default;

    void allocate_runge_kutta_system () override;

    void calculate_stage_solution (int i, real dt, const bool pseudotime) override;

    void calculate_stage_derivative (int i, real dt) override;

    void sum_stages (real dt, const bool pseudotime) override;

    void apply_limiter () override;

    real adjust_time_step (real dt) override;

    /// Partitions the grid for the PERK scheme
    void partition_scheme() override;

private:
    /// Partitions the grid based on cell size
    void cell_size_partition(
        const std::size_t n_groups,
        std::vector<dealii::LinearAlgebra::distributed::Vector<int>> &local_locations_to_evaluate
        );

    /// Partitions the grid based on cell number
    void cell_number_partition(
        const std::size_t n_groups,
        std::vector<dealii::LinearAlgebra::distributed::Vector<int>> &local_locations_to_evaluate
    );

protected:
    /// Stores Butcher tableau a and b, which specify the RK method
    std::shared_ptr<PERKTableauBase<dim,real,MeshType>> butcher_tableau;
    dealii::LinearAlgebra::distributed::Vector<double> stage_solution;

};



} // ODE namespace
} // PHiLiP namespace

#endif