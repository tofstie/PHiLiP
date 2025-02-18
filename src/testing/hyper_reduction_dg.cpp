//
// Created by tyson on 17/02/25.
//

#include "hyper_reduction_dg.h"

#include "dg/hyper_reduced_dg.hpp"
#include "flow_solver/flow_solver_cases/periodic_entropy_tests.h"

namespace PHiLiP {
namespace Tests {

template<int dim, int nstate>
HyperReductionDG<dim,nstate>::HyperReductionDG(const Parameters::AllParameters *const parameters_input,
                                                       const dealii::ParameterHandler &parameter_handler_input)
        : TestsBase::TestsBase(parameters_input)
        , parameter_handler(parameter_handler_input)
{}

template <int dim, int nstate>
int HyperReductionDG<dim,nstate>::run_test() const {
    // Setup
#if PHILIP_DIM == 1
    using Triangulation = typename dealii::Triangulation<dim>;
#else
    using Triangulation = typename dealii::parallel::distributed::Triangulation<dim>;
#endif
    int passing = 0;
    std::unique_ptr<FlowSolver::PeriodicEntropyTests<dim, nstate>> flow_solver_case = std::make_unique<FlowSolver::PeriodicEntropyTests<dim,nstate>>(all_parameters);
    const int poly_degree = all_parameters->flow_solver_param.poly_degree;
    const int grid_degree = all_parameters->flow_solver_param.grid_degree;
    const int max_poly = all_parameters->flow_solver_param.max_poly_degree_for_adaptation;
    std::unique_ptr<DGHyper<dim,nstate,double,Triangulation>> dg = std::make_unique<DGHyper<dim,nstate,double,Triangulation>>(all_parameters,poly_degree,max_poly,grid_degree,flow_solver_case->generate_grid());
    std::vector<double> reduced_weights = {2.,1.,4.,3.,4.,0.5};
    std::vector<unsigned int> reduced_indices = {1,4,6,8,10,14};
    dealii::Vector<double> reduced_mesh_weights(pow(all_parameters->flow_solver_param.number_of_grid_elements_per_dimension,dim));
    // Make Reduced Weights
    reduced_mesh_weights.add(reduced_indices,reduced_weights);
    dg->reduced_mesh_weights = reduced_mesh_weights;
    // Construct Qx, 1 and test
    Epetra_CrsMatrix Q = dg->construct_global_Q();
    std::ofstream Q_outfile("Q.txt");
    Q.Print(Q_outfile);
    Epetra_Vector ones(Q.DomainMap());
    Epetra_Vector result(Q.RangeMap());
    const int global_size = ones.GlobalLength();
    double *one_array = new double[global_size];
    int *indicies = new int[global_size];
    for(int i = 0; i < global_size; ++i)
    {
        one_array[i] = 1.0;
        indicies[i] = i;
    }
    ones.ReplaceGlobalValues(global_size,one_array,indicies);
    delete [] one_array;
    delete [] indicies;
    Q.Multiply(false,ones,result);
    for(int i = 0; i < result.MyLength(); ++i) {
        if(result[i] != 0.0) {
            passing = 1;
        }
    }
    // Construct F
    Epetra_CrsMatrix Fx(Epetra_DataAccess::Copy,Q.RowMap(),global_size);
#if PHILIP_DIM==1
    Epetra_CrsMatrix Fy(Epetra_DataAccess::Copy,Q.RowMap(),0);
    Epetra_CrsMatrix Fz(Epetra_DataAccess::Copy,Q.RowMap(),0);
#elif PHILIP_DIM==2
    Epetra_CrsMatrix Fy(Epetra_DataAccess::Copy,Q.RowMap(),global_size);
    Epetra_CrsMatrix Fz(Epetra_DataAccess::Copy,Q.RowMap(),0);
#elif PHILIP_DIM==3
    Epetra_CrsMatrix Fy(Epetra_DataAccess::Copy,Q.RowMap(),global_size);
    Epetra_CrsMatrix Fz(Epetra_DataAccess::Copy,Q.RowMap(),global_size);
#endif
    dg->calculate_convective_flux_matrix(Fx,Fy,Fz);
    std::ofstream Fx_file("Fx.txt");
    std::ofstream Fy_file("Fy.txt");
    std::ofstream Fz_file("Fz.txt");
    Fx.Print(Fx_file);
    Fy.Print(Fy_file);
    Fz.Print(Fz_file);
    Fx_file.close();
    Fy_file.close();
    Fz_file.close();

    return passing;
}

template class HyperReductionDG<PHILIP_DIM, PHILIP_DIM+2>;

} // End of Tests namespace
} // End of PHiLiP namespace