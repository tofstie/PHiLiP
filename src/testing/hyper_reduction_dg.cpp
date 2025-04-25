//
// Created by tyson on 17/02/25.
//

#include "hyper_reduction_dg.h"

#include "dg/hyper_reduced_dg.hpp"
#include "flow_solver/flow_solver_factory.h"
#include "flow_solver/flow_solver_cases/periodic_entropy_tests.h"
#include "physics/initial_conditions/set_initial_condition.h"
#include "reduced_order/pod_basis_offline.h"
#include "EpetraExt_MatrixMatrix.h"
#include "Epetra_SerialComm.h"
#include "operators/operators.h"

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
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver_full_order = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(all_parameters, parameter_handler);
    const int poly_degree = all_parameters->flow_solver_param.poly_degree;
    const int grid_degree = all_parameters->flow_solver_param.grid_degree;
    const int max_poly = all_parameters->flow_solver_param.max_poly_degree_for_adaptation;
    std::unique_ptr<DGHyper<dim,nstate,double,Triangulation>> dg = std::make_unique<DGHyper<dim,nstate,double,Triangulation>>(all_parameters,poly_degree,max_poly,grid_degree,flow_solver_case->generate_grid());
    std::shared_ptr<DGBase<dim,double>> dg_base = flow_solver_full_order->dg;
    std::vector<double> reduced_weights = {2.,1.,4.,3.,4.,0.5};
    std::vector<unsigned int> reduced_indices = {1,4,6,8,10,14};
    dealii::Vector<double> reduced_mesh_weights(pow(all_parameters->flow_solver_param.number_of_grid_elements_per_dimension,dim));
    dg->allocate_system(true,false,false);
    dg->evaluate_mass_matrices(false);
    dg->solution = dg_base->solution;
    dg->solution.update_ghost_values();
    reduced_mesh_weights.add(reduced_indices,reduced_weights);
    dg->reduced_mesh_weights = reduced_mesh_weights;
    // Make Reduced Weights
    flow_solver_full_order->run();
    dg_base->reduced_mesh_weights = reduced_mesh_weights;
    std::shared_ptr<ProperOrthogonalDecomposition::OfflinePOD<dim>> pod = std::make_shared<ProperOrthogonalDecomposition::OfflinePOD<dim>>(dg_base);
    std::shared_ptr<Epetra_CrsMatrix> V = std::make_shared<Epetra_CrsMatrix>(pod->getPODBasis()->trilinos_matrix());
    std::shared_ptr<Epetra_CrsMatrix> LeV = generate_test_basis(dg_base,*V);
    dg->set_galerkin_basis(LeV,false);
    std::shared_ptr<Epetra_CrsMatrix> LHS = generate_reduced_lhs(dg_base,*LeV,*LeV);
    dg->calculate_projection_matrix(*LHS,*LeV);

    const int global_size = dg->solution.size();
    Epetra_MpiComm comm(MPI_COMM_WORLD);
    Epetra_Map global_map(global_size,0,comm);
    Epetra_Map domain_map = dg->global_mass_matrix.trilinos_matrix().DomainMap();
    // Construct Qx, 1 and test
    Epetra_CrsMatrix Qx(Epetra_DataAccess::Copy,global_map,dg->global_mass_matrix.trilinos_matrix().ColMap().MaxElementSize());
    Epetra_CrsMatrix Qy(Epetra_DataAccess::Copy,global_map,dg->global_mass_matrix.trilinos_matrix().ColMap().MaxElementSize());
    Epetra_CrsMatrix Qz(Epetra_DataAccess::Copy,global_map,dg->global_mass_matrix.trilinos_matrix().ColMap().MaxElementSize());
    dg->construct_global_Q(Qx,Qy,Qz,false);
    Qx.FillComplete(domain_map,global_map);
    Qy.FillComplete(domain_map,global_map);
    Qz.FillComplete(domain_map,global_map);
    Epetra_CrsMatrix Qt = dg->calculate_hyper_reduced_Q(Qx,Qx,0);
    std::ofstream Q_outfile("Q.txt");
    Qx.Print(Q_outfile);
    std::ofstream QtOutfile("Qt.txt");
    Qt.Print(QtOutfile);
    Epetra_Vector ones(Qx.DomainMap());
    Epetra_Vector result(Qx.RangeMap());
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
    Qx.Multiply(false,ones,result);
    for(int i = 0; i < result.MyLength(); ++i) {
        if(result[i] != 0.0) {
            passing = 1;
        }
    }
    std::ofstream result_outfile("result.txt");
    result.Print(result_outfile);
    // Construct F
    Epetra_CrsMatrix Fx(Epetra_DataAccess::Copy,Qx.RowMap(),global_size);
#if PHILIP_DIM==1
    Epetra_CrsMatrix Fy(Epetra_DataAccess::Copy,Qy.RowMap(),0);
    Epetra_CrsMatrix Fz(Epetra_DataAccess::Copy,Qz.RowMap(),0);
#elif PHILIP_DIM==2
    Epetra_CrsMatrix Fy(Epetra_DataAccess::Copy,Qy.RowMap(),global_size);
    Epetra_CrsMatrix Fz(Epetra_DataAccess::Copy,Qz.RowMap(),0);
#elif PHILIP_DIM==3
    Epetra_CrsMatrix Fy(Epetra_DataAccess::Copy,Qy.RowMap(),global_size);
    Epetra_CrsMatrix Fz(Epetra_DataAccess::Copy,Qz.RowMap(),global_size);
#endif
    dg->calculate_global_entropy();
    dg->projected_entropy = dg->global_entropy;
    dg->calculate_convective_flux_matrix(Fx,Fy,Fz,Fx);
    Fx.FillComplete(Qx.DomainMap(),Qx.RowMap());
    Fy.FillComplete(Qy.DomainMap(),Qy.RowMap());
    Fz.FillComplete(Qz.DomainMap(),Qz.RowMap());
    OPERATOR::SumFactorizedOperators<dim,2*dim,double> op(nstate,max_poly,grid_degree);
    Epetra_CrsMatrix QtF(Epetra_DataAccess::View,Qt.RowMap(),Qt.ColMap(),Qt.NumGlobalCols());
    op.Hadamard_product(Qt,Fx,QtF);
    std::ofstream QtF_file("QtF.txt");
    std::ofstream Fx_file("Fx.txt");
    std::ofstream Fy_file("Fy.txt");
    std::ofstream Fz_file("Fz.txt");
    QtF.Print(QtF_file);
    Fx.Print(Fx_file);
    Fy.Print(Fy_file);
    Fz.Print(Fz_file);
    QtF_file.close();
    Fx_file.close();
    Fy_file.close();
    Fz_file.close();

    return passing;
}

template <int dim, int nstate>
std::shared_ptr<Epetra_CrsMatrix> HyperReductionDG<dim,nstate>::generate_reduced_lhs(
    std::shared_ptr< DGBase<dim, double> > dg,
    const Epetra_CrsMatrix &test_basis,
    const Epetra_CrsMatrix &trial_basis) const
{
    Epetra_CrsMatrix lhs_matrix(Epetra_DataAccess::Copy,test_basis.DomainMap(),test_basis.NumGlobalCols());
    // Setup for Local Mass Matrix
    using FR_enum = Parameters::AllParameters::Flux_Reconstruction;
    const FR_enum FR_Type = dg->all_parameters->flux_reconstruction_type;
    using FR_Aux_enum = Parameters::AllParameters::Flux_Reconstruction_Aux;
    const FR_Aux_enum FR_Type_Aux = dg->all_parameters->flux_reconstruction_aux_type;
    std::vector<dealii::types::global_dof_index> dofs_indices;
    const unsigned int init_grid_degree = dg->high_order_grid->fe_system.tensor_degree();
    OPERATOR::mapping_shape_functions<dim,2*dim,double> mapping_basis(1, dg->max_degree, init_grid_degree);//first set at max degree
    OPERATOR::basis_functions<dim,2*dim,double> basis(1, dg->max_degree, init_grid_degree);
    OPERATOR::local_mass<dim,2*dim,double> reference_mass_matrix(1, dg->max_degree, init_grid_degree);//first set at max degree
    OPERATOR::local_Flux_Reconstruction_operator<dim,2*dim,double> reference_FR(1, dg->max_degree, init_grid_degree, FR_Type);
    OPERATOR::local_Flux_Reconstruction_operator_aux<dim,2*dim,double> reference_FR_aux(1, dg->max_degree, init_grid_degree, FR_Type_Aux);
    OPERATOR::derivative_p<dim,2*dim,double> deriv_p(1, dg->max_degree, init_grid_degree);

    const int N_FOM_dim = test_basis.NumGlobalRows(); // Length of solution vector
    auto first_cell = dg->dof_handler.begin_active();
    const bool Cartesian_first_element = (first_cell->manifold_id() == dealii::numbers::flat_manifold_id);

    dg->reinit_operators_for_mass_matrix(Cartesian_first_element, dg->max_degree, init_grid_degree, mapping_basis, basis, reference_mass_matrix, reference_FR, reference_FR_aux, deriv_p);

    //Loop over cells and set the matrices.
    auto metric_cell = dg->high_order_grid->dof_handler_grid.begin_active();
    for (auto cell = dg->dof_handler.begin_active(); cell!=dg->dof_handler.end(); ++cell, ++metric_cell) {

        if (!cell->is_locally_owned()) continue;
        if (dg->reduced_mesh_weights[cell->active_cell_index()] == 0) continue;

        dg->global_mass_matrix.reinit(dg->locally_owned_dofs, dg->mass_sparsity_pattern);
        const bool Cartesian_element = (cell->manifold_id() == dealii::numbers::flat_manifold_id);

        const unsigned int fe_index_curr_cell = cell->active_fe_index();
        const unsigned int curr_grid_degree   = dg->high_order_grid->fe_system.tensor_degree();//in the future the metric cell's should store a local grid degree. currently high_order_grid dof_handler_grid doesn't have that capability

        //Check if need to recompute the 1D basis for the current degree (if different than previous cell)
        //That is, if the poly_degree, manifold type, or grid degree is different than previous reference operator
        if((fe_index_curr_cell != mapping_basis.current_degree) ||
           (curr_grid_degree != mapping_basis.current_grid_degree))
        {
            dg->reinit_operators_for_mass_matrix(Cartesian_element, fe_index_curr_cell, curr_grid_degree, mapping_basis, basis, reference_mass_matrix, reference_FR, reference_FR_aux, deriv_p);

            mapping_basis.current_degree = fe_index_curr_cell;
            basis.current_degree = fe_index_curr_cell;
            reference_mass_matrix.current_degree = fe_index_curr_cell;
            reference_FR.current_degree = fe_index_curr_cell;
            reference_FR_aux.current_degree = fe_index_curr_cell;
            deriv_p.current_degree = fe_index_curr_cell;
        }

        // Current reference element related to this physical cell
        const unsigned int n_dofs_cell = dg->fe_collection[fe_index_curr_cell].n_dofs_per_cell();
        const int n_dofs_cell_int = dg->fe_collection[fe_index_curr_cell].n_dofs_per_cell();
        const unsigned int n_quad_pts  = dg->volume_quadrature_collection[fe_index_curr_cell].size();

        //setup metric cell
        const dealii::FESystem<dim> &fe_metric = dg->high_order_grid->fe_system;
        const unsigned int n_metric_dofs = dg->high_order_grid->fe_system.dofs_per_cell;
        const unsigned int n_grid_nodes  = n_metric_dofs/dim;
        std::vector<dealii::types::global_dof_index> metric_dof_indices(n_metric_dofs);
        metric_cell->get_dof_indices (metric_dof_indices);
        // get mapping_support points
        std::array<std::vector<double>,dim> mapping_support_points;
        for(int idim=0; idim<dim; idim++){
            mapping_support_points[idim].resize(n_metric_dofs/dim);
        }
        const std::vector<unsigned int > &index_renumbering = dealii::FETools::hierarchic_to_lexicographic_numbering<dim>(curr_grid_degree);
        for (unsigned int idof = 0; idof< n_metric_dofs; ++idof) {
            const double val = (dg->high_order_grid->volume_nodes[metric_dof_indices[idof]]);
            const unsigned int istate = fe_metric.system_to_component_index(idof).first;
            const unsigned int ishape = fe_metric.system_to_component_index(idof).second;
            const unsigned int igrid_node = index_renumbering[ishape];
            mapping_support_points[istate][igrid_node] = val;
        }

        //get determinant of Jacobian
        OPERATOR::metric_operators<double, dim, 2*dim> metric_oper(nstate, fe_index_curr_cell, curr_grid_degree);
        metric_oper.build_determinant_volume_metric_Jacobian(
                        n_quad_pts, n_grid_nodes,
                        mapping_support_points,
                        mapping_basis);

        //Get dofs indices to set local matrices in global.
        dofs_indices.resize(n_dofs_cell);
        cell->get_dof_indices (dofs_indices);
        //Compute local matrices and set them in the global system.
        dg->evaluate_local_metric_dependent_mass_matrix_and_set_in_global_mass_matrix(
            Cartesian_element,
            false,
            fe_index_curr_cell,
            curr_grid_degree,
            n_quad_pts,
            n_dofs_cell,
            dofs_indices,
            metric_oper,
            basis,
            reference_mass_matrix,
            reference_FR,
            reference_FR_aux,
            deriv_p);
        // Set up the Le matrix
        const Epetra_SerialComm sComm;
        Epetra_Map LeRowMap(n_dofs_cell_int, 0, sComm);
        Epetra_Map LeTRowMap(N_FOM_dim, 0, sComm);
        Epetra_CrsMatrix L_e(Epetra_DataAccess::Copy, LeRowMap, LeTRowMap, 1);
        Epetra_CrsMatrix L_e_T(Epetra_DataAccess::Copy, LeTRowMap, n_dofs_cell_int);
        double posOne = 1.0;

        for(int i = 0; i < n_dofs_cell_int; i++){
            const int col = dofs_indices[i];
            L_e.InsertGlobalValues(i, 1, &posOne , &col);
            L_e_T.InsertGlobalValues(col, 1, &posOne , &i);
        }
        L_e.FillComplete(LeTRowMap, LeRowMap);
        L_e_T.FillComplete(LeRowMap, LeTRowMap);
        Epetra_CrsMatrix epetra_mass_matrix = dg->global_mass_matrix.trilinos_matrix();
        // Preform the Pre-mult of LeM
        Epetra_CrsMatrix J_L_e_T(Epetra_DataAccess::Copy, epetra_mass_matrix.RowMap(), n_dofs_cell_int);
        Epetra_CrsMatrix J_e_m(Epetra_DataAccess::Copy, LeRowMap, n_dofs_cell_int);
        EpetraExt::MatrixMatrix::Multiply(epetra_mass_matrix, false, L_e_T, false, J_L_e_T, true);
        EpetraExt::MatrixMatrix::Multiply(L_e, false, J_L_e_T, false, J_e_m, true);
        // Preform post-mult of LeMLe
        Epetra_CrsMatrix M_temp(Epetra_DataAccess::Copy, LeRowMap, N_FOM_dim);
        Epetra_CrsMatrix M_global_e(Epetra_DataAccess::Copy, LeTRowMap, N_FOM_dim);
        EpetraExt::MatrixMatrix::Multiply(J_e_m, false, L_e, false, M_temp, true);
        EpetraExt::MatrixMatrix::Multiply(L_e_T, false, M_temp, false, M_global_e, true);
        std::ofstream mass_element_le_file("mass_element.txt");
        M_global_e.Print(mass_element_le_file);
        if (test_basis.RowMap().SameAs(M_global_e.RowMap()) && test_basis.NumGlobalRows() == M_global_e.NumGlobalRows()){
            Epetra_CrsMatrix epetra_reduced_lhs(Epetra_DataAccess::Copy, test_basis.DomainMap(), test_basis.NumGlobalCols());
            Epetra_CrsMatrix epetra_reduced_lhs_tmp(Epetra_DataAccess::Copy, M_global_e.RowMap(), test_basis.NumGlobalCols());
            if (EpetraExt::MatrixMatrix::Multiply(M_global_e, false, test_basis, false, epetra_reduced_lhs_tmp) != 0){
                std::cerr << "Error in first Matrix Multiplication" << std::endl;
                return nullptr;
            };
            if (EpetraExt::MatrixMatrix::Multiply(trial_basis, true, epetra_reduced_lhs_tmp, false, epetra_reduced_lhs) != 0){
                std::cerr << "Error in second Matrix Multiplication" << std::endl;
                return nullptr;
            };
            if (EpetraExt::MatrixMatrix::Add(epetra_reduced_lhs,false,1.0,lhs_matrix,1.0) != 0) {
                std::cerr << "Error in third Matrix Add" << std::endl;
                return nullptr;
            }
        } else {
            if(!(test_basis.RowMap().SameAs(epetra_mass_matrix.RowMap()))){
                std::cerr << "Error: Inconsistent maps" << std::endl;
            } else {
                std::cerr << "Error: Inconsistent row sizes" << std::endl
                << "System: " << std::to_string(epetra_mass_matrix.NumGlobalRows()) << std::endl
                << "Test: " << std::to_string(test_basis.NumGlobalRows()) << std::endl;
            }
        }
    }
    //end of cell loop
    lhs_matrix.FillComplete(test_basis.DomainMap(),test_basis.DomainMap());
    std::ofstream lhs_file("lhs_matrix.txt");
    lhs_matrix.Print(lhs_file);
    dg->evaluate_mass_matrices(false);
    return std::make_shared<Epetra_CrsMatrix>(lhs_matrix);
}

template <int dim, int nstate>
std::shared_ptr<Epetra_CrsMatrix> HyperReductionDG<dim, nstate>::generate_test_basis(std::shared_ptr< DGBase<dim, double> > dg, const Epetra_CrsMatrix &pod_basis)
const {
    Epetra_MpiComm epetra_comm(MPI_COMM_WORLD);
    /*
    Epetra_Map column_map = pod_basis.DomainMap();
    int num_of_modes = column_map.NumGlobalElements();
    const unsigned int max_dofs_per_cell = this->dg->dof_handler.get_fe_collection().max_dofs_per_cell();
    std::vector<dealii::types::global_dof_index> current_dofs_indices(max_dofs_per_cell);
    /// GET ids
    std::set<unsigned int> rowIDs;
    for (const auto &cell : this->dg->dof_handler.active_cell_iterators()) {
        if (ECSW_weights[cell->active_cell_index()] != 0 ) {
            for (unsigned int quad_id : current_dofs_indices) {
                rowIDs.insert(quad_id);
            }
        }
    }
    const int hyper_reduced_size = rowIDs.size();
    Epetra_Map hyper_reduced_row_map(hyper_reduced_size,0,epetra_comm);
    /// Filter Test Basis using Row ids
    Epetra_CrsMatrix hyper_reduced_basis(Epetra_DataAccess::Copy, hyper_reduced_row_map,num_of_modes);
    int hyper_rowID = 0;
    for( int FOM_rowID : rowIDs ) {
        int num_entries = 0;
        double *global_row = new double [num_of_modes];
        int *indicies = new int [num_of_modes];
        pod_basis.ExtractGlobalRowCopy(FOM_rowID,num_of_modes,num_entries,global_row,indicies);
        hyper_reduced_basis.InsertGlobalValues(hyper_rowID,num_of_modes,global_row,indicies);
        hyper_rowID++;
    }
    hyper_reduced_basis.FillComplete(column_map,hyper_reduced_row_map);
    */
    Epetra_Map basis_rowmap = pod_basis.RowMap();
    Epetra_Map basis_domainmap = pod_basis.DomainMap();
    Epetra_CrsMatrix hyper_reduced_basis(Epetra_DataAccess::Copy, basis_rowmap,pod_basis.NumGlobalCols());
    const int N = pod_basis.NumGlobalRows();
    const int number_modes = pod_basis.NumGlobalCols();
    const unsigned int max_dofs_per_cell = dg->dof_handler.get_fe_collection().max_dofs_per_cell();
    std::vector<dealii::types::global_dof_index> current_dofs_indices(max_dofs_per_cell);
    std::vector<dealii::types::global_dof_index> neighbour_dofs_indices(max_dofs_per_cell);
    for (const auto &cell : dg->dof_handler.active_cell_iterators()) {
        if (dg->reduced_mesh_weights[cell->active_cell_index()] != 0 ) {
            const int fe_index_curr_cell = cell->active_fe_index();
            const dealii::FESystem<dim,dim> &current_fe_ref = dg->fe_collection[fe_index_curr_cell];
            const int n_dofs_curr_cell = current_fe_ref.n_dofs_per_cell();

            current_dofs_indices.resize(n_dofs_curr_cell);
            cell->get_dof_indices(current_dofs_indices);
    /*
            double *row = new double[pod_basis.NumGlobalCols()];
            int *global_indices = new int[pod_basis.NumGlobalCols()];
            int numE;
            int row_num = current_dofs_indices[0];
            pod_basis.ExtractGlobalRowCopy(row_num, pod_basis.NumGlobalCols(), numE, row, global_indices);
            int neighbour_dofs_curr_cell = 0;
            for (int i = 0; i < numE; i++){
                neighbour_dofs_curr_cell +=1;
                neighbour_dofs_indices.resize(neighbour_dofs_curr_cell);
                neighbour_dofs_indices[neighbour_dofs_curr_cell-1] = global_indices[i];
            }
            delete[] row;
            delete[] global_indices;
            */
            // Create L_e matrix and transposed L_e matrixfor current cell
            Epetra_Map LeRowMap(n_dofs_curr_cell, 0, epetra_comm);
            Epetra_CrsMatrix L_e(Epetra_DataAccess::Copy, LeRowMap, N);
            const double posOne = 1.0;

            for(int i = 0; i < n_dofs_curr_cell; i++){
                const int col = current_dofs_indices[i];
                L_e.InsertGlobalValues(i, 1, &posOne , &col);
            }
            L_e.FillComplete(basis_rowmap, LeRowMap);

            // Find contribution of element to the JacobianThe root is known to exist due to the
            Epetra_CrsMatrix V_L_e_T(Epetra_DataAccess::Copy, LeRowMap, number_modes);
            Epetra_CrsMatrix V_e_m(Epetra_DataAccess::Copy, basis_rowmap, number_modes);
            EpetraExt::MatrixMatrix::Multiply(L_e, false, pod_basis, false, V_L_e_T, true);
            EpetraExt::MatrixMatrix::Multiply(L_e, true, V_L_e_T, false, V_e_m, true);
            // Add the contribution of the element to the hyper-reduced Jacobian with scaling from the weights
            double scaling = 1.0;
            EpetraExt::MatrixMatrix::Add(V_e_m, false, scaling, hyper_reduced_basis, 1.0);
        }
    }
    hyper_reduced_basis.FillComplete(basis_domainmap, basis_rowmap);
    std::ofstream file("HR_Pod_basis.txt");
    hyper_reduced_basis.Print(file);
    return std::make_shared<Epetra_CrsMatrix>(hyper_reduced_basis);
}

template class HyperReductionDG<PHILIP_DIM, PHILIP_DIM+2>;

} // End of Tests namespace
} // End of PHiLiP namespace