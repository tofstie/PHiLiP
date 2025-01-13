//
// Created by tyson on 09/01/25.
//

#include "hyper_reduced_dg.h"

namespace PHiLiP {

template<int dim, int nstate, typename real, typename MeshType>
DGHyper<dim, nstate, real, MeshType>::DGHyper(
    const Parameters::AllParameters *const parameters_input,
    const unsigned int degree,
    const unsigned int max_degree_input,
    const unsigned int grid_degree_input,
    const std::shared_ptr<Triangulation> triangulation_input)
    : DGBaseState<dim, nstate, real, MeshType>(parameters_input, degree,max_degree_input,grid_degree_input,triangulation_input)
{}
template<int dim, int nstate, typename real, typename MeshType>
void DGHyper<dim, nstate, real, MeshType>::assemble_volume_term_and_build_operators(
    typename dealii::DoFHandler<dim>::active_cell_iterator cell,
    const dealii::types::global_dof_index current_cell_index,
    const std::vector<dealii::types::global_dof_index> &cell_dofs_indices,
    const std::vector<dealii::types::global_dof_index> &metric_dof_indices,
    const unsigned int poly_degree,
    const unsigned int grid_degree,
    OPERATOR::basis_functions<dim, 2 * dim, real> &soln_basis,
    OPERATOR::basis_functions<dim, 2 * dim, real> &flux_basis,
    OPERATOR::local_basis_stiffness<dim, 2 * dim, real> &flux_basis_stiffness,
    OPERATOR::vol_projection_operator<dim, 2 * dim, real> &soln_basis_projection_oper_int,
    OPERATOR::vol_projection_operator<dim, 2 * dim, real> &soln_basis_projection_oper_ext,
    OPERATOR::metric_operators<real, dim, 2 * dim> &metric_oper,
    OPERATOR::mapping_shape_functions<dim, 2 * dim, real> &mapping_basis,
    std::array<std::vector<real>, dim> &mapping_support_points,
    dealii::hp::FEValues<dim, dim> &,
    dealii::hp::FEValues<dim, dim> &,
    const dealii::FESystem<dim, dim> &,
    dealii::Vector<real> &local_rhs_int_cell,
    std::vector<dealii::Tensor<1, dim, real> > &local_auxiliary_RHS,
    const bool /*compute_auxiliary_right_hand_side*/,
    const bool,
    const bool,
    const bool)
{
    // Check if the current cell's poly degree etc is different then previous cell's.
    // If the current cell's poly degree is different, then we recompute the 1D
    // polynomial basis functions. Otherwise, we use the previous values in reference space.
    if(poly_degree != soln_basis.current_degree){
        soln_basis.current_degree = poly_degree;
        flux_basis.current_degree = poly_degree;
        mapping_basis.current_degree  = poly_degree;
        this->reinit_operators_for_cell_residual_loop(poly_degree, poly_degree, grid_degree,
                                                      soln_basis, soln_basis,
                                                      flux_basis, flux_basis,
                                                      flux_basis_stiffness,
                                                      soln_basis_projection_oper_int, soln_basis_projection_oper_ext,
                                                      mapping_basis);
    }
    const dealii::FESystem<dim> &fe_metric = this->high_order_grid->fe_system;
    const unsigned int n_metric_dofs = fe_metric.dofs_per_cell;
    const unsigned int n_grid_nodes  = n_metric_dofs / dim;
    //Rewrite the high_order_grid->volume_nodes in a way we can use sum-factorization on.
    //That is, splitting up the vector by the dimension.
    for(int idim=0; idim<dim; idim++){
        mapping_support_points[idim].resize(n_grid_nodes);
    }
    const std::vector<unsigned int > &index_renumbering = dealii::FETools::hierarchic_to_lexicographic_numbering<dim>(grid_degree);
    for (unsigned int idof = 0; idof< n_metric_dofs; ++idof) {
        const real val = (this->high_order_grid->volume_nodes[metric_dof_indices[idof]]);
        const unsigned int istate = fe_metric.system_to_component_index(idof).first;
        const unsigned int ishape = fe_metric.system_to_component_index(idof).second;
        const unsigned int igrid_node = index_renumbering[ishape];
        mapping_support_points[istate][igrid_node] = val;
    }

    //build the volume metric cofactor matrix and the determinant of the volume metric Jacobian
    //Also, computes the physical volume flux nodes if needed from flag passed to constructor in dg.cpp
    metric_oper.build_volume_metric_operators(
        this->volume_quadrature_collection[poly_degree].size(), n_grid_nodes,
        mapping_support_points,
        mapping_basis,
        this->all_parameters->use_invariant_curl_form);

    assemble_volume_term_strong(
            cell,
            current_cell_index,
            cell_dofs_indices,
            poly_degree,
            soln_basis,
            flux_basis,
            flux_basis_stiffness,
            soln_basis_projection_oper_int,
            metric_oper,
            local_rhs_int_cell);
}

template<int dim, int nstate, typename real, typename MeshType>
void DGHyper<dim, nstate, real, MeshType>::assemble_boundary_term_and_build_operators(
    typename dealii::DoFHandler<dim>::active_cell_iterator,
    const dealii::types::global_dof_index current_cell_index,
    const unsigned int iface,
    const unsigned int boundary_id,
    const real penalty,
    const std::vector<dealii::types::global_dof_index> &cell_dofs_indices,
    const std::vector<dealii::types::global_dof_index> &metric_dof_indices,
    const unsigned int poly_degree,
    const unsigned int grid_degree,
    OPERATOR::basis_functions<dim, 2 * dim, real> &soln_basis,
    OPERATOR::basis_functions<dim, 2 * dim, real> &flux_basis,
    OPERATOR::local_basis_stiffness<dim, 2 * dim, real> &flux_basis_stiffness,
    OPERATOR::vol_projection_operator<dim, 2 * dim, real> &soln_basis_projection_oper_int,
    OPERATOR::vol_projection_operator<dim, 2 * dim, real> &soln_basis_projection_oper_ext,
    OPERATOR::metric_operators<real, dim, 2 * dim> &metric_oper,
    OPERATOR::mapping_shape_functions<dim, 2 * dim, real> &mapping_basis,
    std::array<std::vector<real>, dim> &mapping_support_points,
    dealii::hp::FEFaceValues<dim, dim> &,
    const dealii::FESystem<dim, dim> &,
    dealii::Vector<real> &local_rhs_int_cell,
    std::vector<dealii::Tensor<1, dim, real> > &local_auxiliary_RHS,
    const bool compute_auxiliary_right_hand_side,
    const bool, const bool, const bool)
{
    const dealii::FESystem<dim> &fe_metric = this->high_order_grid->fe_system;
    const unsigned int n_metric_dofs = fe_metric.dofs_per_cell;
    const unsigned int n_grid_nodes  = n_metric_dofs / dim;
    //build the surface metric operators for interior
    metric_oper.build_facet_metric_operators(
        iface,
        this->face_quadrature_collection[poly_degree].size(),
        n_grid_nodes,
        mapping_support_points,
        mapping_basis,
        this->all_parameters->use_invariant_curl_form);

    assemble_boundary_term_strong (
            iface,
            current_cell_index,
            boundary_id, poly_degree, penalty,
            cell_dofs_indices,
            soln_basis,
            flux_basis,
            soln_basis_projection_oper_int,
            metric_oper,
            local_rhs_int_cell);
}

template<int dim, int nstate, typename real, typename MeshType>
void DGHyper<dim, nstate, real, MeshType>::assemble_face_term_and_build_operators(
    typename dealii::DoFHandler<dim>::active_cell_iterator /*cell*/,
    typename dealii::DoFHandler<dim>::active_cell_iterator neighbor_cell,
    const dealii::types::global_dof_index current_cell_index,
    const dealii::types::global_dof_index neighbor_cell_index,
    const unsigned int iface,
    const unsigned int neighbor_iface,
    const real penalty,
    const std::vector<dealii::types::global_dof_index> &current_dofs_indices, const std::vector<dealii::types::global_dof_index> &neighbor_dofs_indices, const std::vector<dealii::types::global_dof_index> &current_metric_dofs_indices, const std::vector<dealii::types::global_dof_index> &neighbor_metric_dofs_indices,
    const unsigned int poly_degree_int,
    const unsigned int poly_degree_ext,
    const unsigned int grid_degree_int,
    const unsigned int grid_degree_ext,
    OPERATOR::basis_functions<dim, 2 * dim, real> &soln_basis_int,
    OPERATOR::basis_functions<dim, 2 * dim, real> &soln_basis_ext,
    OPERATOR::basis_functions<dim, 2 * dim, real> &flux_basis_int,
    OPERATOR::basis_functions<dim, 2 * dim, real> &flux_basis_ext,
    OPERATOR::local_basis_stiffness<dim, 2 * dim, real> &flux_basis_stiffness,
    OPERATOR::vol_projection_operator<dim, 2 * dim, real> &soln_basis_projection_oper_int,
    OPERATOR::vol_projection_operator<dim, 2 * dim, real> &soln_basis_projection_oper_ext,
    OPERATOR::metric_operators<real, dim, 2 * dim> &metric_oper_int,
    OPERATOR::metric_operators<real, dim, 2 * dim> &metric_oper_ext,
    OPERATOR::mapping_shape_functions<dim, 2 * dim, real> &mapping_basis,
    std::array<std::vector<real>, dim> &mapping_support_points,
    dealii::hp::FEFaceValues<dim, dim> &,
    dealii::hp::FEFaceValues<dim, dim> &,
    dealii::Vector<real> &current_cell_rhs,
    dealii::Vector<real> &neighbor_cell_rhs,
    std::vector<dealii::Tensor<1, dim, real> > &current_cell_rhs_aux,
    dealii::LinearAlgebra::distributed::Vector<double> &rhs,
    std::array<dealii::LinearAlgebra::distributed::Vector<double>, dim> &rhs_aux,
    const bool compute_auxiliary_right_hand_side,
    const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R)
{
    const dealii::FESystem<dim> &fe_metric = this->high_order_grid->fe_system;
    const unsigned int n_metric_dofs = fe_metric.dofs_per_cell;
    const unsigned int n_grid_nodes  = n_metric_dofs / dim;
    //build the surface metric operators for interior
    metric_oper_int.build_facet_metric_operators(
        iface,
        this->face_quadrature_collection[poly_degree_int].size(),
        n_grid_nodes,
        mapping_support_points,
        mapping_basis,
        this->all_parameters->use_invariant_curl_form);

    if(poly_degree_ext != soln_basis_ext.current_degree){
        soln_basis_ext.current_degree    = poly_degree_ext;
        flux_basis_ext.current_degree    = poly_degree_ext;
        mapping_basis.current_degree     = poly_degree_ext;
        this->reinit_operators_for_cell_residual_loop(poly_degree_int, poly_degree_ext, grid_degree_ext,
                                                      soln_basis_int, soln_basis_ext,
                                                      flux_basis_int, flux_basis_ext,
                                                      flux_basis_stiffness,
                                                      soln_basis_projection_oper_int, soln_basis_projection_oper_ext,
                                                      mapping_basis);
    }
    std::array<std::vector<real>,dim> mapping_support_points_neigh;
    for(int idim=0; idim<dim; idim++){
        mapping_support_points_neigh[idim].resize(n_grid_nodes);
    }
    const std::vector<unsigned int > &index_renumbering = dealii::FETools::hierarchic_to_lexicographic_numbering<dim>(grid_degree_ext);
    for (unsigned int idof = 0; idof< n_metric_dofs; ++idof) {
        const real val = (this->high_order_grid->volume_nodes[neighbor_metric_dofs_indices[idof]]);
        const unsigned int istate = fe_metric.system_to_component_index(idof).first;
        const unsigned int ishape = fe_metric.system_to_component_index(idof).second;
        const unsigned int igrid_node = index_renumbering[ishape];
        mapping_support_points_neigh[istate][igrid_node] = val;
    }
    //build the metric operators for strong form
    metric_oper_ext.build_volume_metric_operators(
        this->volume_quadrature_collection[poly_degree_ext].size(), n_grid_nodes,
        mapping_support_points_neigh,
        mapping_basis,
        this->all_parameters->use_invariant_curl_form);
    assemble_face_term_strong (
            iface, neighbor_iface,
            current_cell_index,
            neighbor_cell_index,
            poly_degree_int, poly_degree_ext,
            penalty,
            current_dofs_indices, neighbor_dofs_indices,
            soln_basis_int, soln_basis_ext,
            flux_basis_int, flux_basis_ext,
            soln_basis_projection_oper_int, soln_basis_projection_oper_ext,
            metric_oper_int, metric_oper_ext,
            current_cell_rhs, neighbor_cell_rhs);
    // add local contribution from neighbor cell to global vector
    const unsigned int n_dofs_neigh_cell = this->fe_collection[neighbor_cell->active_fe_index()].n_dofs_per_cell();
    for (unsigned int i=0; i<n_dofs_neigh_cell; ++i) {
        rhs[neighbor_dofs_indices[i]] += neighbor_cell_rhs[i];
    }
}

/****************************************************
*
* PRIMARY EQUATIONS STRONG FORM
*
****************************************************/

template<int dim, int nstate, typename real, typename MeshType>
void DGHyper<dim, nstate, real, MeshType>::assemble_volume_term_strong(
    typename dealii::DoFHandler<dim>::active_cell_iterator cell,
    const dealii::types::global_dof_index current_cell_index,
    const std::vector<dealii::types::global_dof_index> &cell_dofs_indices,
    const unsigned int poly_degree, OPERATOR::basis_functions<dim, 2 * dim, real> &soln_basis,
    OPERATOR::basis_functions<dim, 2 * dim, real> &flux_basis,
    OPERATOR::local_basis_stiffness<dim, 2 * dim, real> &flux_basis_stiffness,
    OPERATOR::vol_projection_operator<dim, 2 * dim, real> &soln_basis_projection_oper,
    OPERATOR::metric_operators<real, dim, 2 * dim> &metric_oper,
    dealii::Vector<real> &local_rhs_int_cell)
{
    (void) current_cell_index;

    const unsigned int n_quad_pts  = this->volume_quadrature_collection[poly_degree].size();
    const unsigned int n_dofs_cell = this->fe_collection[poly_degree].dofs_per_cell;
    const unsigned int n_shape_fns = n_dofs_cell / nstate;
    const unsigned int n_quad_pts_1D  = this->oneD_quadrature_collection[poly_degree].size();
    assert(n_quad_pts == pow(n_quad_pts_1D, dim));
    const std::vector<double> &vol_quad_weights = this->volume_quadrature_collection[poly_degree].get_weights();
    const std::vector<double> &oneD_vol_quad_weights = this->oneD_quadrature_collection[poly_degree].get_weights();

    AssertDimension (n_dofs_cell, cell_dofs_indices.size());
}

template<int dim, int nstate, typename real, typename MeshType>
void DGHyper<dim, nstate, real, MeshType>::assemble_boundary_term_strong(
    const unsigned int iface,
    const dealii::types::global_dof_index current_cell_index,
    const unsigned int boundary_id,
    const unsigned int poly_degree,
    const real penalty,
    const std::vector<dealii::types::global_dof_index> &dof_indices,
    OPERATOR::basis_functions<dim, 2 * dim, real> &soln_basis,
    OPERATOR::basis_functions<dim, 2 * dim, real> &flux_basis,
    OPERATOR::vol_projection_operator<dim, 2 * dim, real> &soln_basis_projection_oper,
    OPERATOR::metric_operators<real, dim, 2 * dim> &metric_oper,
    dealii::Vector<real> &local_rhs_cell)
{}

template<int dim, int nstate, typename real, typename MeshType>
void DGHyper<dim, nstate, real, MeshType>::assemble_face_term_strong(
    const unsigned int iface,
    const unsigned int neighbor_iface,
    const dealii::types::global_dof_index current_cell_index,
    const dealii::types::global_dof_index neighbor_cell_index,
    const unsigned int poly_degree_int,
    const unsigned int poly_degree_ext,
    const real penalty,
    const std::vector<dealii::types::global_dof_index> &dof_indices_int,
    const std::vector<dealii::types::global_dof_index> &dof_indices_ext,
    OPERATOR::basis_functions<dim, 2 * dim, real> &soln_basis_int,
    OPERATOR::basis_functions<dim, 2 * dim, real> &soln_basis_ext,
    OPERATOR::basis_functions<dim, 2 * dim, real> &flux_basis_int,
    OPERATOR::basis_functions<dim, 2 * dim, real> &flux_basis_ext,
    OPERATOR::vol_projection_operator<dim, 2 * dim, real> &soln_basis_projection_oper_int,
    OPERATOR::vol_projection_operator<dim, 2 * dim, real> &soln_basis_projection_oper_ext,
    OPERATOR::metric_operators<real, dim, 2 * dim> &metric_oper_int,
    OPERATOR::metric_operators<real, dim, 2 * dim> &metric_oper_ext,
    dealii::Vector<real> &local_rhs_int_cell,
    dealii::Vector<real> &local_rhs_ext_cell)
{}

template<int dim, int nstate, typename real, typename MeshType>
void DGHyper<dim, nstate, real, MeshType>::assemble_hyper_reduced_derivative() {
    /// GET ids
    /// Filter Test Basis using Row ids
    /// Create HyperReduced Mass Matrix
    /// Create Hyper Reduction Projection Matrix
    /// Create Hyper Reduction Volume Operator
    /// Create EB Hyper Reduction
}



}
