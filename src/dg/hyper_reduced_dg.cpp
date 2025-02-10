//
// Created by tyson on 09/01/25.
//

#include "hyper_reduced_dg.hpp"

namespace PHiLiP {

template<int dim, int nstate, typename real, typename MeshType>
DGHyper<dim, nstate, real, MeshType>::DGHyper(
    const Parameters::AllParameters *const parameters_input,
    const unsigned int degree,
    const unsigned int max_degree_input,
    const unsigned int grid_degree_input,
    const std::shared_ptr<Triangulation> triangulation_input)
    : DGBaseState<dim,nstate,real,MeshType>::DGBaseState(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input)
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

        // Fetch the modal soln coefficients and the modal auxiliary soln coefficients
    // We immediately separate them by state as to be able to use sum-factorization
    // in the interpolation operator. If we left it by n_dofs_cell, then the matrix-vector
    // mult would sum the states at the quadrature point.
    std::array<std::vector<real>,nstate> soln_coeff;
    std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> aux_soln_coeff;
    for (unsigned int idof = 0; idof < n_dofs_cell; ++idof) {
        const unsigned int istate = this->fe_collection[poly_degree].system_to_component_index(idof).first;
        const unsigned int ishape = this->fe_collection[poly_degree].system_to_component_index(idof).second;
        if(ishape == 0)
            soln_coeff[istate].resize(n_shape_fns);
        soln_coeff[istate][ishape] = DGBase<dim,real,MeshType>::solution(cell_dofs_indices[idof]);
        for(int idim=0; idim<dim; idim++){
            if(ishape == 0)
                aux_soln_coeff[istate][idim].resize(n_shape_fns);
            if(this->use_auxiliary_eq){
                aux_soln_coeff[istate][idim][ishape] = DGBase<dim,real,MeshType>::auxiliary_solution[idim](cell_dofs_indices[idof]);
            }
            else{
                aux_soln_coeff[istate][idim][ishape] = 0.0;
            }
        }
    }
    std::array<std::vector<real>,nstate> soln_at_q;
    std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> aux_soln_at_q; //auxiliary sol at flux nodes
    std::vector<std::array<real,nstate>> soln_at_q_for_max_CFL(n_quad_pts);//Need soln written in a different for to use pre-existing max CFL function
    // Interpolate each state to the quadrature points using sum-factorization
    // with the basis functions in each reference direction.
    for(int istate=0; istate<nstate; istate++){
        soln_at_q[istate].resize(n_quad_pts);
        soln_basis.matrix_vector_mult_1D(soln_coeff[istate], soln_at_q[istate],
                                         soln_basis.oneD_vol_operator);
        for(int idim=0; idim<dim; idim++){
            aux_soln_at_q[istate][idim].resize(n_quad_pts);
            soln_basis.matrix_vector_mult_1D(aux_soln_coeff[istate][idim], aux_soln_at_q[istate][idim],
                                             soln_basis.oneD_vol_operator);
        }
        for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
            soln_at_q_for_max_CFL[iquad][istate] = soln_at_q[istate][iquad];
        }
    }

    // For pseudotime, we need to compute the time_scaled_solution.
    // Thus, we need to evaluate the max_dt_cell (as previously done in dg/weak_dg.cpp -> assemble_volume_term_explicit)
    // Get max artificial dissipation
    real max_artificial_diss = 0.0;
    const unsigned int n_dofs_arti_diss = this->fe_q_artificial_dissipation.dofs_per_cell;
    typename dealii::DoFHandler<dim>::active_cell_iterator artificial_dissipation_cell(
        this->triangulation.get(), cell->level(), cell->index(), &(this->dof_handler_artificial_dissipation));
    std::vector<dealii::types::global_dof_index> dof_indices_artificial_dissipation(n_dofs_arti_diss);
    artificial_dissipation_cell->get_dof_indices (dof_indices_artificial_dissipation);
    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
        real artificial_diss_coeff_at_q = 0.0;
        if ( this->all_parameters->artificial_dissipation_param.add_artificial_dissipation ) {
            const dealii::Point<dim,real> point = this->volume_quadrature_collection[poly_degree].point(iquad);
            for (unsigned int idof=0; idof<n_dofs_arti_diss; ++idof) {
                const unsigned int index = dof_indices_artificial_dissipation[idof];
                artificial_diss_coeff_at_q += this->artificial_dissipation_c0[index] * this->fe_q_artificial_dissipation.shape_value(idof, point);
            }
            max_artificial_diss = std::max(artificial_diss_coeff_at_q, max_artificial_diss);
        }
    }
    // Get max_dt_cell for time_scaled_solution with pseudotime
    real cell_volume_estimate = 0.0;
    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
        cell_volume_estimate += metric_oper.det_Jac_vol[iquad] * vol_quad_weights[iquad];
    }
    const real cell_volume = cell_volume_estimate;
    const real diameter = cell->diameter();
    const real cell_diameter = cell_volume / std::pow(diameter,dim-1);
    const real cell_radius = 0.5 * cell_diameter;
    this->cell_volume[current_cell_index] = cell_volume;
    this->max_dt_cell[current_cell_index] = this->evaluate_CFL ( soln_at_q_for_max_CFL, max_artificial_diss, cell_radius, poly_degree);

    //get entropy projected variables
    std::array<std::vector<real>,nstate> entropy_var_at_q;
    std::array<std::vector<real>,nstate> projected_entropy_var_at_q;
    if (this->all_parameters->use_split_form || this->all_parameters->use_curvilinear_split_form){
        for(int istate=0; istate<nstate; istate++){
            entropy_var_at_q[istate].resize(n_quad_pts);
            projected_entropy_var_at_q[istate].resize(n_quad_pts);
        }
        for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
            std::array<real,nstate> soln_state;
            for(int istate=0; istate<nstate; istate++){
                soln_state[istate] = soln_at_q[istate][iquad];
            }
            std::array<real,nstate> entropy_var;
            entropy_var = this->pde_physics_double->compute_entropy_variables(soln_state);
            for(int istate=0; istate<nstate; istate++){
                entropy_var_at_q[istate][iquad] = entropy_var[istate];
            }
        }
        for(int istate=0; istate<nstate; istate++){
            std::vector<real> entropy_var_coeff(n_shape_fns);
            soln_basis_projection_oper.matrix_vector_mult_1D(entropy_var_at_q[istate],
                                                             entropy_var_coeff,
                                                             soln_basis_projection_oper.oneD_vol_operator);
            if(this->all_parameters->reduced_order_param.entropy_variables_in_snapshots) {
                for(unsigned int i_shape_fns = 0; i_shape_fns<n_shape_fns; i_shape_fns++){

                    entropy_var_coeff[i_shape_fns] = this->projected_entropy[cell_dofs_indices[istate*n_shape_fns+i_shape_fns]];
                }
            }

            soln_basis.matrix_vector_mult_1D(entropy_var_coeff,
                                             projected_entropy_var_at_q[istate],
                                             soln_basis.oneD_vol_operator);
            if (isnan(projected_entropy_var_at_q[0][0])){
                std::cout << "Nan" << std::endl;
            }
        }
    }


    //Compute the physical fluxes, then convert them into reference fluxes.
    //From the paper: Cicchino, Alexander, et al. "Provably stable flux reconstruction high-order methods on curvilinear elements." Journal of Computational Physics 463 (2022): 111259.
    //For conservative DG, we compute the reference flux as per Eq. (9), to then recover the second volume integral in Eq. (17).
    //For curvilinear split-form in Eq. (22), we apply a two-pt flux of the metric-cofactor matrix on the matrix operator constructed by the entropy stable/conservtive 2pt flux.
    std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> conv_ref_flux_at_q;
    std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> diffusive_ref_flux_at_q;
    std::array<std::vector<real>,nstate> source_at_q;
    std::array<std::vector<real>,nstate> physical_source_at_q;

    // The matrix of two-pt fluxes for Hadamard products
    std::array<std::array<dealii::FullMatrix<real>,dim>,nstate> conv_ref_2pt_flux_at_q;
    //Hadamard tensor-product sparsity pattern
    std::vector<std::array<unsigned int,dim>> Hadamard_rows_sparsity(n_quad_pts * n_quad_pts_1D);//size n^{d+1}
    std::vector<std::array<unsigned int,dim>> Hadamard_columns_sparsity(n_quad_pts * n_quad_pts_1D);
    //allocate reference 2pt flux for Hadamard product
    if (this->all_parameters->use_split_form || this->all_parameters->use_curvilinear_split_form){
        for(int istate=0; istate<nstate; istate++){
            for(int idim=0; idim<dim; idim++){
                conv_ref_2pt_flux_at_q[istate][idim].reinit(n_quad_pts, n_quad_pts_1D);//size n^d x n
            }
        }
        //extract the dof pairs that give non-zero entries for each direction
        //to use the "sum-factorized" Hadamard product.
        flux_basis.sum_factorized_Hadamard_sparsity_pattern(n_quad_pts_1D, n_quad_pts_1D, Hadamard_rows_sparsity, Hadamard_columns_sparsity);
    }


    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
        //extract soln and auxiliary soln at quad pt to be used in physics
        std::array<real,nstate> soln_state;
        std::array<dealii::Tensor<1,dim,real>,nstate> aux_soln_state;
        for(int istate=0; istate<nstate; istate++){
            soln_state[istate] = soln_at_q[istate][iquad];
            for(int idim=0; idim<dim; idim++){
                aux_soln_state[istate][idim] = aux_soln_at_q[istate][idim][iquad];
            }
        }

        // Copy Metric Cofactor in a way can use for transforming Tensor Blocks to reference space
        // The way it is stored in metric_operators is to use sum-factorization in each direction,
        // but here it is cleaner to apply a reference transformation in each Tensor block returned by physics.
        dealii::Tensor<2,dim,real> metric_cofactor;
        for(int idim=0; idim<dim; idim++){
            for(int jdim=0; jdim<dim; jdim++){
                metric_cofactor[idim][jdim] = metric_oper.metric_cofactor_vol[idim][jdim][iquad];
            }
        }

        // Evaluate physical convective flux
        // If 2pt flux, transform to reference at construction to improve performance.
        // We technically use a REFERENCE 2pt flux for all entropy stable schemes.
        std::array<dealii::Tensor<1,dim,real>,nstate> conv_phys_flux;
        if (this->all_parameters->use_split_form || this->all_parameters->use_curvilinear_split_form){
            //get the soln for iquad from projected entropy variables
            std::array<real,nstate> entropy_var;
            for(int istate=0; istate<nstate; istate++){
                entropy_var[istate] = projected_entropy_var_at_q[istate][iquad];
            }
            soln_state = this->pde_physics_double->compute_conservative_variables_from_entropy_variables (entropy_var);

            //loop over all the non-zero entries for "sum-factorized" Hadamard product that corresponds to the iquad.
            for(unsigned int row_index = iquad * n_quad_pts_1D, column_index = 0;
               // Hadamard_rows_sparsity[row_index][0] == iquad;
                column_index < n_quad_pts_1D;
                row_index++, column_index++){

                if(Hadamard_rows_sparsity[row_index][0] != iquad){
                    pcout<<"The volume Hadamard rows sparsity pattern does not match. Aborting..."<<std::endl;
                    std::abort();
                }

                // Copy Metric Cofactor in a way can use for transforming Tensor Blocks to reference space
                // The way it is stored in metric_operators is to use sum-factorization in each direction,
                // but here it is cleaner to apply a reference transformation in each Tensor block returned by physics.

                for(int ref_dim=0; ref_dim<dim; ref_dim++){
                    const unsigned int flux_quad = Hadamard_columns_sparsity[row_index][ref_dim];//extract flux_quad pt that corresponds to a non-zero entry for Hadamard product.

                    dealii::Tensor<2,dim,real> metric_cofactor_flux_basis;
                    for(int idim=0; idim<dim; idim++){
                        for(int jdim=0; jdim<dim; jdim++){
                            metric_cofactor_flux_basis[idim][jdim] = metric_oper.metric_cofactor_vol[idim][jdim][flux_quad];
                        }
                    }
                    std::array<real,nstate> soln_state_flux_basis;
                    std::array<real,nstate> entropy_var_flux_basis;
                    for(int istate=0; istate<nstate; istate++){
                        entropy_var_flux_basis[istate] = projected_entropy_var_at_q[istate][flux_quad];
                    }
                    soln_state_flux_basis = this->pde_physics_double->compute_conservative_variables_from_entropy_variables (entropy_var_flux_basis);

                    //Compute the physical flux
                    std::array<dealii::Tensor<1,dim,real>,nstate> conv_phys_flux_2pt;
                    conv_phys_flux_2pt = this->pde_physics_double->convective_numerical_split_flux(soln_state, soln_state_flux_basis);

                    for(int istate=0; istate<nstate; istate++){
                        dealii::Tensor<1,dim,real> conv_ref_flux_2pt;
                        //For each state, transform the physical flux to a reference flux.
                        metric_oper.transform_physical_to_reference(
                            conv_phys_flux_2pt[istate],
                            0.5*(metric_cofactor + metric_cofactor_flux_basis),
                            conv_ref_flux_2pt);
                        //write into reference Hadamard flux matrix
                        conv_ref_2pt_flux_at_q[istate][ref_dim][iquad][column_index] = conv_ref_flux_2pt[ref_dim];
                    }
                }
            }
        }
        else{
            //Compute the physical flux
            conv_phys_flux = this->pde_physics_double->convective_flux (soln_state);
        }

        //Diffusion
        std::array<dealii::Tensor<1,dim,real>,nstate> diffusive_phys_flux;
        //Compute the physical dissipative flux
        diffusive_phys_flux = this->pde_physics_double->dissipative_flux(soln_state, aux_soln_state, current_cell_index);

        // Manufactured source
        std::array<real,nstate> manufactured_source;
        if(this->all_parameters->manufactured_convergence_study_param.manufactured_solution_param.use_manufactured_source_term) {
            dealii::Point<dim,real> vol_flux_node;
            for(int idim=0; idim<dim; idim++){
                vol_flux_node[idim] = metric_oper.flux_nodes_vol[idim][iquad];
            }
            //compute the manufactured source
            manufactured_source = this->pde_physics_double->source_term (vol_flux_node, soln_state, this->current_time, current_cell_index);
        }

        // Physical source
        std::array<real,nstate> physical_source;
        if(this->pde_physics_double->has_nonzero_physical_source) {
            dealii::Point<dim,real> vol_flux_node;
            for(int idim=0; idim<dim; idim++){
                vol_flux_node[idim] = metric_oper.flux_nodes_vol[idim][iquad];
            }
            //compute the physical source
            physical_source = this->pde_physics_double->physical_source_term (vol_flux_node, soln_state, aux_soln_state, current_cell_index);
        }

        //Write the values in a way that we can use sum-factorization on.
        for(int istate=0; istate<nstate; istate++){
            dealii::Tensor<1,dim,real> conv_ref_flux;
            dealii::Tensor<1,dim,real> diffusive_ref_flux;
            //Trnasform to reference fluxes
            if (this->all_parameters->use_split_form || this->all_parameters->use_curvilinear_split_form){
                //Do Nothing.
                //I am leaving this block here so the diligent reader
                //remembers that, for entropy stable schemes, we construct
                //a REFERENCE two-point flux at construction, where the physical
                //to reference transformation was done by splitting the metric cofactor.
            }
            else{
                //transform the conservative convective physical flux to reference space
                metric_oper.transform_physical_to_reference(
                    conv_phys_flux[istate],
                    metric_cofactor,
                    conv_ref_flux);
            }
            //transform the dissipative flux to reference space
            metric_oper.transform_physical_to_reference(
                diffusive_phys_flux[istate],
                metric_cofactor,
                diffusive_ref_flux);

            //Write the data in a way that we can use sum-factorization on.
            //Since sum-factorization improves the speed for matrix-vector multiplications,
            //We need the values to have their inner elements be vectors.
            for(int idim=0; idim<dim; idim++){
                //allocate
                if(iquad == 0){
                    conv_ref_flux_at_q[istate][idim].resize(n_quad_pts);
                    diffusive_ref_flux_at_q[istate][idim].resize(n_quad_pts);
                }
                //write data
                if (this->all_parameters->use_split_form || this->all_parameters->use_curvilinear_split_form){
                    //Do nothing because written in a Hadamard product sum-factorized form above.
                }
                else{
                    conv_ref_flux_at_q[istate][idim][iquad] = conv_ref_flux[idim];
                }

                diffusive_ref_flux_at_q[istate][idim][iquad] = diffusive_ref_flux[idim];
            }
            if(this->all_parameters->manufactured_convergence_study_param.manufactured_solution_param.use_manufactured_source_term) {
                if(iquad == 0){
                    source_at_q[istate].resize(n_quad_pts);
                }
                source_at_q[istate][iquad] = manufactured_source[istate];
            }
            if(this->pde_physics_double->has_nonzero_physical_source) {
                if(iquad == 0){
                    physical_source_at_q[istate].resize(n_quad_pts);
                }
                physical_source_at_q[istate][iquad] = physical_source[istate];
            }
        }
    }

    // Get a flux basis reference gradient operator in a sum-factorized Hadamard product sparse form. Then apply the divergence.
    std::array<dealii::FullMatrix<real>,dim> flux_basis_stiffness_skew_symm_oper_sparse;
    if (this->all_parameters->use_split_form || this->all_parameters->use_curvilinear_split_form){
        for(int idim=0; idim<dim; idim++){
            flux_basis_stiffness_skew_symm_oper_sparse[idim].reinit(n_quad_pts, n_quad_pts_1D);
        }
        flux_basis.sum_factorized_Hadamard_basis_assembly(n_quad_pts_1D, n_quad_pts_1D,
                                                          Hadamard_rows_sparsity, Hadamard_columns_sparsity,
                                                          flux_basis_stiffness.oneD_skew_symm_vol_oper,
                                                          oneD_vol_quad_weights,
                                                          flux_basis_stiffness_skew_symm_oper_sparse);
    }
    /*
    char zero = '0';
    std::cout << current_cell_index << std::endl;
    flux_basis_stiffness_skew_symm_oper_sparse[0].print_formatted(std::cout,3,true,0,&zero);
    */
    //For each state we:
    //  1. Compute reference divergence.
    //  2. Then compute and write the rhs for the given state.
    for(int istate=0; istate<nstate; istate++){

        //Compute reference divergence of the reference fluxes.
        std::vector<real> conv_flux_divergence(n_quad_pts);
        std::vector<real> diffusive_flux_divergence(n_quad_pts);

        if (this->all_parameters->use_split_form || this->all_parameters->use_curvilinear_split_form){
            //2pt flux Hadamard Product, and then multiply by vector of ones scaled by 1.
            // Same as the volume term in Eq. (15) in Chan, Jesse. "Skew-symmetric entropy stable modal discontinuous Galerkin formulations." Journal of Scientific Computing 81.1 (2019): 459-485. but,
            // where we use the reference skew-symmetric stiffness operator of the flux basis for the Q operator and the reference two-point flux as to make use of Alex's Hadamard product
            // sum-factorization type algorithm that exploits the structure of the flux basis in the reference space to have O(n^{d+1}).

            for(int ref_dim=0; ref_dim<dim; ref_dim++){
                dealii::FullMatrix<real> divergence_ref_flux_Hadamard_product(n_quad_pts, n_quad_pts_1D);
                flux_basis.Hadamard_product(flux_basis_stiffness_skew_symm_oper_sparse[ref_dim], conv_ref_2pt_flux_at_q[istate][ref_dim], divergence_ref_flux_Hadamard_product);
                //char zero = '0';
                //divergence_ref_flux_Hadamard_product.print_formatted(std::cout, 3, true, 0, &zero ,1.,0.);
                //Hadamard product times the vector of ones.
                for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                    if(ref_dim == 0){
                        conv_flux_divergence[iquad] = 0.0;
                    }
                    for(unsigned int iquad_1D=0; iquad_1D<n_quad_pts_1D; iquad_1D++){
                        conv_flux_divergence[iquad] += divergence_ref_flux_Hadamard_product[iquad][iquad_1D];
                    }
                }
            }

        }
        else{
            //Reference divergence of the reference convective flux.
            flux_basis.divergence_matrix_vector_mult_1D(conv_ref_flux_at_q[istate], conv_flux_divergence,
                                                        flux_basis.oneD_vol_operator,
                                                        flux_basis.oneD_grad_operator);
        }
        //Reference divergence of the reference diffusive flux.
        flux_basis.divergence_matrix_vector_mult_1D(diffusive_ref_flux_at_q[istate], diffusive_flux_divergence,
                                                    flux_basis.oneD_vol_operator,
                                                    flux_basis.oneD_grad_operator);


        // Strong form
        // The right-hand side sends all the term to the side of the source term
        // Therefore,
        // \divergence ( Fconv + Fdiss ) = source
        // has the right-hand side
        // rhs = - \divergence( Fconv + Fdiss ) + source
        // Since we have done an integration by parts, the volume term resulting from the divergence of Fconv and Fdiss
        // is negative. Therefore, negative of negative means we add that volume term to the right-hand-side
        std::vector<real> rhs(n_shape_fns);

        // Convective
        if (this->all_parameters->use_split_form || this->all_parameters->use_curvilinear_split_form){
            std::vector<real> ones(n_quad_pts, 1.0);
            soln_basis.inner_product_1D(conv_flux_divergence, ones, rhs, soln_basis.oneD_vol_operator, false, -1.0);
        }
        else {
            soln_basis.inner_product_1D(conv_flux_divergence, vol_quad_weights, rhs, soln_basis.oneD_vol_operator, false, -1.0);
        }

        // Diffusive
        // Note that for diffusion, the negative is defined in the physics. Since we used the auxiliary
        // variable, put a negative here.
        soln_basis.inner_product_1D(diffusive_flux_divergence, vol_quad_weights, rhs, soln_basis.oneD_vol_operator, true, -1.0);

        // Manufactured source
        if(this->all_parameters->manufactured_convergence_study_param.manufactured_solution_param.use_manufactured_source_term) {
            std::vector<real> JxW(n_quad_pts);
            for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                JxW[iquad] = vol_quad_weights[iquad] * metric_oper.det_Jac_vol[iquad];
            }
            soln_basis.inner_product_1D(source_at_q[istate], JxW, rhs, soln_basis.oneD_vol_operator, true, 1.0);
        }

        // Physical source
        if(this->pde_physics_double->has_nonzero_physical_source) {
            std::vector<real> JxW(n_quad_pts);
            for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                JxW[iquad] = vol_quad_weights[iquad] * metric_oper.det_Jac_vol[iquad];
            }
            soln_basis.inner_product_1D(physical_source_at_q[istate], JxW, rhs, soln_basis.oneD_vol_operator, true, 1.0);
        }

        for(unsigned int ishape=0; ishape<n_shape_fns; ishape++){
            local_rhs_int_cell(istate*n_shape_fns + ishape) += rhs[ishape];
        }

    }
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
{
      (void) current_cell_index;

    const unsigned int n_face_quad_pts  = this->face_quadrature_collection[poly_degree].size();
    const unsigned int n_quad_pts_vol   = this->volume_quadrature_collection[poly_degree].size();
    const unsigned int n_quad_pts_1D    = this->oneD_quadrature_collection[poly_degree].size();
    const unsigned int n_dofs = this->fe_collection[poly_degree].dofs_per_cell;
    const unsigned int n_shape_fns = n_dofs / nstate;
    const std::vector<double> &face_quad_weights = this->face_quadrature_collection[poly_degree].get_weights();

    AssertDimension (n_dofs, dof_indices.size());

    // Fetch the modal soln coefficients and the modal auxiliary soln coefficients
    // We immediately separate them by state as to be able to use sum-factorization
    // in the interpolation operator. If we left it by n_dofs_cell, then the matrix-vector
    // mult would sum the states at the quadrature point.
    std::array<std::vector<real>,nstate> soln_coeff;
    std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> aux_soln_coeff;
    for (unsigned int idof = 0; idof < n_dofs; ++idof) {
        const unsigned int istate = this->fe_collection[poly_degree].system_to_component_index(idof).first;
        const unsigned int ishape = this->fe_collection[poly_degree].system_to_component_index(idof).second;
        // allocate
        if(ishape == 0){
            soln_coeff[istate].resize(n_shape_fns);
        }
        // solve
        soln_coeff[istate][ishape] = DGBase<dim,real,MeshType>::solution(dof_indices[idof]);
        for(int idim=0; idim<dim; idim++){
            //allocate
            if(ishape == 0){
                aux_soln_coeff[istate][idim].resize(n_shape_fns);
            }
            //solve
            if(this->use_auxiliary_eq){
                aux_soln_coeff[istate][idim][ishape] = DGBase<dim,real,MeshType>::auxiliary_solution[idim](dof_indices[idof]);
            }
            else{
                aux_soln_coeff[istate][idim][ishape] = 0.0;
            }
        }
    }
    // Interpolate the modal coefficients to the volume cubature nodes.
    std::array<std::vector<real>,nstate> soln_at_vol_q;
    std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> aux_soln_at_vol_q;
    // Interpolate modal soln coefficients to the facet.
    std::array<std::vector<real>,nstate> soln_at_surf_q;
    std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> aux_soln_at_surf_q;
    for(int istate=0; istate<nstate; ++istate){
        //allocate
        soln_at_vol_q[istate].resize(n_quad_pts_vol);
        //solve soln at volume cubature nodes
        soln_basis.matrix_vector_mult_1D(soln_coeff[istate], soln_at_vol_q[istate],
                                         soln_basis.oneD_vol_operator);

        //allocate
        soln_at_surf_q[istate].resize(n_face_quad_pts);
        //solve soln at facet cubature nodes
        soln_basis.matrix_vector_mult_surface_1D(iface,
                                                 soln_coeff[istate], soln_at_surf_q[istate],
                                                 soln_basis.oneD_surf_operator,
                                                 soln_basis.oneD_vol_operator);

        for(int idim=0; idim<dim; idim++){
            //alocate
            aux_soln_at_vol_q[istate][idim].resize(n_quad_pts_vol);
            //solve auxiliary soln at volume cubature nodes
            soln_basis.matrix_vector_mult_1D(aux_soln_coeff[istate][idim], aux_soln_at_vol_q[istate][idim],
                                             soln_basis.oneD_vol_operator);

            //allocate
            aux_soln_at_surf_q[istate][idim].resize(n_face_quad_pts);
            //solve auxiliary soln at facet cubature nodes
            soln_basis.matrix_vector_mult_surface_1D(iface,
                                                     aux_soln_coeff[istate][idim], aux_soln_at_surf_q[istate][idim],
                                                     soln_basis.oneD_surf_operator,
                                                     soln_basis.oneD_vol_operator);
        }
    }

    // Get volume reference fluxes and interpolate them to the facet.
    // Compute reference volume fluxes in both interior and exterior cells.

    // First we do interior.
    std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> conv_ref_flux_at_vol_q;
    std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> diffusive_ref_flux_at_vol_q;
    for (unsigned int iquad=0; iquad<n_quad_pts_vol; ++iquad)
    {
        // Copy Metric Cofactor in a way can use for transforming Tensor Blocks to reference space
        // The way it is stored in metric_operators is to use sum-factorization in each direction,
        // but here it is cleaner to apply a reference transformation in each Tensor block returned by physics.
        dealii::Tensor<2,dim,real> metric_cofactor_vol;
        for(int idim=0; idim<dim; idim++){
            for(int jdim=0; jdim<dim; jdim++){
                metric_cofactor_vol[idim][jdim] = metric_oper.metric_cofactor_vol[idim][jdim][iquad];
            }
        }
        std::array<real,nstate> soln_state;
        std::array<dealii::Tensor<1,dim,real>,nstate> aux_soln_state;
        for(int istate=0; istate<nstate; istate++){
            soln_state[istate] = soln_at_vol_q[istate][iquad];
            for(int idim=0; idim<dim; idim++){
                aux_soln_state[istate][idim] = aux_soln_at_vol_q[istate][idim][iquad];
            }
        }

        // Evaluate physical convective flux
        std::array<dealii::Tensor<1,dim,real>,nstate> conv_phys_flux;
        if(!this->all_parameters->use_split_form && !this->all_parameters->use_curvilinear_split_form){
            conv_phys_flux = this->pde_physics_double->convective_flux (soln_state);
        }

        // Compute the physical dissipative flux
        std::array<dealii::Tensor<1,dim,real>,nstate> diffusive_phys_flux;
        diffusive_phys_flux = this->pde_physics_double->dissipative_flux(soln_state, aux_soln_state, current_cell_index);

        // Write the values in a way that we can use sum-factorization on.
        for(int istate=0; istate<nstate; istate++){
            dealii::Tensor<1,dim,real> conv_ref_flux;
            dealii::Tensor<1,dim,real> diffusive_ref_flux;
            // transform the conservative convective physical flux to reference space
            if(!this->all_parameters->use_split_form && !this->all_parameters->use_curvilinear_split_form){
                metric_oper.transform_physical_to_reference(
                    conv_phys_flux[istate],
                    metric_cofactor_vol,
                    conv_ref_flux);
            }
            // transform the dissipative flux to reference space
            metric_oper.transform_physical_to_reference(
                diffusive_phys_flux[istate],
                metric_cofactor_vol,
                diffusive_ref_flux);

            // Write the data in a way that we can use sum-factorization on.
            // Since sum-factorization improves the speed for matrix-vector multiplications,
            // We need the values to have their inner elements be vectors.
            for(int idim=0; idim<dim; idim++){
                //allocate
                if(iquad == 0){
                    conv_ref_flux_at_vol_q[istate][idim].resize(n_quad_pts_vol);
                    diffusive_ref_flux_at_vol_q[istate][idim].resize(n_quad_pts_vol);
                }
                //write data
                if(!this->all_parameters->use_split_form && !this->all_parameters->use_curvilinear_split_form){
                    conv_ref_flux_at_vol_q[istate][idim][iquad] = conv_ref_flux[idim];
                }

                diffusive_ref_flux_at_vol_q[istate][idim][iquad] = diffusive_ref_flux[idim];
            }
        }
    }
    // Interpolate the volume reference fluxes to the facet.
    // And do the dot product with the UNIT REFERENCE normal.
    // Since we are computing a dot product with the unit reference normal,
    // we exploit the fact that the unit reference normal has a value of 0 in all reference directions except
    // the outward reference normal dircetion.
    const dealii::Tensor<1,dim,double> unit_ref_normal_int = dealii::GeometryInfo<dim>::unit_normal_vector[iface];
    const int dim_not_zero = iface / 2;//reference direction of face integer division

    std::array<std::vector<real>,nstate> conv_int_vol_ref_flux_interp_to_face_dot_ref_normal;
    std::array<std::vector<real>,nstate> diffusive_int_vol_ref_flux_interp_to_face_dot_ref_normal;
    for(int istate=0; istate<nstate; istate++){
        //allocate
        conv_int_vol_ref_flux_interp_to_face_dot_ref_normal[istate].resize(n_face_quad_pts);
        diffusive_int_vol_ref_flux_interp_to_face_dot_ref_normal[istate].resize(n_face_quad_pts);

        //solve
        //Note, since the normal is zero in all other reference directions, we only have to interpolate one given reference direction to the facet

        //interpolate reference volume convective flux to the facet, and apply unit reference normal as scaled by 1.0 or -1.0
        if(!this->all_parameters->use_split_form && !this->all_parameters->use_curvilinear_split_form){
            flux_basis.matrix_vector_mult_surface_1D(iface,
                                                     conv_ref_flux_at_vol_q[istate][dim_not_zero],
                                                     conv_int_vol_ref_flux_interp_to_face_dot_ref_normal[istate],
                                                     flux_basis.oneD_surf_operator,//the flux basis interpolates from the flux nodes
                                                     flux_basis.oneD_vol_operator,
                                                     false, unit_ref_normal_int[dim_not_zero]);//don't add to previous value, scale by unit_normal int
        }

        //interpolate reference volume dissipative flux to the facet, and apply unit reference normal as scaled by 1.0 or -1.0
        flux_basis.matrix_vector_mult_surface_1D(iface,
                                                 diffusive_ref_flux_at_vol_q[istate][dim_not_zero],
                                                 diffusive_int_vol_ref_flux_interp_to_face_dot_ref_normal[istate],
                                                 flux_basis.oneD_surf_operator,
                                                 flux_basis.oneD_vol_operator,
                                                 false, unit_ref_normal_int[dim_not_zero]);
    }

    //Note that for entropy-dissipation and entropy stability, the conservative variables
    //are functions of projected entropy variables. For Euler etc, the transformation is nonlinear
    //so careful attention to what is evaluated where and interpolated to where is needed.
    //For further information, please see Chan, Jesse. "On discretely entropy conservative and entropy stable discontinuous Galerkin methods." Journal of Computational Physics 362 (2018): 346-374.
    //pages 355 (Eq. 57 with text around it) and  page 359 (Eq 86 and text below it).

    // First, transform the volume conservative solution at volume cubature nodes to entropy variables.
    std::array<std::vector<real>,nstate> entropy_var_vol;
    for(unsigned int iquad=0; iquad<n_quad_pts_vol; iquad++){
        std::array<real,nstate> soln_state;
        for(int istate=0; istate<nstate; istate++){
            soln_state[istate] = soln_at_vol_q[istate][iquad];
        }
        std::array<real,nstate> entropy_var;
        entropy_var = this->pde_physics_double->compute_entropy_variables(soln_state);
        for(int istate=0; istate<nstate; istate++){
            if(iquad==0){
                entropy_var_vol[istate].resize(n_quad_pts_vol);
            }
                entropy_var_vol[istate][iquad] = entropy_var[istate];
        }
    }

    //project it onto the solution basis functions and interpolate it
    std::array<std::vector<real>,nstate> projected_entropy_var_vol;
    std::array<std::vector<real>,nstate> projected_entropy_var_surf;
    for(int istate=0; istate<nstate; istate++){
        // allocate
        projected_entropy_var_vol[istate].resize(n_quad_pts_vol);
        projected_entropy_var_surf[istate].resize(n_face_quad_pts);

        //interior
        std::vector<real> entropy_var_coeff(n_shape_fns);
        soln_basis_projection_oper.matrix_vector_mult_1D(entropy_var_vol[istate],
                                                         entropy_var_coeff,
                                                         soln_basis_projection_oper.oneD_vol_operator);
        // Project ROM HERE
        if(this->all_parameters->reduced_order_param.entropy_variables_in_snapshots) {
            for(unsigned int i_shape_fns = 0; i_shape_fns<n_shape_fns; i_shape_fns++){
                entropy_var_coeff[i_shape_fns] = this->projected_entropy[dof_indices[istate*n_shape_fns+i_shape_fns]];
            }
        }
        soln_basis.matrix_vector_mult_1D(entropy_var_coeff,
                                         projected_entropy_var_vol[istate],
                                         soln_basis.oneD_vol_operator);
        soln_basis.matrix_vector_mult_surface_1D(iface,
                                                 entropy_var_coeff,
                                                 projected_entropy_var_surf[istate],
                                                 soln_basis.oneD_surf_operator,
                                                 soln_basis.oneD_vol_operator);
    }

    //get the surface-volume sparsity pattern for a "sum-factorized" Hadamard product only computing terms needed for the operation.
    const unsigned int row_size = n_face_quad_pts * n_quad_pts_1D;
    const unsigned int col_size = n_face_quad_pts * n_quad_pts_1D;
    std::vector<unsigned int> Hadamard_rows_sparsity(row_size);
    std::vector<unsigned int> Hadamard_columns_sparsity(col_size);
    if(this->all_parameters->use_split_form || this->all_parameters->use_curvilinear_split_form){
        flux_basis.sum_factorized_Hadamard_surface_sparsity_pattern(n_face_quad_pts, n_quad_pts_1D, Hadamard_rows_sparsity, Hadamard_columns_sparsity, dim_not_zero);
    }

    std::array<std::vector<real>,nstate> surf_vol_ref_2pt_flux_interp_surf;
    std::array<std::vector<real>,nstate> surf_vol_ref_2pt_flux_interp_vol;
    if(this->all_parameters->use_split_form || this->all_parameters->use_curvilinear_split_form){
        //get surface-volume hybrid 2pt flux from Eq.(15) in Chan, Jesse. "Skew-symmetric entropy stable modal discontinuous Galerkin formulations." Journal of Scientific Computing 81.1 (2019): 459-485.
        std::array<dealii::FullMatrix<real>,nstate> surface_ref_2pt_flux;
        //make use of the sparsity pattern from above to assemble only n^d non-zero entries without ever allocating not computing zeros.
        for(int istate=0; istate<nstate; istate++){
            surface_ref_2pt_flux[istate].reinit(n_face_quad_pts, n_quad_pts_1D);
        }
        for(unsigned int iquad_face=0; iquad_face<n_face_quad_pts; iquad_face++){
            dealii::Tensor<2,dim,real> metric_cofactor_surf;
            for(int idim=0; idim<dim; idim++){
                for(int jdim=0; jdim<dim; jdim++){
                    metric_cofactor_surf[idim][jdim] = metric_oper.metric_cofactor_surf[idim][jdim][iquad_face];
                }
            }

            //Compute the conservative values on the facet from the interpolated entorpy variables.
            std::array<real,nstate> entropy_var_face;
            for(int istate=0; istate<nstate; istate++){
                entropy_var_face[istate] = projected_entropy_var_surf[istate][iquad_face];
            }
            std::array<real,nstate> soln_state_face;
            soln_state_face = this->pde_physics_double->compute_conservative_variables_from_entropy_variables (entropy_var_face);

            //only do the n_quad_1D vol points that give non-zero entries from Hadamard product.
            for(unsigned int row_index = iquad_face * n_quad_pts_1D, column_index = 0;
                column_index < n_quad_pts_1D;
                row_index++, column_index++){

                if(Hadamard_rows_sparsity[row_index] != iquad_face){
                    pcout<<"The boundary Hadamard rows sparsity pattern does not match."<<std::endl;
                    std::abort();
                }

                const unsigned int iquad_vol = Hadamard_columns_sparsity[row_index];//extract flux_quad pt that corresponds to a non-zero entry for Hadamard product.
                // Copy Metric Cofactor in a way can use for transforming Tensor Blocks to reference space
                // The way it is stored in metric_operators is to use sum-factorization in each direction,
                // but here it is cleaner to apply a reference transformation in each Tensor block returned by physics.
                dealii::Tensor<2,dim,real> metric_cofactor_vol;
                for(int idim=0; idim<dim; idim++){
                    for(int jdim=0; jdim<dim; jdim++){
                        metric_cofactor_vol[idim][jdim] = metric_oper.metric_cofactor_vol[idim][jdim][iquad_vol];
                    }
                }
                std::array<real,nstate> entropy_var;
                for(int istate=0; istate<nstate; istate++){
                    entropy_var[istate] = projected_entropy_var_vol[istate][iquad_vol];
                }
                std::array<real,nstate> soln_state;
                soln_state = this->pde_physics_double->compute_conservative_variables_from_entropy_variables (entropy_var);
                //Note that the flux basis is collocated on the volume cubature set so we don't need to evaluate the entropy variables
                //on the volume set then transform back to the conservative variables since the flux basis volume
                //projection is identity.

                //Compute the physical flux
                std::array<dealii::Tensor<1,dim,real>,nstate> conv_phys_flux_2pt;
                conv_phys_flux_2pt = this->pde_physics_double->convective_numerical_split_flux(soln_state, soln_state_face);
                for(int istate=0; istate<nstate; istate++){
                    dealii::Tensor<1,dim,real> conv_ref_flux_2pt;
                    //For each state, transform the physical flux to a reference flux.
                    metric_oper.transform_physical_to_reference(
                        conv_phys_flux_2pt[istate],
                        0.5*(metric_cofactor_surf + metric_cofactor_vol),
                        conv_ref_flux_2pt);
                    //only store the dim not zero in reference space bc dot product with unit ref normal later.
                    surface_ref_2pt_flux[istate][iquad_face][column_index] = conv_ref_flux_2pt[dim_not_zero];
                }
            }
        }
        //get the surface basis operator from Hadamard sparsity pattern
        //to be applied at n^d operations (on the face so n^{d+1-1}=n^d flops)
        //also only allocates n^d terms.
        const int iface_1D = iface % 2;//the reference face number
        const std::vector<double> &oneD_quad_weights_vol= this->oneD_quadrature_collection[poly_degree].get_weights();
        dealii::FullMatrix<real> surf_oper_sparse(n_face_quad_pts, n_quad_pts_1D);
        flux_basis.sum_factorized_Hadamard_surface_basis_assembly(n_face_quad_pts, n_quad_pts_1D,
                                                                  Hadamard_rows_sparsity, Hadamard_columns_sparsity,
                                                                  flux_basis.oneD_surf_operator[iface_1D],
                                                                  oneD_quad_weights_vol,
                                                                  surf_oper_sparse,
                                                                  dim_not_zero);

        // Apply the surface Hadamard products and multiply with vector of ones for both off diagonal terms in
        // Eq.(15) in Chan, Jesse. "Skew-symmetric entropy stable modal discontinuous Galerkin formulations." Journal of Scientific Computing 81.1 (2019): 459-485.
        for(int istate=0; istate<nstate; istate++){
            //first apply Hadamard product with the structure made above.
            dealii::FullMatrix<real> surface_ref_2pt_flux_int_Hadamard_with_surf_oper(n_face_quad_pts, n_quad_pts_1D);
            flux_basis.Hadamard_product(surf_oper_sparse,
                                        surface_ref_2pt_flux[istate],
                                        surface_ref_2pt_flux_int_Hadamard_with_surf_oper);
            //sum with reference unit normal
            surf_vol_ref_2pt_flux_interp_surf[istate].resize(n_face_quad_pts);
            surf_vol_ref_2pt_flux_interp_vol[istate].resize(n_quad_pts_vol);

            for(unsigned int iface_quad=0; iface_quad<n_face_quad_pts; iface_quad++){
                for(unsigned int iquad_int=0; iquad_int<n_quad_pts_1D; iquad_int++){
                    surf_vol_ref_2pt_flux_interp_surf[istate][iface_quad]
                        -= surface_ref_2pt_flux_int_Hadamard_with_surf_oper[iface_quad][iquad_int]
                        * unit_ref_normal_int[dim_not_zero];
                    const unsigned int column_index = iface_quad * n_quad_pts_1D + iquad_int;
                    surf_vol_ref_2pt_flux_interp_vol[istate][Hadamard_columns_sparsity[column_index]]
                        += surface_ref_2pt_flux_int_Hadamard_with_surf_oper[iface_quad][iquad_int]
                        * unit_ref_normal_int[dim_not_zero];
                }
            }
        }
    }//end of if split form or curvilinear split form


    //the outward reference normal dircetion.
    std::array<std::vector<real>,nstate> conv_flux_dot_normal;
    std::array<std::vector<real>,nstate> diss_flux_dot_normal_diff;
    // Get surface numerical fluxes
    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
        // Copy Metric Cofactor on the facet in a way can use for transforming Tensor Blocks to reference space
        // The way it is stored in metric_operators is to use sum-factorization in each direction,
        // but here it is cleaner to apply a reference transformation in each Tensor block returned by physics.
        // Note that for a conforming mesh, the facet metric cofactor matrix is the same from either interioir or exterior metric terms.
        // This is verified for the metric computations in: unit_tests/operator_tests/surface_conforming_test.cpp
        dealii::Tensor<2,dim,real> metric_cofactor_surf;
        for(int idim=0; idim<dim; idim++){
            for(int jdim=0; jdim<dim; jdim++){
                metric_cofactor_surf[idim][jdim] = metric_oper.metric_cofactor_surf[idim][jdim][iquad];
            }
        }
        //numerical fluxes
        dealii::Tensor<1,dim,real> unit_phys_normal_int;
        metric_oper.transform_reference_to_physical(unit_ref_normal_int,
                                                    metric_cofactor_surf,
                                                    unit_phys_normal_int);
        const double face_Jac_norm_scaled = unit_phys_normal_int.norm();
        unit_phys_normal_int /= face_Jac_norm_scaled;//normalize it.

        //get the projected entropy variables, soln, and
        //auxiliary solution on the surface point.
        std::array<real,nstate> entropy_var_face_int;
        std::array<dealii::Tensor<1,dim,real>,nstate> aux_soln_state_int;
        std::array<real,nstate> soln_interp_to_face_int;
        for(int istate=0; istate<nstate; istate++){
            soln_interp_to_face_int[istate] = soln_at_surf_q[istate][iquad];
            entropy_var_face_int[istate] = projected_entropy_var_surf[istate][iquad];
            for(int idim=0; idim<dim; idim++){
                aux_soln_state_int[istate][idim] = aux_soln_at_surf_q[istate][idim][iquad];
            }
        }

        //extract solution on surface from projected entropy variables
        std::array<real,nstate> soln_state_int;
        soln_state_int = this->pde_physics_double->compute_conservative_variables_from_entropy_variables (entropy_var_face_int);


        if(!this->all_parameters->use_split_form && !this->all_parameters->use_curvilinear_split_form){
            for(int istate=0; istate<nstate; istate++){
                soln_state_int[istate] = soln_at_surf_q[istate][iquad];
            }
        }

        std::array<real,nstate> soln_boundary;
        std::array<dealii::Tensor<1,dim,real>,nstate> grad_soln_boundary;
        dealii::Point<dim,real> surf_flux_node;
        for(int idim=0; idim<dim; idim++){
            surf_flux_node[idim] = metric_oper.flux_nodes_surf[iface][idim][iquad];
        }
        //I am not sure if BC should be from solution interpolated to face
        //or solution from the projected entropy variables.
        //Now, it uses projected entropy variables for NSFR, and solution
        //interpolated to face for conservative DG.
        this->pde_physics_double->boundary_face_values (boundary_id, surf_flux_node, unit_phys_normal_int, soln_state_int, aux_soln_state_int, soln_boundary, grad_soln_boundary);

        // Convective numerical flux.
        std::array<real,nstate> conv_num_flux_dot_n_at_q;
        conv_num_flux_dot_n_at_q = this->conv_num_flux_double->evaluate_flux(soln_state_int, soln_boundary, unit_phys_normal_int);

        // Dissipative numerical flux
        std::array<real,nstate> diss_auxi_num_flux_dot_n_at_q;
        diss_auxi_num_flux_dot_n_at_q = this->diss_num_flux_double->evaluate_auxiliary_flux(
            current_cell_index, current_cell_index,
            0.0, 0.0,
            soln_interp_to_face_int, soln_boundary,
            aux_soln_state_int, grad_soln_boundary,
            unit_phys_normal_int, penalty, true);

        for(int istate=0; istate<nstate; istate++){
            // allocate
            if(iquad==0){
                conv_flux_dot_normal[istate].resize(n_face_quad_pts);
                diss_flux_dot_normal_diff[istate].resize(n_face_quad_pts);
            }
            // write data
            conv_flux_dot_normal[istate][iquad] = face_Jac_norm_scaled * conv_num_flux_dot_n_at_q[istate];
            diss_flux_dot_normal_diff[istate][iquad] = face_Jac_norm_scaled * diss_auxi_num_flux_dot_n_at_q[istate]
                                                     - diffusive_int_vol_ref_flux_interp_to_face_dot_ref_normal[istate][iquad];
        }
    }

    //solve rhs
    for(int istate=0; istate<nstate; istate++){
        std::vector<real> rhs(n_shape_fns);
        //Convective flux on the facet
        if(this->all_parameters->use_split_form || this->all_parameters->use_curvilinear_split_form){
            std::vector<real> ones_surf(n_face_quad_pts, 1.0);
            soln_basis.inner_product_surface_1D(iface,
                                                surf_vol_ref_2pt_flux_interp_surf[istate],
                                                ones_surf, rhs,
                                                soln_basis.oneD_surf_operator,
                                                soln_basis.oneD_vol_operator,
                                                false, -1.0);
            std::vector<real> ones_vol(n_quad_pts_vol, 1.0);
            soln_basis.inner_product_1D(surf_vol_ref_2pt_flux_interp_vol[istate],
                                            ones_vol, rhs,
                                            soln_basis.oneD_vol_operator,
                                            true, -1.0);
        }
        else{
            soln_basis.inner_product_surface_1D(iface, conv_int_vol_ref_flux_interp_to_face_dot_ref_normal[istate],
                                                face_quad_weights, rhs,
                                                soln_basis.oneD_surf_operator,
                                                soln_basis.oneD_vol_operator,
                                                false, 1.0);//adding=false, scaled by factor=-1.0 bc subtract it
        }
        //Convective surface nnumerical flux.
        soln_basis.inner_product_surface_1D(iface, conv_flux_dot_normal[istate],
                                            face_quad_weights, rhs,
                                            soln_basis.oneD_surf_operator,
                                            soln_basis.oneD_vol_operator,
                                            true, -1.0);//adding=true, scaled by factor=-1.0 bc subtract it
        //Dissipative surface numerical flux.
        soln_basis.inner_product_surface_1D(iface, diss_flux_dot_normal_diff[istate],
                                            face_quad_weights, rhs,
                                            soln_basis.oneD_surf_operator,
                                            soln_basis.oneD_vol_operator,
                                            true, -1.0);//adding=true, scaled by factor=-1.0 bc subtract it

        for(unsigned int ishape=0; ishape<n_shape_fns; ishape++){
            local_rhs_cell(istate*n_shape_fns + ishape) += rhs[ishape];
        }
    }
}

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
{
    (void) current_cell_index;
    (void) neighbor_cell_index;

    const unsigned int n_face_quad_pts = this->face_quadrature_collection[poly_degree_int].size();//assume interior cell does the work

    const unsigned int n_quad_pts_vol_int  = this->volume_quadrature_collection[poly_degree_int].size();
    const unsigned int n_quad_pts_vol_ext  = this->volume_quadrature_collection[poly_degree_ext].size();
    const unsigned int n_quad_pts_1D_int  = this->oneD_quadrature_collection[poly_degree_int].size();
    const unsigned int n_quad_pts_1D_ext  = this->oneD_quadrature_collection[poly_degree_ext].size();

    const unsigned int n_dofs_int = this->fe_collection[poly_degree_int].dofs_per_cell;
    const unsigned int n_dofs_ext = this->fe_collection[poly_degree_ext].dofs_per_cell;

    const unsigned int n_shape_fns_int = n_dofs_int / nstate;
    const unsigned int n_shape_fns_ext = n_dofs_ext / nstate;

    AssertDimension (n_dofs_int, dof_indices_int.size());
    AssertDimension (n_dofs_ext, dof_indices_ext.size());

    // Extract interior modal coefficients of solution
    std::array<std::vector<real>,nstate> soln_coeff_int;
    std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> aux_soln_coeff_int;
    for (unsigned int idof = 0; idof < n_dofs_int; ++idof) {
        const unsigned int istate = this->fe_collection[poly_degree_int].system_to_component_index(idof).first;
        const unsigned int ishape = this->fe_collection[poly_degree_int].system_to_component_index(idof).second;
        if(ishape == 0)
            soln_coeff_int[istate].resize(n_shape_fns_int);

        soln_coeff_int[istate][ishape] = DGBase<dim,real,MeshType>::solution(dof_indices_int[idof]);
        for(int idim=0; idim<dim; idim++){
            if(ishape == 0){
                aux_soln_coeff_int[istate][idim].resize(n_shape_fns_int);
            }
            if(this->use_auxiliary_eq){
                aux_soln_coeff_int[istate][idim][ishape] = DGBase<dim,real,MeshType>::auxiliary_solution[idim](dof_indices_int[idof]);
            }
            else{
                aux_soln_coeff_int[istate][idim][ishape] = 0.0;
            }
        }
    }

    // Extract exterior modal coefficients of solution
    std::array<std::vector<real>,nstate> soln_coeff_ext;
    std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> aux_soln_coeff_ext;
    for (unsigned int idof = 0; idof < n_dofs_ext; ++idof) {
        const unsigned int istate = this->fe_collection[poly_degree_ext].system_to_component_index(idof).first;
        const unsigned int ishape = this->fe_collection[poly_degree_ext].system_to_component_index(idof).second;
        if(ishape == 0){
            soln_coeff_ext[istate].resize(n_shape_fns_ext);
        }
        soln_coeff_ext[istate][ishape] = DGBase<dim,real,MeshType>::solution(dof_indices_ext[idof]);
        for(int idim=0; idim<dim; idim++){
            if(ishape == 0){
                aux_soln_coeff_ext[istate][idim].resize(n_shape_fns_ext);
            }
            if(this->use_auxiliary_eq){
                aux_soln_coeff_ext[istate][idim][ishape] = DGBase<dim,real,MeshType>::auxiliary_solution[idim](dof_indices_ext[idof]);
            }
            else{
                aux_soln_coeff_ext[istate][idim][ishape] = 0.0;
            }
        }
    }

    // Interpolate the modal coefficients to the volume cubature nodes.
    std::array<std::vector<real>,nstate> soln_at_vol_q_int;
    std::array<std::vector<real>,nstate> soln_at_vol_q_ext;
    std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> aux_soln_at_vol_q_int;
    std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> aux_soln_at_vol_q_ext;
    // Interpolate modal soln coefficients to the facet.
    std::array<std::vector<real>,nstate> soln_at_surf_q_int;
    std::array<std::vector<real>,nstate> soln_at_surf_q_ext;
    std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> aux_soln_at_surf_q_int;
    std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> aux_soln_at_surf_q_ext;
    for(int istate=0; istate<nstate; ++istate){
        // allocate
        soln_at_vol_q_int[istate].resize(n_quad_pts_vol_int);
        soln_at_vol_q_ext[istate].resize(n_quad_pts_vol_ext);
        // solve soln at volume cubature nodes
        soln_basis_int.matrix_vector_mult_1D(soln_coeff_int[istate], soln_at_vol_q_int[istate],
                                             soln_basis_int.oneD_vol_operator);
        soln_basis_ext.matrix_vector_mult_1D(soln_coeff_ext[istate], soln_at_vol_q_ext[istate],
                                             soln_basis_ext.oneD_vol_operator);

        // allocate
        soln_at_surf_q_int[istate].resize(n_face_quad_pts);
        soln_at_surf_q_ext[istate].resize(n_face_quad_pts);
        // solve soln at facet cubature nodes
        soln_basis_int.matrix_vector_mult_surface_1D(iface,
                                                     soln_coeff_int[istate], soln_at_surf_q_int[istate],
                                                     soln_basis_int.oneD_surf_operator,
                                                     soln_basis_int.oneD_vol_operator);
        soln_basis_ext.matrix_vector_mult_surface_1D(neighbor_iface,
                                                     soln_coeff_ext[istate], soln_at_surf_q_ext[istate],
                                                     soln_basis_ext.oneD_surf_operator,
                                                     soln_basis_ext.oneD_vol_operator);

        for(int idim=0; idim<dim; idim++){
            // alocate
            aux_soln_at_vol_q_int[istate][idim].resize(n_quad_pts_vol_int);
            aux_soln_at_vol_q_ext[istate][idim].resize(n_quad_pts_vol_ext);
            // solve auxiliary soln at volume cubature nodes
            soln_basis_int.matrix_vector_mult_1D(aux_soln_coeff_int[istate][idim], aux_soln_at_vol_q_int[istate][idim],
                                                 soln_basis_int.oneD_vol_operator);
            soln_basis_ext.matrix_vector_mult_1D(aux_soln_coeff_ext[istate][idim], aux_soln_at_vol_q_ext[istate][idim],
                                                 soln_basis_ext.oneD_vol_operator);

            // allocate
            aux_soln_at_surf_q_int[istate][idim].resize(n_face_quad_pts);
            aux_soln_at_surf_q_ext[istate][idim].resize(n_face_quad_pts);
            // solve auxiliary soln at facet cubature nodes
            soln_basis_int.matrix_vector_mult_surface_1D(iface,
                                                         aux_soln_coeff_int[istate][idim], aux_soln_at_surf_q_int[istate][idim],
                                                         soln_basis_int.oneD_surf_operator,
                                                         soln_basis_int.oneD_vol_operator);
            soln_basis_ext.matrix_vector_mult_surface_1D(neighbor_iface,
                                                         aux_soln_coeff_ext[istate][idim], aux_soln_at_surf_q_ext[istate][idim],
                                                         soln_basis_ext.oneD_surf_operator,
                                                         soln_basis_ext.oneD_vol_operator);
        }
    }




    // Get volume reference fluxes and interpolate them to the facet.
    // Compute reference volume fluxes in both interior and exterior cells.

    // First we do interior.
    std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> conv_ref_flux_at_vol_q_int;
    std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> diffusive_ref_flux_at_vol_q_int;
    for (unsigned int iquad=0; iquad<n_quad_pts_vol_int; ++iquad) {
        // Copy Metric Cofactor in a way can use for transforming Tensor Blocks to reference space
        // The way it is stored in metric_operators is to use sum-factorization in each direction,
        // but here it is cleaner to apply a reference transformation in each Tensor block returned by physics.
        dealii::Tensor<2,dim,real> metric_cofactor_vol_int;
        for(int idim=0; idim<dim; idim++){
            for(int jdim=0; jdim<dim; jdim++){
                metric_cofactor_vol_int[idim][jdim] = metric_oper_int.metric_cofactor_vol[idim][jdim][iquad];
            }
        }
        std::array<real,nstate> soln_state;
        std::array<dealii::Tensor<1,dim,real>,nstate> aux_soln_state;
        for(int istate=0; istate<nstate; istate++){
            soln_state[istate] = soln_at_vol_q_int[istate][iquad];
            for(int idim=0; idim<dim; idim++){
                aux_soln_state[istate][idim] = aux_soln_at_vol_q_int[istate][idim][iquad];
            }
        }

        // Evaluate physical convective flux
        std::array<dealii::Tensor<1,dim,real>,nstate> conv_phys_flux;
        //Only for conservtive DG do we interpolate volume fluxes to the facet
        if(!this->all_parameters->use_split_form && !this->all_parameters->use_curvilinear_split_form){
            conv_phys_flux = this->pde_physics_double->convective_flux (soln_state);
        }

        // Compute the physical dissipative flux
        std::array<dealii::Tensor<1,dim,real>,nstate> diffusive_phys_flux;
        diffusive_phys_flux = this->pde_physics_double->dissipative_flux(soln_state, aux_soln_state, current_cell_index);

        // Write the values in a way that we can use sum-factorization on.
        for(int istate=0; istate<nstate; istate++){
            dealii::Tensor<1,dim,real> conv_ref_flux;
            dealii::Tensor<1,dim,real> diffusive_ref_flux;
            // transform the conservative convective physical flux to reference space
            if(!this->all_parameters->use_split_form && !this->all_parameters->use_curvilinear_split_form){
                metric_oper_int.transform_physical_to_reference(
                    conv_phys_flux[istate],
                    metric_cofactor_vol_int,
                    conv_ref_flux);
            }
            // transform the dissipative flux to reference space
            metric_oper_int.transform_physical_to_reference(
                diffusive_phys_flux[istate],
                metric_cofactor_vol_int,
                diffusive_ref_flux);

            // Write the data in a way that we can use sum-factorization on.
            // Since sum-factorization improves the speed for matrix-vector multiplications,
            // We need the values to have their inner elements be vectors.
            for(int idim=0; idim<dim; idim++){
                // allocate
                if(iquad == 0){
                    conv_ref_flux_at_vol_q_int[istate][idim].resize(n_quad_pts_vol_int);
                    diffusive_ref_flux_at_vol_q_int[istate][idim].resize(n_quad_pts_vol_int);
                }
                // write data
                if(!this->all_parameters->use_split_form && !this->all_parameters->use_curvilinear_split_form){
                    conv_ref_flux_at_vol_q_int[istate][idim][iquad] = conv_ref_flux[idim];
                }
                diffusive_ref_flux_at_vol_q_int[istate][idim][iquad] = diffusive_ref_flux[idim];
            }
        }
    }

    // Next we do exterior volume reference fluxes.
    // Note we split the quad integrals because the interior and exterior could be of different poly basis
    std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> conv_ref_flux_at_vol_q_ext;
    std::array<dealii::Tensor<1,dim,std::vector<real>>,nstate> diffusive_ref_flux_at_vol_q_ext;
    for (unsigned int iquad=0; iquad<n_quad_pts_vol_ext; ++iquad) {

        // Extract exterior volume metric cofactor matrix at given volume cubature node.
        dealii::Tensor<2,dim,real> metric_cofactor_vol_ext;
        for(int idim=0; idim<dim; idim++){
            for(int jdim=0; jdim<dim; jdim++){
                metric_cofactor_vol_ext[idim][jdim] = metric_oper_ext.metric_cofactor_vol[idim][jdim][iquad];
            }
        }

        std::array<real,nstate> soln_state;
        std::array<dealii::Tensor<1,dim,real>,nstate> aux_soln_state;
        for(int istate=0; istate<nstate; istate++){
            soln_state[istate] = soln_at_vol_q_ext[istate][iquad];
            for(int idim=0; idim<dim; idim++){
                aux_soln_state[istate][idim] = aux_soln_at_vol_q_ext[istate][idim][iquad];
            }
        }

        // Evaluate physical convective flux
        std::array<dealii::Tensor<1,dim,real>,nstate> conv_phys_flux;
        if(!this->all_parameters->use_split_form && !this->all_parameters->use_curvilinear_split_form){
            conv_phys_flux = this->pde_physics_double->convective_flux (soln_state);
        }

        // Compute the physical dissipative flux
        std::array<dealii::Tensor<1,dim,real>,nstate> diffusive_phys_flux;
        diffusive_phys_flux = this->pde_physics_double->dissipative_flux(soln_state, aux_soln_state, neighbor_cell_index);

        // Write the values in a way that we can use sum-factorization on.
        for(int istate=0; istate<nstate; istate++){
            dealii::Tensor<1,dim,real> conv_ref_flux;
            dealii::Tensor<1,dim,real> diffusive_ref_flux;
            // transform the conservative convective physical flux to reference space
            if(!this->all_parameters->use_split_form && !this->all_parameters->use_curvilinear_split_form){
                metric_oper_ext.transform_physical_to_reference(
                    conv_phys_flux[istate],
                    metric_cofactor_vol_ext,
                    conv_ref_flux);
            }
            // transform the dissipative flux to reference space
            metric_oper_ext.transform_physical_to_reference(
                diffusive_phys_flux[istate],
                metric_cofactor_vol_ext,
                diffusive_ref_flux);

            // Write the data in a way that we can use sum-factorization on.
            // Since sum-factorization improves the speed for matrix-vector multiplications,
            // We need the values to have their inner elements be vectors.
            for(int idim=0; idim<dim; idim++){
                // allocate
                if(iquad == 0){
                    conv_ref_flux_at_vol_q_ext[istate][idim].resize(n_quad_pts_vol_ext);
                    diffusive_ref_flux_at_vol_q_ext[istate][idim].resize(n_quad_pts_vol_ext);
                }
                // write data
                if(!this->all_parameters->use_split_form && !this->all_parameters->use_curvilinear_split_form){
                    conv_ref_flux_at_vol_q_ext[istate][idim][iquad] = conv_ref_flux[idim];
                }
                diffusive_ref_flux_at_vol_q_ext[istate][idim][iquad] = diffusive_ref_flux[idim];
            }
        }
    }

    // Interpolate the volume reference fluxes to the facet.
    // And do the dot product with the UNIT REFERENCE normal.
    // Since we are computing a dot product with the unit reference normal,
    // we exploit the fact that the unit reference normal has a value of 0 in all reference directions except
    // the outward reference normal dircetion.
    const dealii::Tensor<1,dim,double> unit_ref_normal_int = dealii::GeometryInfo<dim>::unit_normal_vector[iface];
    const dealii::Tensor<1,dim,double> unit_ref_normal_ext = dealii::GeometryInfo<dim>::unit_normal_vector[neighbor_iface];
    // Extract the reference direction that is outward facing on the facet.
    const int dim_not_zero_int = iface / 2;//reference direction of face integer division
    const int dim_not_zero_ext = neighbor_iface / 2;//reference direction of face integer division

    std::array<std::vector<real>,nstate> conv_int_vol_ref_flux_interp_to_face_dot_ref_normal;
    std::array<std::vector<real>,nstate> conv_ext_vol_ref_flux_interp_to_face_dot_ref_normal;
    std::array<std::vector<real>,nstate> diffusive_int_vol_ref_flux_interp_to_face_dot_ref_normal;
    std::array<std::vector<real>,nstate> diffusive_ext_vol_ref_flux_interp_to_face_dot_ref_normal;
    for(int istate=0; istate<nstate; istate++){
        //allocate
        conv_int_vol_ref_flux_interp_to_face_dot_ref_normal[istate].resize(n_face_quad_pts);
        conv_ext_vol_ref_flux_interp_to_face_dot_ref_normal[istate].resize(n_face_quad_pts);
        diffusive_int_vol_ref_flux_interp_to_face_dot_ref_normal[istate].resize(n_face_quad_pts);
        diffusive_ext_vol_ref_flux_interp_to_face_dot_ref_normal[istate].resize(n_face_quad_pts);

        // solve
        // Note, since the normal is zero in all other reference directions, we only have to interpolate one given reference direction to the facet

        // interpolate reference volume convective flux to the facet, and apply unit reference normal as scaled by 1.0 or -1.0
        if(!this->all_parameters->use_split_form && !this->all_parameters->use_curvilinear_split_form){
            flux_basis_int.matrix_vector_mult_surface_1D(iface,
                                                         conv_ref_flux_at_vol_q_int[istate][dim_not_zero_int],
                                                         conv_int_vol_ref_flux_interp_to_face_dot_ref_normal[istate],
                                                         flux_basis_int.oneD_surf_operator,//the flux basis interpolates from the flux nodes
                                                         flux_basis_int.oneD_vol_operator,
                                                         false, unit_ref_normal_int[dim_not_zero_int]);//don't add to previous value, scale by unit_normal int
            flux_basis_ext.matrix_vector_mult_surface_1D(neighbor_iface,
                                                         conv_ref_flux_at_vol_q_ext[istate][dim_not_zero_ext],
                                                         conv_ext_vol_ref_flux_interp_to_face_dot_ref_normal[istate],
                                                         flux_basis_ext.oneD_surf_operator,
                                                         flux_basis_ext.oneD_vol_operator,
                                                         false, unit_ref_normal_ext[dim_not_zero_ext]);//don't add to previous value, unit_normal ext is -unit normal int
        }

        // interpolate reference volume dissipative flux to the facet, and apply unit reference normal as scaled by 1.0 or -1.0
        flux_basis_int.matrix_vector_mult_surface_1D(iface,
                                                     diffusive_ref_flux_at_vol_q_int[istate][dim_not_zero_int],
                                                     diffusive_int_vol_ref_flux_interp_to_face_dot_ref_normal[istate],
                                                     flux_basis_int.oneD_surf_operator,
                                                     flux_basis_int.oneD_vol_operator,
                                                     false, unit_ref_normal_int[dim_not_zero_int]);
        flux_basis_ext.matrix_vector_mult_surface_1D(neighbor_iface,
                                                     diffusive_ref_flux_at_vol_q_ext[istate][dim_not_zero_ext],
                                                     diffusive_ext_vol_ref_flux_interp_to_face_dot_ref_normal[istate],
                                                     flux_basis_ext.oneD_surf_operator,
                                                     flux_basis_ext.oneD_vol_operator,
                                                     false, unit_ref_normal_ext[dim_not_zero_ext]);
    }


    //Note that for entropy-dissipation and entropy stability, the conservative variables
    //are functions of projected entropy variables. For Euler etc, the transformation is nonlinear
    //so careful attention to what is evaluated where and interpolated to where is needed.
    //For further information, please see Chan, Jesse. "On discretely entropy conservative and entropy stable discontinuous Galerkin methods." Journal of Computational Physics 362 (2018): 346-374.
    //pages 355 (Eq. 57 with text around it) and  page 359 (Eq 86 and text below it).

    // First, transform the volume conservative solution at volume cubature nodes to entropy variables.
    std::array<std::vector<real>,nstate> entropy_var_vol_int;
    for(unsigned int iquad=0; iquad<n_quad_pts_vol_int; iquad++){
        std::array<real,nstate> soln_state;
        for(int istate=0; istate<nstate; istate++){
            soln_state[istate] = soln_at_vol_q_int[istate][iquad];
        }
        std::array<real,nstate> entropy_var;
        entropy_var = this->pde_physics_double->compute_entropy_variables(soln_state);
        for(int istate=0; istate<nstate; istate++){
            if(iquad==0){
                entropy_var_vol_int[istate].resize(n_quad_pts_vol_int);
            }
            entropy_var_vol_int[istate][iquad] = entropy_var[istate];
        }
    }
    std::array<std::vector<real>,nstate> entropy_var_vol_ext;
    for(unsigned int iquad=0; iquad<n_quad_pts_vol_ext; iquad++){
        std::array<real,nstate> soln_state;
        for(int istate=0; istate<nstate; istate++){
            soln_state[istate] = soln_at_vol_q_ext[istate][iquad];
        }
        std::array<real,nstate> entropy_var;
        entropy_var = this->pde_physics_double->compute_entropy_variables(soln_state);
        for(int istate=0; istate<nstate; istate++){
            if(iquad==0){
                entropy_var_vol_ext[istate].resize(n_quad_pts_vol_ext);
            }
            entropy_var_vol_ext[istate][iquad] = entropy_var[istate];
        }
    }

    //project it onto the solution basis functions and interpolate it
    std::array<std::vector<real>,nstate> projected_entropy_var_vol_int;
    std::array<std::vector<real>,nstate> projected_entropy_var_vol_ext;
    std::array<std::vector<real>,nstate> projected_entropy_var_surf_int;
    std::array<std::vector<real>,nstate> projected_entropy_var_surf_ext;
    for(int istate=0; istate<nstate; istate++){
        // allocate
        projected_entropy_var_vol_int[istate].resize(n_quad_pts_vol_int);
        projected_entropy_var_vol_ext[istate].resize(n_quad_pts_vol_ext);
        projected_entropy_var_surf_int[istate].resize(n_face_quad_pts);
        projected_entropy_var_surf_ext[istate].resize(n_face_quad_pts);

        //interior
        std::vector<real> entropy_var_coeff_int(n_shape_fns_int);
        soln_basis_projection_oper_int.matrix_vector_mult_1D(entropy_var_vol_int[istate],
                                                             entropy_var_coeff_int,
                                                             soln_basis_projection_oper_int.oneD_vol_operator);
        // ROM Projection here
        if(this->all_parameters->reduced_order_param.entropy_variables_in_snapshots) {
            for(unsigned int i_shape_fns = 0; i_shape_fns<n_shape_fns_int; i_shape_fns++){
                entropy_var_coeff_int[i_shape_fns] = this->projected_entropy[dof_indices_int[istate*n_shape_fns_int+i_shape_fns]];
            }
        }
        soln_basis_int.matrix_vector_mult_1D(entropy_var_coeff_int,
                                             projected_entropy_var_vol_int[istate],
                                             soln_basis_int.oneD_vol_operator);
        soln_basis_int.matrix_vector_mult_surface_1D(iface,
                                                     entropy_var_coeff_int,
                                                     projected_entropy_var_surf_int[istate],
                                                     soln_basis_int.oneD_surf_operator,
                                                     soln_basis_int.oneD_vol_operator);

        //exterior
        std::vector<real> entropy_var_coeff_ext(n_shape_fns_ext);
        soln_basis_projection_oper_ext.matrix_vector_mult_1D(entropy_var_vol_ext[istate],
                                                             entropy_var_coeff_ext,
                                                             soln_basis_projection_oper_ext.oneD_vol_operator);
        // ROM Projection Here
        if(this->all_parameters->reduced_order_param.entropy_variables_in_snapshots) {
            for(unsigned int i_shape_fns = 0; i_shape_fns<n_shape_fns_ext; i_shape_fns++){
                entropy_var_coeff_ext[i_shape_fns] = this->projected_entropy[dof_indices_ext[istate*n_shape_fns_ext+i_shape_fns]];
            }
        }
        soln_basis_ext.matrix_vector_mult_1D(entropy_var_coeff_ext,
                                             projected_entropy_var_vol_ext[istate],
                                             soln_basis_ext.oneD_vol_operator);
        soln_basis_ext.matrix_vector_mult_surface_1D(neighbor_iface,
                                                     entropy_var_coeff_ext,
                                                     projected_entropy_var_surf_ext[istate],
                                                     soln_basis_ext.oneD_surf_operator,
                                                     soln_basis_ext.oneD_vol_operator);
    }

    //get the surface-volume sparsity pattern for a "sum-factorized" Hadamard product only computing terms needed for the operation.
    const unsigned int row_size_int = n_face_quad_pts * n_quad_pts_1D_int;
    const unsigned int col_size_int = n_face_quad_pts * n_quad_pts_1D_int;
    std::vector<unsigned int> Hadamard_rows_sparsity_int(row_size_int);
    std::vector<unsigned int> Hadamard_columns_sparsity_int(col_size_int);
    const unsigned int row_size_ext = n_face_quad_pts * n_quad_pts_1D_ext;
    const unsigned int col_size_ext = n_face_quad_pts * n_quad_pts_1D_ext;
    std::vector<unsigned int> Hadamard_rows_sparsity_ext(row_size_ext);
    std::vector<unsigned int> Hadamard_columns_sparsity_ext(col_size_ext);
    if(this->all_parameters->use_split_form || this->all_parameters->use_curvilinear_split_form){
        flux_basis_int.sum_factorized_Hadamard_surface_sparsity_pattern(n_face_quad_pts, n_quad_pts_1D_int, Hadamard_rows_sparsity_int, Hadamard_columns_sparsity_int, dim_not_zero_int);
        flux_basis_ext.sum_factorized_Hadamard_surface_sparsity_pattern(n_face_quad_pts, n_quad_pts_1D_ext, Hadamard_rows_sparsity_ext, Hadamard_columns_sparsity_ext, dim_not_zero_ext);
    }

    std::array<std::vector<real>,nstate> surf_vol_ref_2pt_flux_interp_surf_int;
    std::array<std::vector<real>,nstate> surf_vol_ref_2pt_flux_interp_surf_ext;
    std::array<std::vector<real>,nstate> surf_vol_ref_2pt_flux_interp_vol_int;
    std::array<std::vector<real>,nstate> surf_vol_ref_2pt_flux_interp_vol_ext;
    if(this->all_parameters->use_split_form || this->all_parameters->use_curvilinear_split_form){
        //get surface-volume hybrid 2pt flux from Eq.(15) in Chan, Jesse. "Skew-symmetric entropy stable modal discontinuous Galerkin formulations." Journal of Scientific Computing 81.1 (2019): 459-485.
        std::array<dealii::FullMatrix<real>,nstate> surface_ref_2pt_flux_int;
        std::array<dealii::FullMatrix<real>,nstate> surface_ref_2pt_flux_ext;
        //make use of the sparsity pattern from above to assemble only n^d non-zero entries without ever allocating not computing zeros.
        for(int istate=0; istate<nstate; istate++){
            surface_ref_2pt_flux_int[istate].reinit(n_face_quad_pts, n_quad_pts_1D_int);
            surface_ref_2pt_flux_ext[istate].reinit(n_face_quad_pts, n_quad_pts_1D_ext);
        }
        for(unsigned int iquad_face=0; iquad_face<n_face_quad_pts; iquad_face++){
            dealii::Tensor<2,dim,real> metric_cofactor_surf;
            for(int idim=0; idim<dim; idim++){
                for(int jdim=0; jdim<dim; jdim++){
                    metric_cofactor_surf[idim][jdim] = metric_oper_int.metric_cofactor_surf[idim][jdim][iquad_face];
                }
            }

            //Compute the conservative values on the facet from the interpolated entorpy variables.
            std::array<real,nstate> entropy_var_face_int;
            std::array<real,nstate> entropy_var_face_ext;
            for(int istate=0; istate<nstate; istate++){
                entropy_var_face_int[istate] = projected_entropy_var_surf_int[istate][iquad_face];
                entropy_var_face_ext[istate] = projected_entropy_var_surf_ext[istate][iquad_face];
            }
            std::array<real,nstate> soln_state_face_int;
            std::array<real,nstate> soln_state_face_ext;
            soln_state_face_int = this->pde_physics_double->compute_conservative_variables_from_entropy_variables (entropy_var_face_int);
            soln_state_face_ext = this->pde_physics_double->compute_conservative_variables_from_entropy_variables (entropy_var_face_ext);

            //only do the n_quad_1D vol points that give non-zero entries from Hadamard product.
            for(unsigned int row_index = iquad_face * n_quad_pts_1D_int, column_index = 0;
                column_index < n_quad_pts_1D_int;
                row_index++, column_index++){

                if(Hadamard_rows_sparsity_int[row_index] != iquad_face){
                    pcout<<"The interior Hadamard rows sparsity pattern does not match."<<std::endl;
                    std::abort();
                }

                const unsigned int iquad_vol = Hadamard_columns_sparsity_int[row_index];//extract flux_quad pt that corresponds to a non-zero entry for Hadamard product.
                // Copy Metric Cofactor in a way can use for transforming Tensor Blocks to reference space
                // The way it is stored in metric_operators is to use sum-factorization in each direction,
                // but here it is cleaner to apply a reference transformation in each Tensor block returned by physics.
                dealii::Tensor<2,dim,real> metric_cofactor_vol_int;
                for(int idim=0; idim<dim; idim++){
                    for(int jdim=0; jdim<dim; jdim++){
                        metric_cofactor_vol_int[idim][jdim] = metric_oper_int.metric_cofactor_vol[idim][jdim][iquad_vol];
                    }
                }
                std::array<real,nstate> entropy_var;
                for(int istate=0; istate<nstate; istate++){
                    entropy_var[istate] = projected_entropy_var_vol_int[istate][iquad_vol];
                }
                std::array<real,nstate> soln_state;
                soln_state = this->pde_physics_double->compute_conservative_variables_from_entropy_variables (entropy_var);
                //Note that the flux basis is collocated on the volume cubature set so we don't need to evaluate the entropy variables
                //on the volume set then transform back to the conservative variables since the flux basis volume
                //projection is identity.

                //Compute the physical flux
                std::array<dealii::Tensor<1,dim,real>,nstate> conv_phys_flux_2pt;
                conv_phys_flux_2pt = this->pde_physics_double->convective_numerical_split_flux(soln_state, soln_state_face_int);
                for(int istate=0; istate<nstate; istate++){
                    dealii::Tensor<1,dim,real> conv_ref_flux_2pt;
                    //For each state, transform the physical flux to a reference flux.
                    metric_oper_int.transform_physical_to_reference(
                        conv_phys_flux_2pt[istate],
                        0.5*(metric_cofactor_surf + metric_cofactor_vol_int),
                        conv_ref_flux_2pt);
                    //only store the dim not zero in reference space bc dot product with unit ref normal later.
                    surface_ref_2pt_flux_int[istate][iquad_face][column_index] = conv_ref_flux_2pt[dim_not_zero_int];
                }
            }
            for(unsigned int row_index = iquad_face * n_quad_pts_1D_ext, column_index = 0;
                column_index < n_quad_pts_1D_ext;
                row_index++, column_index++){

                if(Hadamard_rows_sparsity_ext[row_index] != iquad_face){
                    pcout<<"The exterior Hadamard rows sparsity pattern does not match."<<std::endl;
                    std::abort();
                }

                const unsigned int iquad_vol = Hadamard_columns_sparsity_ext[row_index];//extract flux_quad pt that corresponds to a non-zero entry for Hadamard product.
                // Copy Metric Cofactor in a way can use for transforming Tensor Blocks to reference space
                // The way it is stored in metric_operators is to use sum-factorization in each direction,
                // but here it is cleaner to apply a reference transformation in each Tensor block returned by physics.
                dealii::Tensor<2,dim,real> metric_cofactor_vol_ext;
                for(int idim=0; idim<dim; idim++){
                    for(int jdim=0; jdim<dim; jdim++){
                        metric_cofactor_vol_ext[idim][jdim] = metric_oper_ext.metric_cofactor_vol[idim][jdim][iquad_vol];
                    }
                }
                std::array<real,nstate> entropy_var;
                for(int istate=0; istate<nstate; istate++){
                    entropy_var[istate] = projected_entropy_var_vol_ext[istate][iquad_vol];
                }
                std::array<real,nstate> soln_state;
                soln_state = this->pde_physics_double->compute_conservative_variables_from_entropy_variables (entropy_var);
                //Compute the physical flux
                std::array<dealii::Tensor<1,dim,real>,nstate> conv_phys_flux_2pt;
                conv_phys_flux_2pt = this->pde_physics_double->convective_numerical_split_flux(soln_state, soln_state_face_ext);
                for(int istate=0; istate<nstate; istate++){
                    dealii::Tensor<1,dim,real> conv_ref_flux_2pt;
                    //For each state, transform the physical flux to a reference flux.
                    metric_oper_ext.transform_physical_to_reference(
                        conv_phys_flux_2pt[istate],
                        0.5*(metric_cofactor_surf + metric_cofactor_vol_ext),
                        conv_ref_flux_2pt);
                    //only store the dim not zero in reference space bc dot product with unit ref normal later.
                    surface_ref_2pt_flux_ext[istate][iquad_face][column_index] = conv_ref_flux_2pt[dim_not_zero_ext];
                }
            }
        }

        //get the surface basis operator from Hadamard sparsity pattern
        //to be applied at n^d operations (on the face so n^{d+1-1}=n^d flops)
        //also only allocates n^d terms.
        const int iface_1D = iface % 2;//the reference face number
        const std::vector<double> &oneD_quad_weights_vol_int = this->oneD_quadrature_collection[poly_degree_int].get_weights();
        dealii::FullMatrix<real> surf_oper_sparse_int(n_face_quad_pts, n_quad_pts_1D_int);
        flux_basis_int.sum_factorized_Hadamard_surface_basis_assembly(n_face_quad_pts, n_quad_pts_1D_int,
                                                                      Hadamard_rows_sparsity_int, Hadamard_columns_sparsity_int,
                                                                      flux_basis_int.oneD_surf_operator[iface_1D],
                                                                      oneD_quad_weights_vol_int,
                                                                      surf_oper_sparse_int,
                                                                      dim_not_zero_int);
        const int neighbor_iface_1D = neighbor_iface % 2;//the reference neighbour face number
        const std::vector<double> &oneD_quad_weights_vol_ext = this->oneD_quadrature_collection[poly_degree_ext].get_weights();
        dealii::FullMatrix<real> surf_oper_sparse_ext(n_face_quad_pts, n_quad_pts_1D_ext);
        flux_basis_ext.sum_factorized_Hadamard_surface_basis_assembly(n_face_quad_pts, n_quad_pts_1D_ext,
                                                                      Hadamard_rows_sparsity_ext, Hadamard_columns_sparsity_ext,
                                                                      flux_basis_ext.oneD_surf_operator[neighbor_iface_1D],
                                                                      oneD_quad_weights_vol_ext,
                                                                      surf_oper_sparse_ext,
                                                                      dim_not_zero_ext);

        // Apply the surface Hadamard products and multiply with vector of ones for both off diagonal terms in
        // Eq.(15) in Chan, Jesse. "Skew-symmetric entropy stable modal discontinuous Galerkin formulations." Journal of Scientific Computing 81.1 (2019): 459-485.
        /*
        char zero = '0';
        std::cout << current_cell_index << std::endl;
        std::cout << iface << std::endl;
        surf_oper_sparse_int.print_formatted(std::cout,3,true,0,&zero);
        */
        for(int istate=0; istate<nstate; istate++){
            //first apply Hadamard product with the structure made above.
            dealii::FullMatrix<real> surface_ref_2pt_flux_int_Hadamard_with_surf_oper(n_face_quad_pts, n_quad_pts_1D_int);
            flux_basis_int.Hadamard_product(surf_oper_sparse_int,
                                            surface_ref_2pt_flux_int[istate],
                                            surface_ref_2pt_flux_int_Hadamard_with_surf_oper);
            dealii::FullMatrix<real> surface_ref_2pt_flux_ext_Hadamard_with_surf_oper(n_face_quad_pts, n_quad_pts_1D_ext);
            flux_basis_ext.Hadamard_product(surf_oper_sparse_ext,
                                            surface_ref_2pt_flux_ext[istate],
                                            surface_ref_2pt_flux_ext_Hadamard_with_surf_oper);
            //sum with reference unit normal
            surf_vol_ref_2pt_flux_interp_surf_int[istate].resize(n_face_quad_pts);
            surf_vol_ref_2pt_flux_interp_surf_ext[istate].resize(n_face_quad_pts);
            surf_vol_ref_2pt_flux_interp_vol_int[istate].resize(n_quad_pts_vol_int);
            surf_vol_ref_2pt_flux_interp_vol_ext[istate].resize(n_quad_pts_vol_ext);

            for(unsigned int iface_quad=0; iface_quad<n_face_quad_pts; iface_quad++){
                for(unsigned int iquad_int=0; iquad_int<n_quad_pts_1D_int; iquad_int++){
                    surf_vol_ref_2pt_flux_interp_surf_int[istate][iface_quad]
                        -= surface_ref_2pt_flux_int_Hadamard_with_surf_oper[iface_quad][iquad_int]
                        * unit_ref_normal_int[dim_not_zero_int];
                    const unsigned int column_index = iface_quad * n_quad_pts_1D_int + iquad_int;
                    surf_vol_ref_2pt_flux_interp_vol_int[istate][Hadamard_columns_sparsity_int[column_index]]
                        += surface_ref_2pt_flux_int_Hadamard_with_surf_oper[iface_quad][iquad_int]
                        * unit_ref_normal_int[dim_not_zero_int];
                }
                for(unsigned int iquad_ext=0; iquad_ext<n_quad_pts_1D_ext; iquad_ext++){
                    surf_vol_ref_2pt_flux_interp_surf_ext[istate][iface_quad]
                        -= surface_ref_2pt_flux_ext_Hadamard_with_surf_oper[iface_quad][iquad_ext]
                        * (unit_ref_normal_ext[dim_not_zero_ext]);
                    const unsigned int column_index = iface_quad * n_quad_pts_1D_ext + iquad_ext;
                    surf_vol_ref_2pt_flux_interp_vol_ext[istate][Hadamard_columns_sparsity_ext[column_index]]
                        += surface_ref_2pt_flux_ext_Hadamard_with_surf_oper[iface_quad][iquad_ext]
                        * (unit_ref_normal_ext[dim_not_zero_ext]);
                }
            }
        }
    }//end of if split form or curvilinear split form



    // Evaluate reference numerical fluxes.

    std::array<std::vector<real>,nstate> conv_num_flux_dot_n;
    std::array<std::vector<real>,nstate> diss_auxi_num_flux_dot_n;
    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
        // Copy Metric Cofactor on the facet in a way can use for transforming Tensor Blocks to reference space
        // The way it is stored in metric_operators is to use sum-factorization in each direction,
        // but here it is cleaner to apply a reference transformation in each Tensor block returned by physics.
        // Note that for a conforming mesh, the facet metric cofactor matrix is the same from either interioir or exterior metric terms.
        // This is verified for the metric computations in: unit_tests/operator_tests/surface_conforming_test.cpp
        dealii::Tensor<2,dim,real> metric_cofactor_surf;
        for(int idim=0; idim<dim; idim++){
            for(int jdim=0; jdim<dim; jdim++){
                metric_cofactor_surf[idim][jdim] = metric_oper_int.metric_cofactor_surf[idim][jdim][iquad];
            }
        }

        std::array<real,nstate> entropy_var_face_int;
        std::array<real,nstate> entropy_var_face_ext;
        std::array<dealii::Tensor<1,dim,real>,nstate> aux_soln_state_int;
        std::array<dealii::Tensor<1,dim,real>,nstate> aux_soln_state_ext;
        std::array<real,nstate> soln_interp_to_face_int;
        std::array<real,nstate> soln_interp_to_face_ext;
        for(int istate=0; istate<nstate; istate++){
            soln_interp_to_face_int[istate] = soln_at_surf_q_int[istate][iquad];
            soln_interp_to_face_ext[istate] = soln_at_surf_q_ext[istate][iquad];
            entropy_var_face_int[istate] = projected_entropy_var_surf_int[istate][iquad];
            entropy_var_face_ext[istate] = projected_entropy_var_surf_ext[istate][iquad];
            for(int idim=0; idim<dim; idim++){
                aux_soln_state_int[istate][idim] = aux_soln_at_surf_q_int[istate][idim][iquad];
                aux_soln_state_ext[istate][idim] = aux_soln_at_surf_q_ext[istate][idim][iquad];
            }
        }

        std::array<real,nstate> soln_state_int;
        soln_state_int = this->pde_physics_double->compute_conservative_variables_from_entropy_variables (entropy_var_face_int);
        std::array<real,nstate> soln_state_ext;
        soln_state_ext = this->pde_physics_double->compute_conservative_variables_from_entropy_variables (entropy_var_face_ext);


        if(!this->all_parameters->use_split_form && !this->all_parameters->use_curvilinear_split_form){
            for(int istate=0; istate<nstate; istate++){
                soln_state_int[istate] = soln_at_surf_q_int[istate][iquad];
                soln_state_ext[istate] = soln_at_surf_q_ext[istate][iquad];
            }
        }

        // numerical fluxes
        dealii::Tensor<1,dim,real> unit_phys_normal_int;
        metric_oper_int.transform_reference_to_physical(unit_ref_normal_int,
                                                        metric_cofactor_surf,
                                                        unit_phys_normal_int);
        const double face_Jac_norm_scaled = unit_phys_normal_int.norm();
        unit_phys_normal_int /= face_Jac_norm_scaled;//normalize it.
        // Note that the facet determinant of metric jacobian is the above norm multiplied by the determinant of the metric Jacobian evaluated on the facet.
        // Since the determinant of the metric Jacobian evaluated on the face cancels off, we can just scale the numerical flux by the norm.

        std::array<real,nstate> conv_num_flux_dot_n_at_q;
        std::array<real,nstate> diss_auxi_num_flux_dot_n_at_q;
        // Convective numerical flux.
        conv_num_flux_dot_n_at_q = this->conv_num_flux_double->evaluate_flux(soln_state_int, soln_state_ext, unit_phys_normal_int);
        // dissipative numerical flux
        diss_auxi_num_flux_dot_n_at_q = this->diss_num_flux_double->evaluate_auxiliary_flux(
            current_cell_index, neighbor_cell_index,
            0.0, 0.0,
            soln_interp_to_face_int, soln_interp_to_face_ext,
            aux_soln_state_int, aux_soln_state_ext,
            unit_phys_normal_int, penalty, false);

        // Write the values in a way that we can use sum-factorization on.
        for(int istate=0; istate<nstate; istate++){
            // Write the data in a way that we can use sum-factorization on.
            // Since sum-factorization improves the speed for matrix-vector multiplications,
            // We need the values to have their inner elements be vectors of n_face_quad_pts.

            // allocate
            if(iquad == 0){
                conv_num_flux_dot_n[istate].resize(n_face_quad_pts);
                diss_auxi_num_flux_dot_n[istate].resize(n_face_quad_pts);
            }

            // write data
            conv_num_flux_dot_n[istate][iquad] = face_Jac_norm_scaled * conv_num_flux_dot_n_at_q[istate];
            diss_auxi_num_flux_dot_n[istate][iquad] = face_Jac_norm_scaled * diss_auxi_num_flux_dot_n_at_q[istate];
        }
    }

    // Compute RHS
    const std::vector<double> &surf_quad_weights = this->face_quadrature_collection[poly_degree_int].get_weights();
    for(int istate=0; istate<nstate; istate++){
        // interior RHS
        std::vector<real> rhs_int(n_shape_fns_int);

        // convective flux
        if(this->all_parameters->use_split_form || this->all_parameters->use_curvilinear_split_form){
            std::vector<real> ones_surf(n_face_quad_pts, 1.0);
            soln_basis_int.inner_product_surface_1D(iface,
                                                    surf_vol_ref_2pt_flux_interp_surf_int[istate],
                                                    ones_surf, rhs_int,
                                                    soln_basis_int.oneD_surf_operator,
                                                    soln_basis_int.oneD_vol_operator,
                                                    false, -1.0);
            std::vector<real> ones_vol(n_quad_pts_vol_int, 1.0);
            soln_basis_int.inner_product_1D(surf_vol_ref_2pt_flux_interp_vol_int[istate],
                                            ones_vol, rhs_int,
                                            soln_basis_int.oneD_vol_operator,
                                            true, -1.0);
        }
        else
        {
            soln_basis_int.inner_product_surface_1D(iface,
                                                    conv_int_vol_ref_flux_interp_to_face_dot_ref_normal[istate],
                                                    surf_quad_weights, rhs_int,
                                                    soln_basis_int.oneD_surf_operator,
                                                    soln_basis_int.oneD_vol_operator,
                                                    false, 1.0);
        }
        // dissipative flux
        soln_basis_int.inner_product_surface_1D(iface,
                                                diffusive_int_vol_ref_flux_interp_to_face_dot_ref_normal[istate],
                                                surf_quad_weights, rhs_int,
                                                soln_basis_int.oneD_surf_operator,
                                                soln_basis_int.oneD_vol_operator,
                                                true, 1.0);//adding=true, subtract the negative so add it
        // convective numerical flux
        soln_basis_int.inner_product_surface_1D(iface, conv_num_flux_dot_n[istate],
                                                surf_quad_weights, rhs_int,
                                                soln_basis_int.oneD_surf_operator,
                                                soln_basis_int.oneD_vol_operator,
                                                true, -1.0);//adding=true, scaled by factor=-1.0 bc subtract it
        // dissipative numerical flux
        soln_basis_int.inner_product_surface_1D(iface, diss_auxi_num_flux_dot_n[istate],
                                                surf_quad_weights, rhs_int,
                                                soln_basis_int.oneD_surf_operator,
                                                soln_basis_int.oneD_vol_operator,
                                                true, -1.0);//adding=true, scaled by factor=-1.0 bc subtract it


        for(unsigned int ishape=0; ishape<n_shape_fns_int; ishape++){
            local_rhs_int_cell(istate*n_shape_fns_int + ishape) += rhs_int[ishape];
        }

        // exterior RHS
        std::vector<real> rhs_ext(n_shape_fns_ext);

        // convective flux
        if(this->all_parameters->use_split_form || this->all_parameters->use_curvilinear_split_form){
            std::vector<real> ones_surf(n_face_quad_pts, 1.0);
            soln_basis_ext.inner_product_surface_1D(neighbor_iface,
                                                    surf_vol_ref_2pt_flux_interp_surf_ext[istate],
                                                    ones_surf, rhs_ext,
                                                    soln_basis_ext.oneD_surf_operator,
                                                    soln_basis_ext.oneD_vol_operator,
                                                    false, -1.0);//the negative sign is bc the surface Hadamard function computes it on the otherside.
                                                    //to satisfy the unit test that checks consistency with Jesse Chan's formulation.
            std::vector<real> ones_vol(n_quad_pts_vol_ext, 1.0);
            soln_basis_ext.inner_product_1D(surf_vol_ref_2pt_flux_interp_vol_ext[istate],
                                            ones_vol, rhs_ext,
                                            soln_basis_ext.oneD_vol_operator,
                                            true, -1.0);
        }
        else
        {
            soln_basis_ext.inner_product_surface_1D(neighbor_iface,
                                                    conv_ext_vol_ref_flux_interp_to_face_dot_ref_normal[istate],
                                                    surf_quad_weights, rhs_ext,
                                                    soln_basis_ext.oneD_surf_operator,
                                                    soln_basis_ext.oneD_vol_operator,
                                                    false, 1.0);//adding false
        }
        // dissipative flux
        soln_basis_ext.inner_product_surface_1D(neighbor_iface,
                                                diffusive_ext_vol_ref_flux_interp_to_face_dot_ref_normal[istate],
                                                surf_quad_weights, rhs_ext,
                                                soln_basis_ext.oneD_surf_operator,
                                                soln_basis_ext.oneD_vol_operator,
                                                true, 1.0);//adding=true
        // convective numerical flux
        soln_basis_ext.inner_product_surface_1D(neighbor_iface, conv_num_flux_dot_n[istate],
                                                surf_quad_weights, rhs_ext,
                                                soln_basis_ext.oneD_surf_operator,
                                                soln_basis_ext.oneD_vol_operator,
                                                true, 1.0);//adding=true, scaled by factor=1.0 because negative numerical flux and subtract it
        // dissipative numerical flux
        soln_basis_ext.inner_product_surface_1D(neighbor_iface, diss_auxi_num_flux_dot_n[istate],
                                                surf_quad_weights, rhs_ext,
                                                soln_basis_ext.oneD_surf_operator,
                                                soln_basis_ext.oneD_vol_operator,
                                                true, 1.0);//adding=true, scaled by factor=1.0 because negative numerical flux and subtract it


        for(unsigned int ishape=0; ishape<n_shape_fns_ext; ishape++){
            local_rhs_ext_cell(istate*n_shape_fns_ext + ishape) += rhs_ext[ishape];
        }
    }
}

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
