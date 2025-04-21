#include <iomanip>
#include <cmath>
#include <limits>
#include <type_traits>
#include <math.h>
#include <iostream>
#include <stdlib.h>

#include <ctime>

#include <deal.II/distributed/solution_transfer.h>

#include "testing/tests.h"

#include<fstream>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/tensor.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/meshworker/dof_info.h>

// Finally, we take our exact solution from the library as well as volume_quadrature
// and additional tools.
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_dof_data.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools.templates.h>

#include <deal.II/grid/manifold_lib.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/mapping_q_generic.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/mapping_fe_field.h> 

#include "parameters/all_parameters.h"
#include "parameters/parameters.h"
#include "operators/operators.h"

const double TOLERANCE = 1E-6;
using namespace std;



int main (int argc, char * argv[])
{

    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    using real = double;
    using namespace PHiLiP;
    std::cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << std::scientific;
    const int dim = PHILIP_DIM;
    const int nstate = 4;
    dealii::ParameterHandler parameter_handler;
    PHiLiP::Parameters::AllParameters::declare_parameters (parameter_handler);

    PHiLiP::Parameters::AllParameters all_parameters_new;
    all_parameters_new.parse_parameters (parameter_handler);
    all_parameters_new.nstate = nstate;
    dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0);

    using FR_enum = Parameters::AllParameters::Flux_Reconstruction;
    all_parameters_new.flux_reconstruction_type = FR_enum::cHU;

    bool equiv = true;
    bool sum_fact = true;
    for(unsigned int poly_degree=1; poly_degree<4; poly_degree++){
        const unsigned int n_faces = dim*2;
        const unsigned int n_dofs = nstate * pow(poly_degree+1,dim);
        dealii::QGauss<dim> vol_quad_dim (poly_degree+1);
        const dealii::FE_DGQ<dim> fe_dim(poly_degree);
        const dealii::FESystem<dim,dim> fe_system_dim(fe_dim, nstate);

        dealii::QGauss<dim> quad_dimD (poly_degree+1);
        dealii::QGauss<dim-1> face_quad(poly_degree+1);
        dealii::QGauss<1> quad_1D (poly_degree+1);
        dealii::QGauss<0> face_quad1D (poly_degree+1);
        const dealii::FE_DGQ<1> fe(poly_degree);
        const dealii::FESystem<1,1> fe_system(fe, nstate);
        PHiLiP::OPERATOR::basis_functions<dim,2*dim,real> basis_1D(nstate, poly_degree, 1);
        PHiLiP::OPERATOR::vol_integral_basis<dim,2*dim,real> vol_int_1D(nstate, poly_degree, 1);
        PHiLiP::OPERATOR::local_basis_stiffness<dim,2*dim,real> stiffess_1D(nstate, poly_degree, 1,true);
        PHiLiP::OPERATOR::local_mass<dim,2*dim,real> mass_1D(nstate,poly_degree,1);
        PHiLiP::OPERATOR::face_integral_basis<dim,2*dim,real> face_integral_1D(nstate, poly_degree,1);
        PHiLiP::OPERATOR::vol_projection_operator<dim,2*dim,real> projection_1D(nstate, poly_degree,1);
        mass_1D.build_1D_volume_operator(fe,quad_1D);
        basis_1D.build_1D_volume_operator(fe, quad_1D);
        basis_1D.build_1D_surface_operator(fe, face_quad1D);
        basis_1D.build_1D_gradient_operator(fe, quad_1D);
        vol_int_1D.build_1D_volume_operator(fe, quad_1D);
        stiffess_1D.build_1D_volume_operator(fe, quad_1D);
        face_integral_1D.build_1D_surface_operator(fe, face_quad1D);
        projection_1D.build_1D_volume_operator(fe, quad_1D);
        dealii::FullMatrix<double> basis_dim(n_dofs);
        basis_dim = basis_1D.tensor_product(basis_1D.oneD_grad_operator, basis_1D.oneD_vol_operator,basis_1D.oneD_vol_operator);
        dealii::FullMatrix<double>  basis(n_dofs/nstate);
        basis = basis_1D.tensor_product(basis_1D.oneD_vol_operator,basis_1D.oneD_vol_operator,basis_1D.oneD_vol_operator);
        dealii::FullMatrix<double>  vol_int(n_dofs/nstate);
        vol_int = vol_int_1D.tensor_product(vol_int_1D.oneD_vol_operator,vol_int_1D.oneD_vol_operator,vol_int_1D.oneD_vol_operator);
        dealii::FullMatrix<double>  mass(n_dofs/nstate);
        mass = mass_1D.tensor_product(mass_1D.oneD_vol_operator,mass_1D.oneD_vol_operator,mass_1D.oneD_vol_operator);
        dealii::FullMatrix<double>  projection_op(n_dofs/nstate);
        projection_op = projection_1D.tensor_product(projection_1D.oneD_vol_operator,projection_1D.oneD_vol_operator,projection_1D.oneD_vol_operator);
        std::vector<double> weights_1D = quad_1D.get_weights();
        std::vector<double> face_weights_1D = face_quad1D.get_weights();
        std::vector<double> weights = quad_dimD.get_weights();
        std::vector<double> face_weights = face_quad.get_weights();
        dealii::FullMatrix<double> W_1d(weights_1D.size());

        const unsigned int n_quad_1D = weights_1D.size();
        for (unsigned int i =0; i < weights_1D.size(); i++) {
            W_1d.set(i,i,weights_1D[i]);
        }
        dealii::FullMatrix<double> W(weights.size());
        for (unsigned int i =0; i < weights.size(); i++) {
            W.set(i,i,weights[i]);
        }
        dealii::FullMatrix<double> Wf_1d(face_weights_1D.size());
        for (unsigned int i =0; i < face_weights_1D.size(); i++) {
            Wf_1d.set(i,i,face_weights_1D[i]);
        }
        dealii::FullMatrix<double> testing(n_dofs/nstate);
        testing = face_integral_1D.tensor_product(face_integral_1D.oneD_surf_operator[1],basis_1D.oneD_vol_operator,basis_1D.oneD_vol_operator);
        testing.print_formatted(std::cout, 14, true, 10, "0", 1., 0.);
        dealii::FullMatrix<double> int_step(weights.size());
        dealii::FullMatrix<double> mass_no_tensor(weights.size());
        basis.Tmmult(int_step,W);
        int_step.mmult(mass_no_tensor,basis);
        dealii::FullMatrix<double> Qx(n_dofs/nstate);
        dealii::FullMatrix<double> Qy(n_dofs/nstate);

        Qx = stiffess_1D.tensor_product(stiffess_1D.oneD_vol_operator,W_1d,W_1d);
        Qy = stiffess_1D.tensor_product(W_1d,stiffess_1D.oneD_vol_operator,W_1d);
        dealii::FullMatrix<double> QxminusQxt(Qx);
        QxminusQxt.Tadd(-1,Qx);
        dealii::FullMatrix<double> QyminusQyt(Qx);
        QyminusQyt.Tadd(-1,Qy);
        dealii::FullMatrix<double> chi_fx(n_dofs/nstate,n_dofs/nstate);
        dealii::FullMatrix<double> chi_fy(n_dofs/nstate,n_dofs/nstate);
        dealii::FullMatrix<double> Bx(n_faces);
        dealii::FullMatrix<double> By(n_faces);
        dealii::FullMatrix<double> Ex(n_faces,n_dofs/nstate);
        dealii::FullMatrix<double> Ey(n_faces,n_dofs/nstate);
        for(unsigned int i_quad_oneD = 0; i_quad_oneD < n_quad_1D; i_quad_oneD++) {
            for(unsigned int i_face = 0; i_face < n_faces; i_face++) {
                const int i_face_1D = i_face % 2;
                const int i_dim = i_face / 2;
                const dealii::Tensor<1,dim,double> unit_ref_normal_int = dealii::GeometryInfo<dim>::unit_normal_vector[i_face];
                if(i_dim == 0) {
                    Bx.set(i_quad_oneD*n_quad_1D+i_face_1D,i_quad_oneD*n_quad_1D+i_face_1D,face_weights[i_quad_oneD]*unit_ref_normal_int[i_dim]);
                } else {
                    By.set(i_quad_oneD+n_quad_1D*i_face_1D,i_quad_oneD+n_quad_1D*i_face_1D,face_weights[i_quad_oneD]*unit_ref_normal_int[i_dim]);
                }
            }
        }
        std::cout << " Bx : " << std::endl;
        Bx.print_formatted(std::cout, 14, true, 10, "0", 1., 0.);
        std::cout << " By : " << std::endl;
        By.print_formatted(std::cout, 14, true, 10, "0", 1., 0.);
        for (unsigned int iface =0; iface < n_faces; iface++) {
            dealii::FullMatrix<double> chi_small(n_faces/2,n_dofs/nstate);
            unsigned int face_dim = iface / 2;
            unsigned int face_1d = iface % 2;
            if (face_dim == 0) {
                chi_small = basis_1D.tensor_product(face_integral_1D.oneD_surf_operator[face_1d],basis_1D.oneD_vol_operator,basis_1D.oneD_vol_operator);
                chi_fx.add(chi_small,1.,face_1d*2);
            } else {
                chi_small = basis_1D.tensor_product(basis_1D.oneD_vol_operator,face_integral_1D.oneD_surf_operator[face_1d],basis_1D.oneD_vol_operator);
                chi_fy.add(chi_small,1.,face_1d*2);
            }
        }
        chi_fx.mmult(Ex,projection_op);
        chi_fy.mmult(Ey,projection_op);
        std::cout << " Ex: " << std::endl;
        Ex.print_formatted(std::cout, 14, true, 10, "0", 1., 0.);
        std::cout << " Ey: " << std::endl;
        Ey.print_formatted(std::cout, 14, true, 10, "0", 1., 0.);
        dealii::FullMatrix<double> vol_termx(n_dofs/nstate);
        std::cout << " vol_termx: " << std::endl;
        vol_termx.triple_product(Bx,Ex,Ex,true,false);
        vol_termx *= -0.5;
        vol_termx.print_formatted(std::cout, 14, true, 16, "0", 1., 0.);
        dealii::FullMatrix<double> off_diagx(n_dofs/nstate);
        Bx.mmult(off_diagx,Ex);
        std::cout << " off_diagx: " << std::endl;
        off_diagx.print_formatted(std::cout, 14, true, 16, "0", 1., 0.);
//Y
        dealii::FullMatrix<double> vol_termy(n_dofs/nstate);
        std::cout << " vol_termy: " << std::endl;
        vol_termy.triple_product(By,Ey,Ey,true,false);
        vol_termy *= 0.5;
        vol_termy.print_formatted(std::cout, 14, true, 16, "0", 1., 0.);
        dealii::FullMatrix<double> off_diagy(n_dofs/nstate);
        By.mmult(off_diagy,Ey);
        std::cout << " off_diag: " << std::endl;
        off_diagy.print_formatted(std::cout, 14, true, 16, "0", 1., 0.);
        Qx.add(0.5,vol_termx);
        Qy.add(0.5,vol_termy);
        std::cout << "Basis _ not tensored" << std::endl;
        basis_1D.oneD_vol_operator.print_formatted(std::cout, 14, true, 16, "0", 1., 0.);
        std::cout << "Qx" << std::endl;
        Qx.print_formatted(std::cout, 14, true, 10, "0", 1., 0.);
        std::cout << "Qy" << std::endl;
        Qy.print_formatted(std::cout, 14, true, 10, "0", 1., 0.);
        std::cout << "Basis" << std::endl;
        basis.print_formatted(std::cout, 14, true, 10, "0", 1., 0.);
        std::cout << "Wbasis" << std::endl;
        vol_int.print_formatted(std::cout, 14, true, 10, "0", 1., 0.);
        std::cout << "mass" << std::endl;
        mass.print_formatted(std::cout, 14, true, 10, "0", 1., 0.);
        std::cout << "NonTensorMass" << std::endl;
        mass_no_tensor.print_formatted(std::cout, 14, true, 10, "0", 1., 0.);
        for(unsigned int idof=0; idof<n_dofs; idof++){
            for(unsigned int iquad=0; iquad<n_dofs; iquad++){
                dealii::Point<dim> qpoint = vol_quad_dim.point(iquad);
                if(fe_system_dim.shape_grad_component(idof,qpoint,0)[0] != basis_dim[iquad][idof])
                    equiv = false;
            } 
        } 
        if(dim >= 2){
            basis_dim = basis_1D.tensor_product(basis_1D.oneD_vol_operator, basis_1D.oneD_grad_operator,basis_1D.oneD_vol_operator);
            for(unsigned int idof=0; idof<n_dofs; idof++){
                for(unsigned int iquad=0; iquad<n_dofs; iquad++){
                    dealii::Point<dim> qpoint = vol_quad_dim.point(iquad);
                    if(fe_system_dim.shape_grad_component(idof,qpoint,0)[1] != basis_dim[iquad][idof])
                        equiv = false;
                } 
            } 
        }
        if(dim >= 3){
            basis_dim = basis_1D.tensor_product(basis_1D.oneD_vol_operator,basis_1D.oneD_vol_operator, basis_1D.oneD_grad_operator);
            for(unsigned int idof=0; idof<n_dofs; idof++){
                for(unsigned int iquad=0; iquad<n_dofs; iquad++){
                    dealii::Point<dim> qpoint = vol_quad_dim.point(iquad);
                    if(fe_system_dim.shape_grad_component(idof,qpoint,0)[2] != basis_dim[iquad][idof])
                        equiv = false;
                } 
            } 
        }

        std::vector<double> sol_hat(n_dofs);
        for(unsigned int idof=0; idof<n_dofs; idof++){
           // sol_hat[idof] = 1e-8 + static_cast <float> (rand()) / ( static_cast <float> (RAND_MAX/(30-1e-8)));
            sol_hat[idof] = sqrt( 1e-8 + static_cast <float> (rand()) / ( static_cast <float> (RAND_MAX/(30-1e-8))) );
        }
        std::vector<double> sol_dim(n_dofs);
        for(unsigned int idof=0; idof<n_dofs; idof++){
            sol_dim[idof] = 0.0;
            for(unsigned int iquad=0; iquad<n_dofs; iquad++){
                sol_dim[idof] += basis_dim[idof][iquad] * sol_hat[iquad];
            }
        }
        std::vector<double> sol_sum_fact(n_dofs);
        if(dim==1)
            basis_1D.matrix_vector_mult(sol_hat, sol_sum_fact, basis_1D.oneD_grad_operator, basis_1D.oneD_vol_operator, basis_1D.oneD_vol_operator);
        if(dim==2)
            basis_1D.matrix_vector_mult(sol_hat, sol_sum_fact, basis_1D.oneD_vol_operator, basis_1D.oneD_grad_operator, basis_1D.oneD_vol_operator);
        if(dim==3)
            basis_1D.matrix_vector_mult(sol_hat, sol_sum_fact, basis_1D.oneD_vol_operator, basis_1D.oneD_vol_operator, basis_1D.oneD_grad_operator);
        

        for(unsigned int idof=0; idof<n_dofs; idof++){
            if(std::abs(sol_dim[idof] - sol_sum_fact[idof])>1e-12){
                sum_fact = false;
                pcout<<"sum fact wrong "<<sol_dim[idof]<<" "<<sol_sum_fact[idof]<<std::endl;
            }
        }


    }//end of poly_degree loop

    if( equiv == false){
        pcout<<" Tensor product not recover original !"<<std::endl;
        return 1;
    }
    if(sum_fact == false){
        pcout<<" sum fcatorization not recover A*u"<<std::endl;
        return 1;
    }
    else{
        return 0;
    }

}//end of main

