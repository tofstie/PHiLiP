#include "assemble_greedy_residual.h"
#include "assemble_greedy_cubature.h"
#include "linear_solver/linear_solver.h"
#include "linear_solver/NNLS_solver.h"
#include "linear_solver/helper_functions.h"
#include "parameters/all_parameters.h"

#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/mpi.h>
#include <deal.II/lac/vector_operation.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools.templates.h>

#include <eigen/Eigen/Dense>

#include <Epetra_CrsMatrix.h>
#include <Epetra_Map.h>
#include <Epetra_Vector.h>
#include <EpetraExt_MatrixMatrix.h>

#include <cmath>
#include <set>
#include <iostream>

namespace PHiLiP
{
namespace HyperReduction
{
template <int dim, int nstate>
AssembleGreedyRes<dim,nstate>::AssembleGreedyRes(const PHiLiP::Parameters::AllParameters *const parameters_input,
        const dealii::ParameterHandler &parameter_handler_input,
        std::shared_ptr<DGBase<dim,double>> &dg_input, 
        std::shared_ptr<ProperOrthogonalDecomposition::PODBase<dim>> pod,
        Parameters::ODESolverParam::ODESolverEnum &ode_solver_type)
        : all_parameters(parameters_input)
        , parameter_handler(parameter_handler_input)
        , dg(dg_input)
        , pod(pod)
        , mpi_communicator(MPI_COMM_WORLD)
        , ode_solver_type(ode_solver_type)
        , A(std::make_shared<dealii::TrilinosWrappers::SparseMatrix>())
{}

template<int dim, int nstate>
void AssembleGreedyRes<dim,nstate>::build_problem(){
    AssembleGreedyCubature<dim, nstate> First_Cubature_Problem(all_parameters, this->parameter_handler, this->initial_weights, this->b, this->V_target);
    First_Cubature_Problem.build_problem();
    dealii::LinearAlgebra::distributed::Vector<double> weights = First_Cubature_Problem.get_weights();
    int length_of_weights = weights.size();
    Eigen::VectorXd final_weights_eigen(length_of_weights); // Set up from final_weights
    weights.extract_subvector_to(weights.begin(),weights.end(),final_weights_eigen.begin());
    std::vector<int> z_vector = First_Cubature_Problem.get_indices();
    Eigen::MatrixXd Vt_id = this->pod->getTestBasis()(z_vector,Eigen::placeholders::all); 
    Eigen::MatrixXd diag_weights = final_weights_eigen.asDiagonal();
    Eigen::MatrixXd M_test = Vt_id.transpose()*diag_weights*Vt_id;

    dealii::LAPACKFullMatrix lapack_M_test = eig_to_lapack_matrix(M_test);
    double l1_norm = lapack_M_test.l1_norm();
    double condition_number_reciprocal = 1E-11;
    //{ // Block here to destroy a temp M_test for the condition number (Change to function later?)
        dealii::LAPACKFullMatrix M_test_temp(lapack_M_test);
        M_test_temp.compute_cholesky_factorization(); // This assumes that M_test is SPD, which it will be no matter Vt
        condition_number_reciprocal = M_test_temp.reciprocal_condition_number(l1_norm); 
    ///}
    std::cout << "Initial Reciprocal Condition of Mtest: " <<  condition_number_reciprocal << std::endl;
    if(condition_number_reciprocal < 1E-10){
        std::cout << "Initial Reciprocal Condition of Mtest: " <<  condition_number_reciprocal << std::endl;
        std::cout << "Getting additional points for stability" << std::endl;
    }

    
    return;
}

template<int dim, int nstate>
void AssembleGreedyRes<dim,nstate>::build_weights(){
    int rows = this->dg->global_mass_matrix.m();
    this->initial_weights.reinit(rows);
    for(int i = 0; i < rows; i++){
        this->initial_weights[i] = this->dg->global_mass_matrix.diag_element(i);
    }
    return;
}


template<int dim, int nstate>
void AssembleGreedyRes<dim, nstate>::build_initial_target(){
    Eigen::MatrixXd snapshot_matrix = this->pod->getSnapshotMatrix();
    Eigen::MatrixXd adjusted_snapshot_matrix(snapshot_matrix.rows(), snapshot_matrix.cols());
    Eigen::VectorXd weights(this->initial_weights.size());
    double V = 0;
    for (int I = 0; I < snapshot_matrix.cols(); I++){
        double F_I = 0;
        Eigen::VectorXd f_I_hat(snapshot_matrix.rows());
        for(int intergration_point = 0; intergration_point < snapshot_matrix.rows(); intergration_point++){
            weights[intergration_point] = this->initial_weights[intergration_point];
            F_I += this->initial_weights[intergration_point] * snapshot_matrix(intergration_point,I);
            if(I == 0){
                V += this->initial_weights[intergration_point];
            }
        }
        Eigen::VectorXd f_I_hat_temp = F_I/V*Eigen::VectorXd::Ones(snapshot_matrix.rows());
        f_I_hat = snapshot_matrix.col(I) - f_I_hat_temp;
        // Haramard Product of sqrt(w) and (RHS-F_I/V 1)
        for(int intergration_point = 0; intergration_point < snapshot_matrix.rows(); intergration_point++){
            f_I_hat[intergration_point] = sqrt(this->initial_weights[intergration_point]) * f_I_hat[intergration_point];
        }
        adjusted_snapshot_matrix.col(I) = f_I_hat;
    }
    // Building b
    this->b.reinit(adjusted_snapshot_matrix.cols()+1); // p+1 size
    this->b[this->b.size()-1] = V;
    // Building J
    for (unsigned int idx = 0; idx < weights.size(); idx++){
        weights[idx] = sqrt(weights[idx]);
    }
    Eigen::BDCSVD<Eigen::MatrixXd, Eigen::DecompositionOptions::ComputeThinU> svd(adjusted_snapshot_matrix);
    Eigen::MatrixXd Lamda = svd.matrixU();
    Lamda.conservativeResize(Lamda.rows(),Lamda.cols()+1);
    Lamda.col(Lamda.cols()-1) = weights.transpose();
    Lamda.transposeInPlace(); 
    V_target = Lamda;
    // Tranpose from equation 55 📢 Might require some dim investigation
    // covert Lamda to dealii::TrilinosWrappers::SparseMatrix
    //const Epetra_CrsMatrix epetra_system_matrix  = this->dg->global_mass_matrix.trilinos_matrix();
    //Epetra_Map system_matrix_map = epetra_system_matrix.RowMap();

    /* Commenting this section out as I might just carry V_target as MatrixXd for easy splicing for sets
    Epetra_MpiComm epetra_comm(MPI_COMM_WORLD);
    Epetra_Map row_map((int)Lamda.rows(), 0, epetra_comm);
    Epetra_Map domain_map((int)Lamda.cols(), 0, epetra_comm);
    //std::ofstream system_map_file("system_map_file.txt");
    //system_matrix_map.Print(system_map_file);
    Epetra_CrsMatrix epetra_basis(Epetra_DataAccess::Copy, row_map, Lamda.cols());

    const int numMyElements = row_map.NumMyElements(); //Number of elements on the calling processor
    for (int localRow = 0; localRow < numMyElements; ++localRow){
        const int globalRow = row_map.GID(localRow);
        for(int n = 0 ; n < Lamda.cols() ; n++){
            epetra_basis.InsertGlobalValues(globalRow, 1, &Lamda(globalRow, n), &n);
        }
    }
   

    epetra_basis.FillComplete(domain_map, row_map);
    this->V_target.reinit(epetra_basis);
    */
    return;
}

template<int dim, int nstate>
void AssembleGreedyRes<dim, nstate>::build_chan_target(){
    Eigen::MatrixXd V_t_1 = this->pod->getTestBasis();
    Eigen::MatrixXd V_t_2 = this->pod->getTestBasis();
    Eigen::MatrixXd V_mass(V_t_1.rows(),V_t_1.cols()*(V_t_1.col()+1)/2);
    int sk = 0;
    for(int i = 0; i < V_t_1.cols();++i){
        for(int j = 0; j < V_t_2.cols();++j){
            V_mass.col(sk) = V_t_1.col(i).array()* V_t_2.col(j).array();
        }
    }
    Eigen::BDCSVD<MatrixXd, Eigen::DecompositionOptions::ComputeThinU> svd(V_mass);
    Eigen::MatrixXd V_target_temp = svd.matrixU();
    Eigen::VectorXd singular_values = svd.singularValues();

    V_target = V_target_temp
    this->b.reinit(V_target.cols()); // Fix this
    
}

template <int dim, int nstate>
void AssembleGreedyRes<dim,nstate>::epetra_to_dealii(Epetra_Vector &epetra_vector, 
                                                    dealii::LinearAlgebra::distributed::Vector<double> &dealii_vector,
                                                    dealii::IndexSet index_set)
{
    const Epetra_BlockMap &epetra_map = epetra_vector.Map();
    dealii_vector.reinit(index_set,this->mpi_communicator); // Need one for different size (reduced), one approach is to take in an IndexSet
    for(int i = 0; i < epetra_map.NumMyElements();++i){
        int epetra_global_idx = epetra_map.GID(i);
        int dealii_global_idx = epetra_global_idx;
        if(dealii_vector.in_local_range(dealii_global_idx)){
            dealii_vector[dealii_global_idx] = epetra_vector[epetra_global_idx];
        }
    }
    dealii_vector.compress(dealii::VectorOperation::insert);

}

template class AssembleGreedyRes<PHILIP_DIM,PHILIP_DIM+2>;

} /// HyperReduction Namespace
} /// PHiLiP Namespace


