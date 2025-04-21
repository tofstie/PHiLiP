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
#include <numeric>

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
    std::cout << "Build first problem" << std::endl;
    std::ofstream testing_file("testing_file.txt");
    for(unsigned int i = 0; i < dg->solution.size(); i++) {
        testing_file << i << " " <<dg->solution[i] << std::endl;
    }
    AssembleGreedyCubature<dim, nstate> First_Cubature_Problem(all_parameters, this->parameter_handler, this->initial_weights, this->b, this->V_target,pod->hyper_reduction_tolerance);
    First_Cubature_Problem.build_problem();
    dealii::LinearAlgebra::distributed::Vector<double> weights = First_Cubature_Problem.get_weights();
    int length_of_weights = weights.size();
    std::cout <<" Length of Weights: " + std::to_string(length_of_weights) << std::endl;
    Eigen::VectorXd final_weights_eigen(length_of_weights); // Set up from final_weights
    weights.extract_subvector_to(weights.begin(),weights.end(),final_weights_eigen.begin());

    std::vector<int> z_vector = First_Cubature_Problem.get_indices();
    Eigen::VectorXd final_weights_size_z = final_weights_eigen(z_vector);
    final_weights = weights;
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> Vt = this->pod->getPODBasis(); //update this to Vt later
    Eigen::MatrixXd Vt_id = epetra_to_eig_matrix(Vt->trilinos_matrix())(z_vector,Eigen::placeholders::all);
    Eigen::MatrixXd diag_weights = final_weights_size_z.asDiagonal();
    Eigen::MatrixXd M_test = Vt_id.transpose()*diag_weights*Vt_id;

    Eigen::EigenSolver<Eigen::MatrixXd> eigensolver;
    eigensolver.compute(M_test);

    
    Eigen::VectorXd eigenvalues = eigensolver.eigenvalues().real();
    Eigen::MatrixXd eigenvectors = eigensolver.eigenvectors().real();
 
    double min_eigenvalue = eigenvalues.minCoeff(); /// The eigen values are strictly greater than 0
    double max_eigenvalue = eigenvalues.maxCoeff();
    double condition_number;
    condition_number = abs(max_eigenvalue)/abs(min_eigenvalue);
    std::cout << "Initial Reciprocal Condition of Mtest: " <<  condition_number << std::endl;
    if(condition_number > 1E10){
        std::cout << "Initial Reciprocal Condition of Mtest: " <<  condition_number << std::endl;
        std::cout << "Getting additional points for stability" << std::endl;
        std::vector<int> p(eigenvalues.size());
        for(unsigned int i = 0; i < p.size();++i){
            p[i] = i;
        }
        
        std::sort(p.begin(),p.end(),[&eigenvalues](int a, int b) {return eigenvalues[a] < eigenvalues[b];});
        std::vector<int> columns_to_keep;
        for (unsigned int i = 0; i < eigenvalues.size(); ++i) {
            std::cout << eigenvalues[p[i]] << std::endl;
            if (eigenvalues[p[i]] > 1e-12) {
                columns_to_keep.push_back(p[i]);
            }
        }
        Eigen::MatrixXd Vx(M_test.rows(),columns_to_keep.size());
        // NEED TO MAKE vX THE RIGHT EIGENVECTORS
        for(unsigned int i = 0;i<columns_to_keep.size(); ++i){
            Vx.col(i) = eigenvectors.col(columns_to_keep[i]);
        }
        Eigen::MatrixXd Z = Vx; // Need to fix this line later, as I had to remove this->pod->getTestBasis() due to a type change
        Eigen::MatrixXd V_mass = this->V_target;
        this->build_chan_target(Z);
        AssembleGreedyCubature<dim, nstate> Second_Cubature_Problem(all_parameters, this->parameter_handler, this->initial_weights, this->b, this->V_target,pod->hyper_reduction_tolerance);
        Second_Cubature_Problem.build_problem();
        dealii::LinearAlgebra::distributed::Vector<double> weights = Second_Cubature_Problem.get_weights();
        std::vector<int> z_additional = Second_Cubature_Problem.get_indices();
        double z_scale = 1e-2;
        std::set<int> all_pts;
        for(const int& i: z_vector){
            all_pts.insert(i);
        }
        for(const int& i: z_additional){
            all_pts.insert(i);
        }
        std::vector<int> unique_pts(all_pts.begin(), all_pts.end());
        Eigen::MatrixXd Z_mass = this->V_target;
        Eigen::MatrixXd J(V_mass.rows(),V_mass.cols()+Z_mass.cols());
        for(int j = 0; j < J.cols(); ++j){
            if(j < V_mass.cols()){
                J.col(j) = V_mass.col(j);
            } else {
                J.col(j) = z_scale*Z_mass.col(j-V_mass.cols());
            }
        }
        Epetra_MpiComm epetra_comm(MPI_COMM_WORLD);
        Eigen::VectorXd b = J.transpose().rowwise().sum();

        Epetra_CrsMatrix J_epetra = eig_to_epetra_matrix(J,J.cols(),J.rows(),epetra_comm);
        Epetra_Vector b_epetra = eig_to_epetra_vector(b,b.size(),epetra_comm);

        NNLS_solver NNLS_prob(all_parameters, this->parameter_handler, J_epetra, epetra_comm, b_epetra);
        std::cout << "Solve NNLS problem...";
        bool exit_con = NNLS_prob.solve();
        std::cout << " Exit code: " << exit_con << std::endl;
        Epetra_Vector w_unscaled = NNLS_prob.getSolution();
        Eigen::MatrixXd w_unscaled_eigen;
        epetra_to_eig_vec(w_unscaled.GlobalLength(),w_unscaled,w_unscaled_eigen);
        double w_unscaled_sum = w_unscaled_eigen.sum();
        Eigen::VectorXd w_eigen = w_unscaled_eigen*w_unscaled_sum/length_of_weights;
        // Eigen::MatrixXd Vt_idt = this->pod->getTestBasis()(unique_pts,Eigen::placeholders::all); Commenting out for now as
        // this->pod->getTestBasis type was changed
        Eigen::MatrixXd Vt_idt = M_test; // Fake line
        final_weights_eigen = w_eigen;
        diag_weights = final_weights_eigen.asDiagonal();    
        Eigen::MatrixXd M_test_2 = Vt_id.transpose()*diag_weights*Vt_id;
        Eigen::EigenSolver<Eigen::MatrixXd> eigensolver2;
        eigensolver2.compute(M_test_2);
        Eigen::VectorXd eigenvalues2 = eigensolver2.eigenvalues().real();
        //Eigen::MatrixXd eigenvectors2 = eigensolver2.eigenvectors().real();
    
        double min_eigenvalue2 = eigenvalues2.minCoeff(); /// The eigen values are strictly greater than 0
        double max_eigenvalue2 = eigenvalues2.maxCoeff();
        double condition_number2;
        condition_number2 = abs(max_eigenvalue2)/abs(min_eigenvalue2);
        std::cout << "Initial Reciprocal Condition of Mtest: " <<  condition_number2 << std::endl;


    }

    
    return;
}

template<int dim, int nstate>
void AssembleGreedyRes<dim,nstate>::build_weights(){
    int rows = this->dg->global_mass_matrix.m()/nstate;
    this->initial_weights.reinit(rows);
    for (auto cell = this->dg->dof_handler.begin_active(); cell!=this->dg->dof_handler.end(); ++cell) {
        if (!cell->is_locally_owned()) continue;

        const unsigned int fe_index_curr_cell = cell->active_fe_index();

        // Current reference element related to this physical cell
        const unsigned int n_quad_pts  = this->dg->volume_quadrature_collection[fe_index_curr_cell].size();
        const int active_cell_index  = cell->active_cell_index();
        const std::vector<double> &quad_weights = this->dg->volume_quadrature_collection[fe_index_curr_cell].get_weights();
        for(int i_quad = 0; i_quad < (int)n_quad_pts; ++i_quad) {
            this->initial_weights[active_cell_index * n_quad_pts + i_quad] = quad_weights[i_quad];
        }
    }
    std::ofstream file("b.txt");
    for(int i =0; i < rows; i++) {
        file << this->initial_weights[i] << std::endl;
    }
    file.close();
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
void AssembleGreedyRes<dim, nstate>::build_chan_target(Eigen::MatrixXd &Input_Matrix){
    std::cout << "Building Chan" << std::endl;
    Eigen::MatrixXd V_t_1 = Input_Matrix;
    Eigen::MatrixXd V_t_2 = Input_Matrix;
    Eigen::MatrixXd V_mass(V_t_1.rows(),V_t_1.cols()*(V_t_1.cols()+1)/2);
    int sk = 0;
    for(int i = 0; i < V_t_1.cols();++i){
        for(int j = i; j < V_t_2.cols();++j){
            V_mass.col(sk) = V_t_1.col(i).array()* V_t_2.col(j).array();
            sk++;
        }
    }
    std::cout << "SVD of big ass matrix" << std::endl;
    Eigen::BDCSVD<MatrixXd, Eigen::DecompositionOptions::ComputeThinU> svd(V_mass);
    Eigen::MatrixXd V_target_temp = svd.matrixU();
    Eigen::VectorXd singular_values = svd.singularValues();
    Eigen::VectorXd singular_values_squared = singular_values.array().square();
    Eigen::VectorXd cum_sum_sv_squared(singular_values.size());
    std::partial_sum(singular_values_squared.begin(),singular_values_squared.end(),cum_sum_sv_squared.begin());
    double l2_norm_squared = singular_values_squared.sum();
    Eigen::VectorXd singular_value_energy = (1-(cum_sum_sv_squared.array()/l2_norm_squared)).sqrt();
    std::ofstream singular_value("sing_value.txt");
    singular_value << singular_value_energy;
    std::vector<int> columns_to_keep;
    double tol = 5.5E-5; // Hard coding tolerance for now based on Chan's results
    for (int i = 0; i < singular_value_energy.size(); ++i) {
        if (singular_value_energy(i) > tol) {
            columns_to_keep.push_back(i);
        }
    }
    Eigen::MatrixXd V_filtered(V_target_temp.rows(), columns_to_keep.size());
    for(unsigned int i = 0; i < columns_to_keep.size(); ++i){
        V_filtered.col(i) = V_target_temp.col(columns_to_keep[i]);
    }
    MatrixXd V_target_deep_copy = V_filtered;
    V_target = V_target_deep_copy;
    V_filtered.transposeInPlace();
    int size_of_weights = this->initial_weights.size();
    Eigen::VectorXd weights_eigen(size_of_weights);
    for(int i = 0; i < size_of_weights; ++i){
        weights_eigen(i) = this->initial_weights(i);
    }
    std::ofstream file("weights_target.txt");
    const static Eigen::IOFormat CSVFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
    if (file.is_open()){
        file << weights_eigen.format(CSVFormat);
    }
    file.close();
    std::cout << "Creating b matrix" << std::endl;
    Eigen::VectorXd b_eigen = V_filtered*weights_eigen;

    this->b.reinit(b_eigen.size());
    for(int i = 0; i < b_eigen.size(); ++i){
        this->b(i) = b_eigen(i);
    }

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
#if PHILIP_DIM==1
template class AssembleGreedyRes <PHILIP_DIM,PHILIP_DIM>;
template class AssembleGreedyRes <PHILIP_DIM,PHILIP_DIM+2>;
#endif

#if PHILIP_DIM!=1
template class AssembleGreedyRes <PHILIP_DIM,1>;
template class AssembleGreedyRes <PHILIP_DIM,2>;
template class AssembleGreedyRes <PHILIP_DIM,3>;
template class AssembleGreedyRes <PHILIP_DIM,4>;
template class AssembleGreedyRes <PHILIP_DIM,5>;
template class AssembleGreedyRes <PHILIP_DIM,6>;
#endif

} /// HyperReduction Namespace
} /// PHiLiP Namespace


