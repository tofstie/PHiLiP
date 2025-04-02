#include "assemble_greedy_cubature.h"
#include "linear_solver/linear_solver.h"
#include "linear_solver/NNLS_solver.h"
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
AssembleGreedyCubature<dim,nstate>::AssembleGreedyCubature(const PHiLiP::Parameters::AllParameters *const parameters_input,
        const dealii::ParameterHandler &parameter_handler_input,
        const dealii::LinearAlgebra::ReadWriteVector<double> &initial_weights,
        dealii::LinearAlgebra::ReadWriteVector<double> &b_input,
        const Eigen::MatrixXd &V_target_input,
        const double tolerance)
        : all_parameters(parameters_input)
        , parameter_handler(parameter_handler_input)
        , mpi_communicator(MPI_COMM_WORLD)
        , V_target(V_target_input)
        , initial_weights(initial_weights)
        , A(std::make_shared<dealii::TrilinosWrappers::SparseMatrix>())
        , cell_count(std::pow(all_parameters->flow_solver_param.number_of_grid_elements_per_dimension,dim))
        , n_quad_pts(std::pow(all_parameters->flow_solver_param.poly_degree+1,dim))
        , tolerance(tolerance)

{
    b = b_input;
    std::ofstream file("V_target.txt");
    const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");
    if (file.is_open()){
        file << V_target.format(CSVFormat);
    }
    file.close();
}

template<int dim, int nstate>
dealii::LinearAlgebra::distributed::Vector<double> AssembleGreedyCubature<dim,nstate>::get_weights(){
    return final_weights;
}

template<int dim, int nstate>
std::vector<int> AssembleGreedyCubature<dim,nstate>::get_indices(){
    return z_vector;
}

template<int dim, int nstate>
void AssembleGreedyCubature<dim,nstate>::build_problem(){
    std::cout << "Starting problem" << std::endl;
    std::set<int> z; // Set of integration points
    std::set<int> y; // Set of candidate points

    for (unsigned int i = 0; i < this->initial_weights.size(); i++){
        y.insert(y.end(),i);
    }

    std::vector<int> y_vector(y.begin(), y.end());
    this->z_vector.assign(z.begin(), z.end());
    int total_num_of_int_points = V_target.rows(); // Number of quad(row)
    unsigned int snapshots_and_weights = V_target.cols();
    int iteration = 0;
    unsigned int non_zeros = 0;
    // Define dealii IndexSet for p+1 size    
    dealii::IndexSet p_plus_one_index(snapshots_and_weights);
    const unsigned int my_procs = dealii::Utilities::MPI::this_mpi_process(this->mpi_communicator);
    const unsigned int n_procs = dealii::Utilities::MPI::n_mpi_processes(this->mpi_communicator);
    const unsigned int size_on_each_core = snapshots_and_weights / n_procs;
    const unsigned int remainder = snapshots_and_weights % n_procs;

    const unsigned int start = my_procs*size_on_each_core + std::min(my_procs, remainder);
    const unsigned int end = start + size_on_each_core + (my_procs < remainder ? 1 : 0);

    p_plus_one_index.add_range(start,end);
    p_plus_one_index.compress();
    dealii::IndexSet n_quad_index(total_num_of_int_points);
    const unsigned int size_on_each_core_quad = total_num_of_int_points / n_procs;
    const unsigned int remainder_quad = total_num_of_int_points % n_procs;

    const unsigned int start_quad = my_procs*size_on_each_core_quad + std::min(my_procs, remainder_quad);
    const unsigned int end_quad = start_quad + size_on_each_core_quad + (my_procs < remainder_quad ? 1 : 0);

    n_quad_index.add_range(start_quad,end_quad);
    n_quad_index.compress();
    // Creating b and the residual to be in the distributed vector for norms
    dealii::LinearAlgebra::distributed::Vector<double> residual(p_plus_one_index, this->mpi_communicator);

    for(unsigned int integration_idx = 0; integration_idx < snapshots_and_weights; integration_idx++){
        if(residual.in_local_range(integration_idx)){
            residual[integration_idx] = b[integration_idx];
        }
    }
    // residual.import(b, dealii::VectorOperation::insert, this->mpi_communicator); // This functionality is added in dealii 9.10
    std::ofstream file_b("b_input.txt");
    dealii::LinearAlgebra::distributed::Vector<double> b_distributed(p_plus_one_index, this->mpi_communicator);
    for(unsigned int integration_idx = 0; integration_idx < snapshots_and_weights; integration_idx++){
        if(b_distributed.in_local_range(integration_idx)){
            b_distributed[integration_idx] = b[integration_idx];
            file_b << b_distributed[integration_idx] << '\n';
        }
    }
    file_b.close();
    // b_distributed.import(b, dealii::VectorOperation::insert, this->mpi_communicator); // Current version is 9.01

    // Storing Varibles to access after while loop
    dealii::LinearAlgebra::distributed::Vector<double> alpha_g;
    dealii::IndexSet m_index_g;
    int old_i = -1; /// Setting this to UINT_MAX to avoid old_i being the same as i. Might need to think of a more elegant solution
    Epetra_MpiComm epetra_comm(MPI_COMM_WORLD);
    std::cout << "start of while loop" << std::endl;
    while(residual.l2_norm()/b_distributed.l2_norm() > tolerance && non_zeros <= snapshots_and_weights){
    /// 1. Compute new point i
        // Reduce J to columns of y
        unsigned int size_of_set_y= y.size();
        //const Epetra_CrsMatrix epetra_V_target  = V_target.trilinos_matrix();
        //Epetra_Map row_matrix_map = epetra_V_target.RowMap();
        Epetra_Map row_matrix_map((int)size_of_set_y, 0, epetra_comm);
        Epetra_Map domain_map((int)V_target.cols(), 0, epetra_comm);
        Epetra_CrsMatrix V_target_y_Epetra(Epetra_DataAccess::Copy, row_matrix_map, size_of_set_y);
        /* Commenting this out for now
        for(unsigned int row = 0; row < snapshots_and_weights; row++){
                int numEntries;
                epetra_V_target.NumGlobalEntries(i, numEntries);
                double* value = new double[numEntries];
                int* indices = new int[numEntries];
                epetra_V_target.ExtractGlobalRowCopy(row, 1, numEntries, &value, &col_number);
            for(std::set<int>::iterator col = y.begin(); col != y.end(); col++){
 
                V_target_y_Epetra.InsertGlobalValues(row, 1, &value, &col_number); // Pointers here may be funky
            }
        }
        */
        Eigen::MatrixXd V_target_y_Eigen = V_target(y_vector, Eigen::placeholders::all);
        for(int row = 0; row < V_target_y_Eigen.rows(); row++){
            double row_norm = V_target_y_Eigen.row(row).norm();
            V_target_y_Eigen.row(row) /= row_norm;
        }
        std::cout << "Building Vy" << std::endl;
        const int numMyElements_y = row_matrix_map.NumMyElements(); //Number of elements on the calling processor
        for (int localRow = 0; localRow < numMyElements_y; ++localRow){
            const int globalRow = row_matrix_map.GID(localRow);
            for(int n = 0 ; n < V_target_y_Eigen.cols() ; n++){
                V_target_y_Epetra.InsertGlobalValues(globalRow, 1, &V_target_y_Eigen(globalRow, n), &n);
            }
        }
        V_target_y_Epetra.FillComplete(domain_map,row_matrix_map);
        dealii::TrilinosWrappers::SparseMatrix V_target_y;
        V_target_y.reinit(V_target_y_Epetra);
        /// Compute the i_vector
        std::cout << "compute i vector" << std::endl;
        double norm_residual = residual.linfty_norm();
        residual /= norm_residual;
        dealii::IndexSet y_map = V_target_y.locally_owned_range_indices();
        dealii::LinearAlgebra::distributed::Vector<double> i_values_in_set_y(y_map, this->mpi_communicator);
        V_target_y.vmult(i_values_in_set_y, residual);
        /// Below I need to be careful with multiple cores as the i may be different on different cores leading to complications
        /// Testing will need to be done after getting it to work on 1 core

        //dealii::Utilities::MPI::MinMaxAvg indexstore;
        //int processor_containing_max;
        std::cout << V_target_y_Eigen.rows() << std::endl;


        // int smudge_factor = 0;
        // for (int quad = 0; quad < (int)i_values_at_each_quad_pt.size() ; quad++) {
        //     const int cell_number = quad / n_quad_pts;
        //     const int local_quad_num = quad % n_quad_pts;
        //     int starting_index = cell_number * (nstate*n_quad_pts)+local_quad_num;
        //     if(z.find(starting_index) != z.end()) {
        //         i_values_at_each_quad_pt(quad) = 0;
        //         smudge_factor++;
        //     } else {
        //         starting_index -= smudge_factor;
        //         for(int istate = 0; istate < nstate; istate++) {
        //             std::cout << starting_index+istate*n_quad_pts << " " << quad << std::endl;
        //             i_values_at_each_quad_pt(quad) += i_values_in_set_y(starting_index+istate*n_quad_pts);
        //         }
        //     }
        // }

        std::cout << "out" << std::endl;
        auto local_max_iteration = std::max_element(i_values_in_set_y.begin(),i_values_in_set_y.end());
        double local_max_value = *local_max_iteration;
        double max_value;
        MPI_Allreduce(&local_max_value, &max_value, 1, MPI_DOUBLE, MPI_MAX, this->mpi_communicator);

        //indexstore = dealii::Utilities::MPI::min_max_avg(i_values_in_set_y, this->mpi_communicator);
        //processor_containing_max = indexstore.max_index;
        //max_value = indexstore.max;
        unsigned int local_max_idx = std::distance(i_values_in_set_y.begin(),local_max_iteration);
        unsigned int i;
        const int mpi_rank = dealii::Utilities::MPI::this_mpi_process(this->mpi_communicator);
        if(local_max_value == max_value){
            const auto local_elements = i_values_in_set_y.locally_owned_elements();
            i = *local_elements.begin() + local_max_idx;
        }
        MPI_Bcast(&i, 1, MPI_UNSIGNED, mpi_rank, this->mpi_communicator);
        //static_assert(max_value != 0.,"A point in set z was selected in the Greedy Empirical cubatrure method");
        /// Broadcast to all cores the index of i
        /*
        for(unsigned int int_point  = 0; int_point < snapshots_and_weights; int_point++){
            if(i_values_in_set_y.in_local_range(int_point) && this->mpi_rank = processor_containing_max){
                if(max_value == i_values_in_set_y[int_point]){
                    MPI_Bcast(int_point, i, MPI_UNSIGNED, processor_containing_max, this->mpi_communicator);
                    break;
                }
            }
        }
        */
    ///2. Move i from y to z
        auto y_i = std::next(y.begin(), i);
        z.insert(*y_i);
        y.erase(y_i);

        unsigned int size_of_set_z = z.size();
        y_vector.assign(y.begin(),y.end());
        this->z_vector.assign(z.begin(),z.end());
        // for (long unsigned int j = 0; j < this->z_vector.size(); j++) {
        //     std::cout << this->z_vector[j] << std::endl;
        // }
        // Create IndexSet for size m
        dealii::IndexSet m_index(size_of_set_z);
        
        const unsigned int size_on_each_core_m = size_of_set_z / n_procs;
        const unsigned int remainder_m = size_of_set_z % n_procs;

        const unsigned int start = my_procs*size_on_each_core_m + std::min(my_procs, remainder_m);
        const unsigned int end = start + size_on_each_core_m + (my_procs < remainder_m ? 1 : 0);

        m_index.add_range(start,end);
        m_index.compress();
        m_index_g = m_index;
    ///3. Preform Least Squares
        /// Reassembling Target matrix to only include z indices
        
        Epetra_Map row_map_z((int)size_of_set_z, 0, epetra_comm);
        Epetra_CrsMatrix V_target_z_Epetra(Epetra_DataAccess::Copy, row_map_z, (int)V_target.cols());
        Epetra_CrsMatrix V_target_z_T_Epetra(Epetra_DataAccess::Copy, domain_map, (int)size_of_set_z);
        Eigen::MatrixXd V_target_z_Eigen = V_target(this->z_vector, Eigen::placeholders::all); 
        Eigen::MatrixXd V_target_z_T_Eigen = V_target(this->z_vector, Eigen::placeholders::all);
        V_target_z_T_Eigen.transposeInPlace();
        // Building Vz
        const int numMyElements_z = row_map_z.NumMyElements(); //Number of elements on the calling processor
        for (int localRow = 0; localRow < numMyElements_z; ++localRow){
            const int globalRow = row_map_z.GID(localRow);
            for(int n = 0 ; n < V_target_z_Eigen.cols() ; n++){
                V_target_z_Epetra.InsertGlobalValues(globalRow, 1, &V_target_z_Eigen(globalRow, n), &n);
            }
        }
        V_target_z_Epetra.FillComplete(domain_map,row_map_z);
        // Building VzT
        const int numMyElements_z_T = domain_map.NumMyElements(); //Number of elements on the calling processor
        for (int localRow = 0; localRow < numMyElements_z_T; ++localRow){
            const int globalRow = domain_map.GID(localRow);
            for(int n = 0 ; n < V_target_z_T_Eigen.cols() ; n++){
                V_target_z_T_Epetra.InsertGlobalValues(globalRow, 1, &V_target_z_T_Eigen(globalRow, n), &n);
            }
        }
        V_target_z_T_Epetra.FillComplete(row_map_z,domain_map);

        dealii::TrilinosWrappers::SparseMatrix V_target_z;
        V_target_z.reinit(V_target_z_Epetra);
        dealii::TrilinosWrappers::SparseMatrix V_target_z_T;
        V_target_z_T.reinit(V_target_z_T_Epetra);
        std::cout << "V_target_z_T rows: " << V_target_z_T.m() << " cols: " << V_target_z_T.n() << std::endl;
        std::cout << "V_target_z rows: " << V_target_z.m() << " cols: " << V_target_z.n() << std::endl;
        dealii::LinearAlgebra::distributed::Vector<double> alpha(m_index, this->mpi_communicator);
        dealii::TrilinosWrappers::SparseMatrix LS_LHS;
        dealii::LinearAlgebra::distributed::Vector<double> LS_RHS(alpha);
        const Parameters::LinearSolverParam Linear_Solver_Param = this->all_parameters->linear_solver_param;
        V_target_z.mmult(LS_LHS, V_target_z_T);
        V_target_z.vmult(LS_RHS, b_distributed);
        solve_linear (
            LS_LHS,
            LS_RHS,
            alpha,
            Linear_Solver_Param);
    ///4. Check if all Entries of Alpha are non-negative
        //dealii::Utilities::MPI::MinMaxAvg minstore;
        auto local_min_iteration = std::min_element(alpha.begin(),alpha.end());
        double local_min_value = *local_min_iteration;
        double min_value;
        MPI_Allreduce(&local_min_value, &min_value, 1, MPI_DOUBLE, MPI_MIN, this->mpi_communicator);
        std::cout << "Min value: "<< min_value << std::endl;
        //minstore = dealii::Utilities::MPI::min_max_avg(alpha, this->mpi_communicator);
        //min_value = minstore.min;
    ///5. Preform NNLS
        if(min_value <= 0.){
            std::cout << "Create NNLS problem..."<< std::endl;
            V_target_z_Eigen.transposeInPlace();
            Epetra_CrsMatrix V_target_z_Epetra_nnls_solver(Epetra_DataAccess::Copy, domain_map, total_num_of_int_points);
            const int numMyElements_nnls_solver = domain_map.NumMyElements(); //Number of elements on the calling processor
            for (int localRow = 0; localRow < numMyElements_nnls_solver; ++localRow){
                const int globalRow = domain_map.GID(localRow);
                for(int n = 0 ; n < V_target_z_Eigen.cols() ; n++){
                    V_target_z_Epetra_nnls_solver.InsertGlobalValues(globalRow, 1, &V_target_z_Eigen(globalRow, n), &n);
                }
            }
            V_target_z_Epetra_nnls_solver.FillComplete(row_map_z,domain_map);
            Epetra_Vector b_Epetra(Epetra_DataAccess::View, domain_map, b_distributed.begin());
            /*  Todo: NNLS must always be solved with one core. However, it would be nice to run with multiple cores
                One way to achieve this would be to build single core versions of each of the matrices used below,
                after all MPI processes catch up with MPI_Barrier, compute the rank=0 ones, then redistribute the 
                epetra matrices. This is not implemented until 1 core is working.
            */
            
            NNLS_solver NNLS_prob(this->all_parameters, this->parameter_handler, V_target_z_Epetra_nnls_solver, epetra_comm, b_Epetra, false, 200,1E-8 );
            std::cout << "Solve NNLS problem..."<< std::endl;
            bool exit_con = NNLS_prob.solve();
            std::cout << exit_con << std::endl;

            Epetra_Vector alpha_nnls_epetra = NNLS_prob.getSolution(); // Keep in Epetra until the new size of z can be determined
            double l2_norm_alpha;
            alpha_nnls_epetra.Norm2(&l2_norm_alpha);
    ///6. Fiddle with Sets for all alpha(z_0) = 0 
            // Fiddle here

            std::cout << "Fiddle" <<std::endl;
            std::set<int> z_0;
            for(unsigned int idx = 0; idx < size_of_set_z; idx++){
                if(alpha_nnls_epetra[idx] == 0.){ // Float comp here, careful.
                    auto it = std::next(z.begin(), idx);
                    z_0.insert(*it); // Due to parallelization, this may always return 0 on non-local indices. More investigation will have to take place
                }                    // Need to be careful as z_0 would not be shared. If using NNLS (1-core) maybe keep it in one core here
            }
            /// Remove z_0 from z and add to y
            for(auto value_in_z_0: z_0){
                y.insert(value_in_z_0);
                z.erase(value_in_z_0);
            }
            std::cout << "Recreate" <<std::endl;

            /// Recreate m_index set after changing size of z
            size_of_set_z = z.size();
            y_vector.assign(y.begin(),y.end());
            this->z_vector.assign(z.begin(),z.end());
            // for (long unsigned int j = 0; j < this->z_vector.size(); j++) {
            //     std::cout << this->z_vector[j] << std::endl;
            // }
            m_index.clear();
            m_index.set_size(size_of_set_z);
            const unsigned int size_on_each_core_m_nlss = size_of_set_z / n_procs;
            const unsigned int remainder_m_nlss = size_of_set_z % n_procs;

            const unsigned int start = my_procs*size_on_each_core_m_nlss + std::min(my_procs, remainder_m_nlss);
            const unsigned int end = start + size_on_each_core_m_nlss + (my_procs < remainder_m_nlss ? 1 : 0);

            m_index.add_range(start,end);
            m_index.compress();
            m_index_g = m_index;
            /// Only get the new indices in z
            Epetra_Map row_map_z_nnls((int)size_of_set_z, 0, epetra_comm);
            Epetra_CrsMatrix V_target_z_Epetra_nnls(Epetra_DataAccess::Copy, row_map_z_nnls, (int)V_target.cols());
            Eigen::MatrixXd V_target_z_Eigen_nnls = V_target(this->z_vector, Eigen::placeholders::all); 
            const int numMyElements_nnls = row_map_z_nnls.NumMyElements(); //Number of elements on the calling processor
            for (int localRow = 0; localRow < numMyElements_nnls; ++localRow){
                const int globalRow = row_map_z_nnls.GID(localRow);
                for(int n = 0 ; n < V_target_z_Eigen_nnls.cols() ; n++){
                    V_target_z_Epetra_nnls.InsertGlobalValues(globalRow, 1, &V_target_z_Eigen_nnls(globalRow, n), &n);
                }
            }
            V_target_z_Epetra_nnls.FillComplete(domain_map,row_map_z_nnls);
            V_target_z.reinit(V_target_z_Epetra_nnls);
            // Remake smaller alpha
            std::cout << "Remakeing alpha" <<std::endl;
            alpha.reinit(m_index, this->mpi_communicator);
            std::set<int>::iterator z_iter = z.begin();
            for(unsigned int new_idx = 0; new_idx < size_of_set_z; new_idx++){
                alpha[new_idx] = alpha_nnls_epetra[*z_iter];
                std::next(z_iter,1);
            }
            
        }
    ///7. Update the Residual
        dealii::LinearAlgebra::distributed::Vector<double> Jz_alpha(b_distributed);
        V_target_z.Tvmult(Jz_alpha, alpha);
        residual.reinit(b_distributed); // Reset to Zero Vector
        residual.add(1,b_distributed,-1,Jz_alpha);

    ///8. Update Iterations and store last value of alpha
        iteration++;
        if (iteration % 1 == 0) {
            std::cout << "Points: " << size_of_set_z << std::endl;
            std::cout << "Residual L2: " << residual.l2_norm() << std::endl;
            std::cout << "Escape: " << residual.l2_norm()/b_distributed.l2_norm() << std::endl;
        }
        non_zeros = size_of_set_z; // Equal to Cardinality of set z
        alpha_g.reinit(alpha);
        alpha_g.import(alpha,dealii::VectorOperation::insert);
        if(old_i != *y_i){ // Remove posibility of infinite loops, might be better to add a random point from set y instead or a iter counter
            old_i = *y_i;
        } else {
            std::cout << "Infinite Loop, breaking" << std::endl;
            //break;
        }
    };
    // Building weights of set z
    std::cout << "Free" << std::endl;
    dealii::LinearAlgebra::distributed::Vector<double> initial_weights_set_z(alpha_g);
    int set_iter = 0;
    this->final_weights.reinit(n_quad_index,this->mpi_communicator);
    for(std::set<int>::iterator idx = z.begin(); idx != z.end(); idx++, set_iter++) {
        this->final_weights[*idx] = alpha_g[set_iter];
    }
    // for(std::set<int>::iterator idx = z.begin(); idx != z.end(); idx++){
    //     initial_weights_set_z[set_iter] = this->initial_weights[*idx];
    //     set_iter++;
    // }
    // this->final_weights.reinit(alpha_g);
    // for(unsigned int idx = 0; idx <  this->final_weights.size(); idx++){
    //     this->final_weights[idx] = sqrt(initial_weights_set_z[idx])*alpha_g[idx];
    // }

    return;
}


template <int dim, int nstate>
void AssembleGreedyCubature<dim,nstate>::epetra_to_dealii(Epetra_Vector &epetra_vector, 
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
template class AssembleGreedyCubature <PHILIP_DIM,PHILIP_DIM>;
template class AssembleGreedyCubature <PHILIP_DIM,PHILIP_DIM+2>;
#endif

#if PHILIP_DIM!=1
template class AssembleGreedyCubature <PHILIP_DIM,1>;
template class AssembleGreedyCubature <PHILIP_DIM,2>;
template class AssembleGreedyCubature <PHILIP_DIM,3>;
template class AssembleGreedyCubature <PHILIP_DIM,4>;
template class AssembleGreedyCubature <PHILIP_DIM,5>;
template class AssembleGreedyCubature <PHILIP_DIM,6>;
#endif
} /// HyperReduction Namespace
} /// PHiLiP Namespace


