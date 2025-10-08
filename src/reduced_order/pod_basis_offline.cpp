#include "pod_basis_offline.h"

#include <EpetraExt_MatrixMatrix.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_Map.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/fe/mapping_q1_eulerian.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <eigen/Eigen/SVD>
#include <filesystem>
#include <iostream>
#include <algorithm>

#include "dg/dg_base.hpp"
#include "pod_basis_base.h"
#include "linear_solver/helper_functions.h"

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {

template <int dim>
OfflinePOD<dim>::OfflinePOD(std::shared_ptr<DGBase<dim,double>> &dg_input)
        : basis(std::make_shared<dealii::TrilinosWrappers::SparseMatrix>())
        , dg(dg_input)
        , mpi_communicator(MPI_COMM_WORLD)
        , mpi_rank(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
        , pcout(std::cout, mpi_rank==0)
{
    const bool compute_dRdW = true;
    dg->evaluate_mass_matrices(compute_dRdW);

    pcout << "Searching files..." << std::endl;
    if(dg->all_parameters->reduced_order_param.entropy_varibles_in_snapshots){
        getEntropyPODBasisFromSnapshots();
    } else {
        getPODBasisFromSnapshots();
    }
}

template <int dim>
bool OfflinePOD<dim>::getPODBasisFromSnapshots() {
    bool file_found = false;
    snapshotMatrix.resize(0,0);
    std::string path = dg->all_parameters->reduced_order_param.path_to_search; //Search specified directory for files containing "solutions_table"
    std::string reference_type = "mean";
    std::vector<std::filesystem::path> files_in_directory;
    std::copy(std::filesystem::directory_iterator(path), std::filesystem::directory_iterator(), std::back_inserter(files_in_directory));
    std::sort(files_in_directory.begin(), files_in_directory.end()); //Sort files so that the order is the same as for the sensitivity basis

    for (const auto & entry : files_in_directory){
        if(std::string(entry.filename()).std::string::find("solution_snapshot") != std::string::npos){
            pcout << "Processing " << entry << std::endl;
            file_found = true;
            std::ifstream myfile(entry);
            if(!myfile)
            {
                pcout << "Error opening file." << std::endl;
                std::abort();
            }
            std::string line;
            int rows = 0;
            int cols = 0;
            //First loop set to count rows and columns
            while(std::getline(myfile, line)){ //for each line
                std::istringstream stream(line);
                std::string field;
                cols = 0;
                while (getline(stream, field,' ')){ //parse data values on each line
                    if (field.empty()){ //due to whitespace
                        continue;
                    } else {
                        cols++;
                    }
                }
                rows++;
            }

            snapshotMatrix.conservativeResize(rows, snapshotMatrix.cols()+cols);

            int row = 0;
            myfile.clear();
            myfile.seekg(0); //Bring back to beginning of file
            //Second loop set to build solutions matrix
            while(std::getline(myfile, line)){ //for each line
                std::istringstream stream(line);
                std::string field;
                int col = 0;
                while (getline(stream, field,' ')) { //parse data values on each line
                    if (field.empty()) {
                        continue;
                    } else {
                        snapshotMatrix(row, snapshotMatrix.cols()-cols+col) = std::stod(field); //This will work for however many solutions in each file
                        col++;
                    }
                }
                row++;
            }
            myfile.close();
        }
    }

    pcout << "Snapshot matrix generated." << std::endl;
    calculatePODBasis(snapshotMatrix, reference_type);

    return file_found;
}

template <int dim>
bool OfflinePOD<dim>::getEntropyPODBasisFromSnapshots(){
    //const bool compute_dRdW = true;
    //dg->assemble_residual(compute_dRdW);
    int const nstate = dim+2; // Program this into varible later
    Physics::Euler<dim,nstate,double> euler_physics_double
    = Physics::Euler<dim, nstate, double>(
            dg->all_parameters,
            dg->all_parameters->euler_param.ref_length,
            dg->all_parameters->euler_param.gamma_gas,
            dg->all_parameters->euler_param.mach_inf,
            dg->all_parameters->euler_param.angle_of_attack,
            dg->all_parameters->euler_param.side_slip_angle);
    bool file_found = false;
    auto mpi_comm(MPI_COMM_WORLD);
    const unsigned int n_procs = dealii::Utilities::MPI::n_mpi_processes(mpi_comm);
    int num_of_snapshots = 0;
    int global_quad_points = 0;
    int n_quad_pts = dg->volume_quadrature_collection[dg->all_parameters->flow_solver_param.poly_degree].size();

    const int energy_case = 0;
    const int density_case = nstate-1;

    snapshotMatrix.conservativeResize(0,0);
    MatrixXd density(0,0);
    std::array<MatrixXd,dim> momentum;
    for(int idim = 0; idim < dim; idim++){
        momentum[idim].conservativeResize(0,0);
    }
    MatrixXd energy(0,0);
    std::string path = dg->all_parameters->reduced_order_param.path_to_search; //Search specified directory for files containing "solutions_table"
    std::string reference_type = "zero";
    std::vector<std::filesystem::path> files_in_directory;
    std::copy(std::filesystem::directory_iterator(path), std::filesystem::directory_iterator(), std::back_inserter(files_in_directory));
    std::sort(files_in_directory.begin(), files_in_directory.end()); //Sort files so that the order is the same as for the sensitivity basis

    for (const auto & entry : files_in_directory){
        int old_amount_of_snapshots = snapshotMatrix.cols();
        if(std::string(entry.filename()).std::string::find("solution_snapshot") != std::string::npos){
            pcout << "Processing " << entry << std::endl;
            file_found = true;
            std::ifstream myfile(entry);
            if(!myfile)
            {
                pcout << "Error opening file." << std::endl;
                std::abort();
            }
            std::string line;
            int rows = 0;
            int cols = 0;
            //First loop set to count rows and columns
            while(std::getline(myfile, line)){ //for each line
                std::istringstream stream(line);
                std::string field;
                cols = 0;
                while (getline(stream, field,' ')){ //parse data values on each line
                    if (field.empty()){ //due to whitespace
                        continue;
                    } else {
                        cols++;
                    }
                }
                rows++;
            }
            // ROWS = nstate*global_quad_pts
            // COLS = num_of_snapshots
            num_of_snapshots += cols;
            global_quad_points = rows;
            snapshotMatrix.conservativeResize(rows, old_amount_of_snapshots + 2*cols); // Changing rows from rows/nstate and cols from 2*nstate*cols
            density.conservativeResize(rows/nstate, density.cols() + cols);
            for(int idim = 0; idim < dim; idim++){
                momentum[idim].conservativeResize(rows/nstate, momentum[idim].cols() + cols);
            }
            energy.conservativeResize(rows/nstate, energy.cols() + cols);

            int row = 0;
            int energy_row = 0;
            std::array<int,dim> momentum_row;
            std::fill(momentum_row.begin(),momentum_row.end(), 0);
            int density_row = 0;
            int istate = 0;
            int i_quad = 0;
            myfile.clear();
            myfile.seekg(0); //Bring back to beginning of file
            //Second loop set to build solutions matrix
            while(std::getline(myfile, line)){ //for each line
                std::istringstream stream(line);
                std::string field;
                int col = 0;
                if (i_quad != n_quad_pts) { i_quad++;}
                else { i_quad = 1;istate++;}
                if (istate == nstate){ istate = 0;}
                while (getline(stream, field,' ')) { //parse data values on each line
                    if (field.empty()) {
                        continue;
                    } else {
                        switch(istate){
                            case energy_case:
                                energy(energy_row,energy.cols() - cols + col) = std::stod(field);
                                break;
                            case density_case:
                                density(density_row, density.cols() - cols + col) = std::stod(field);
                                break;
                            default:
                                momentum[istate-1](momentum_row[istate-1],momentum[istate-1].cols() - cols + col) = std::stod(field);
                                break;
                        }
                        col++;
                    }
                }
                switch(istate){
                    case energy_case:
                        energy_row++;
                        break;
                    case density_case:
                        density_row++;
                        break;
                    default:
                        momentum_row[istate-1]++;
                        break;
                }
                row++;
            }
            myfile.close();
        }
    }

    // USING NODAL ENTROPY

    for(int col = 0; col < num_of_snapshots; col++){
        int quad_num = 0;
        int i_quad = -1;
        int solution_num = 0;
        int istate = 0;
        for(int row = 0; row < global_quad_points; row++){
            i_quad++;
            if(i_quad == n_quad_pts){i_quad = 0; istate++;}
            if(istate == nstate){istate = 0; solution_num += n_quad_pts*nstate;quad_num += n_quad_pts;}
            double val  = 0;
            switch(istate){
                case energy_case:
                    val = energy(quad_num+i_quad,col);
                    break;
                case density_case:
                    val = density(quad_num+i_quad,col);
                    break;
                default:
                    val = momentum[istate-1](quad_num+i_quad,col);
                    break;
            }
            if(dg->solution.in_local_range(solution_num+istate*n_quad_pts+i_quad)){
                dg->solution[solution_num+istate*n_quad_pts+i_quad] = val;
            }
            snapshotMatrix(solution_num+istate*n_quad_pts+i_quad,col) = val;
        }
        dg->calculate_global_entropy();

        // Serialize the data in the global entropy

        const unsigned int local_size = dg->global_entropy.local_size();

        std::vector<double> local_data(local_size);
        for(unsigned int i = 0; i < local_size; i++) {
            local_data[i] = dg->global_entropy.local_element(i);
        }

        dealii::LinearAlgebra::distributed::Vector<double> this_entropy(this->dg->global_entropy);
        int back = this_entropy.locally_owned_elements().pop_back();
        int front = this_entropy.locally_owned_elements().pop_front();
        std::vector<std::vector<double>> entropy_vectors = dealii::Utilities::MPI::all_gather(mpi_comm, local_data);
        std::vector<int> back_vector = dealii::Utilities::MPI::all_gather(mpi_comm, back);
        std::vector<int> front_vector = dealii::Utilities::MPI::all_gather(mpi_comm, front);


    for(unsigned int m_proc = 0; m_proc < n_procs; m_proc++) {
        for(int row = 0; row < global_quad_points; row++){
            if(row >= front_vector[m_proc] && row <= back_vector[m_proc]){
                snapshotMatrix(row, col+1*num_of_snapshots) = entropy_vectors[m_proc][row-front_vector[m_proc]];
            }
        }
    }

    for(int row = 0; row < global_quad_points; row++){
        if(dg->solution.in_local_range(row)){
            dg->solution[row] = snapshotMatrix(row, 0);
        }
    }
    }
    pcout << "Snapshot matrix generated." << std::endl;
    calculatePODBasis(snapshotMatrix, reference_type);

    const unsigned int rank = dealii::Utilities::MPI::this_mpi_process(mpi_comm);
    std::ofstream file("Entropy_snapshot_"+std::to_string(rank)+".txt");
    const static Eigen::IOFormat CSVFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
    if (file.is_open()){
        file << snapshotMatrix.format(CSVFormat);
    }
    file.close();
    return !file_found;
}
template <int dim>
void OfflinePOD<dim>::calculatePODBasis(MatrixXd snapshots, std::string reference_type) {
    /* Reference for simple POD basis computation: Refer to Algorithm 1 in the following reference:
    "Efficient non-linear model reduction via a least-squares Petrov–Galerkin projection and compressive tensor approximations"
    Kevin Carlberg, Charbel Bou-Mosleh, Charbel Farhat
    International Journal for Numerical Methods in Engineering, 2011
    */
    VectorXd reference_state;
    VectorXd reference_entropy;
    pcout << "Computing POD basis..." << std::endl;
    if (reference_type == "mean"){
        reference_state = snapshots.rowwise().mean();
    } else if (reference_type == "zero"){
        reference_state = VectorXd::Zero(snapshots.rows(),1);
    }
    referenceState.reinit(reference_state.size());
    for(unsigned int i = 0 ; i < reference_state.size() ; i++){
        referenceState(i) = reference_state(i);
    }

    MatrixXd pod_basis;
    if(mpi_rank == 0) {
        MatrixXd snapshotMatrixCentered = snapshots.colwise() - reference_state;
        Eigen::BDCSVD<MatrixXd, Eigen::DecompositionOptions::ComputeThinU> svd_one(snapshotMatrixCentered);
        pod_basis = svd_one.matrixU();
        // Reduce POD Size using either number of modes or a singular value threshold
        if(dg->all_parameters->reduced_order_param.number_modes > 0){
            const int num_modes = dg->all_parameters->reduced_order_param.number_modes;
            Assert(num_modes < pod_basis.cols(),
            dealii::ExcMessage("The number of modes selected must be less than the number of snapshots"));
            Eigen::MatrixXd pod_basis_n_modes = pod_basis(Eigen::placeholders::all, Eigen::seqN(0,num_modes));
            pod_basis = pod_basis_n_modes;
        }
        else if (dg->all_parameters->reduced_order_param.singular_value_threshold < 1) {
            const double threshold = dg->all_parameters->reduced_order_param.singular_value_threshold;
            Eigen::VectorXd singular_values = svd_one.singularValues();
            double l1_norm = singular_values.sum();
            double singular_value_cumm_sum = 0;
            int iter = 0;
            while(singular_value_cumm_sum/l1_norm < threshold){
                singular_value_cumm_sum += singular_values(iter);
                iter++;
            }
            Eigen::MatrixXd pod_basis_n_modes = pod_basis(Eigen::placeholders::all, Eigen::seqN(0,iter));
            pod_basis = pod_basis_n_modes;
        }
        pcout << "Final size of POD: " << pod_basis.cols() << std::endl;
        fullBasis.reinit(pod_basis.rows(), pod_basis.cols());

        for (unsigned int m = 0; m < pod_basis.rows(); m++) {
            for (unsigned int n = 0; n < pod_basis.cols(); n++) {
                fullBasis.set(m, n, pod_basis(m, n));
            }
        }

        std::ofstream out_file("POD_basis.txt");
        unsigned int precision = 16;
        fullBasis.print_formatted(out_file, precision, true, 0, "0");
        fullBasis.reinit(0,0); // Clear the memory
    }
    Epetra_MpiComm epetra_comm(MPI_COMM_WORLD);
    const Epetra_CrsMatrix epetra_system_matrix  = this->dg->global_mass_matrix.trilinos_matrix();
    Epetra_Map system_matrix_map = epetra_system_matrix.RowMap();
    int GlobalElements = system_matrix_map.NumGlobalElements();
    Epetra_Map rank_zero_row_map(system_matrix_map.NumGlobalElements(),(mpi_rank == 0) ? GlobalElements : 0,0,epetra_comm);
    //Epetra_Map col_map((int)pod_basis.cols(),(int)pod_basis.cols(), 0, epetra_comm);
    int pod_basis_cols = (int)pod_basis.cols();
    epetra_comm.Broadcast(&pod_basis_cols,1,0);
    Epetra_Map domain_map(pod_basis_cols, 0, epetra_comm);

    Epetra_Map rank_zero_domain_map(pod_basis_cols, (mpi_rank == 0) ? pod_basis_cols : 0,0,epetra_comm);
    Epetra_CrsMatrix rank_zero_epetra_basis(Epetra_DataAccess::Copy, rank_zero_row_map, pod_basis_cols);

    const int numMyElements = rank_zero_row_map.NumMyElements(); //Number of elements on the calling processor

    for (int localRow = 0; localRow < numMyElements; ++localRow){
        const int globalRow = rank_zero_row_map.GID(localRow);
        for(int n = 0 ; n < pod_basis.cols() ; n++){
            double value = pod_basis(globalRow, n);
            rank_zero_epetra_basis.InsertGlobalValues(globalRow, 1, &value, &n);
        }
    }
    rank_zero_epetra_basis.FillComplete(rank_zero_domain_map,rank_zero_row_map);
    Epetra_Import importer(system_matrix_map,rank_zero_row_map);
    Epetra_CrsMatrix epetra_basis(Epetra_DataAccess::Copy,system_matrix_map,pod_basis_cols);
    epetra_basis.Import(rank_zero_epetra_basis,importer,Epetra_CombineMode::Insert);
    epetra_basis.FillComplete(domain_map,system_matrix_map);
    basis->reinit(epetra_basis);

    return;
}

template <int dim>
std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> OfflinePOD<dim>::getPODBasis() {
    return basis;
}

template <int dim>
dealii::LinearAlgebra::ReadWriteVector<double> OfflinePOD<dim>::getReferenceState() {
    return referenceState;
}

template <int dim>
MatrixXd OfflinePOD<dim>::getSnapshotMatrix() {
    return snapshotMatrix;
}

template class OfflinePOD <PHILIP_DIM>;

}
}
