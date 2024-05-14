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

#include "dg/dg_base.hpp"
#include "pod_basis_base.h"

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
    //const bool compute_dRdW = true;
    //dg->assemble_residual(compute_dRdW); // Curently there is an error with this line ... Was there before I touched it
    //Additional information: 
    //Dimension 12288 not equal to 0.

    pcout << "Searching files..." << std::endl;
    addEntropyVaribles();
    //getPODBasisFromSnapshots();
}

template <int dim>
bool OfflinePOD<dim>::getPODBasisFromSnapshots() {
    bool file_found = false;
    MatrixXd snapshotMatrix(0,0);
    std::string path = dg->all_parameters->reduced_order_param.path_to_search; //Search specified directory for files containing "solutions_table"

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


    /* Reference for simple POD basis computation: Refer to Algorithm 1 in the following reference:
    "Efficient non-linear model reduction via a least-squares Petrov–Galerkin projection and compressive tensor approximations"
    Kevin Carlberg, Charbel Bou-Mosleh, Charbel Farhat
    International Journal for Numerical Methods in Engineering, 2011
    */

    pcout << "Computing POD basis..." << std::endl;

    VectorXd reference_state = snapshotMatrix.rowwise().mean();

    referenceState.reinit(reference_state.size());
    for(unsigned int i = 0 ; i < reference_state.size() ; i++){
        referenceState(i) = reference_state(i);
    }

    MatrixXd snapshotMatrixCentered = snapshotMatrix.colwise() - reference_state;

    Eigen::BDCSVD<MatrixXd, Eigen::DecompositionOptions::ComputeThinU> svd(snapshotMatrixCentered);
    MatrixXd pod_basis = svd.matrixU();

    fullBasis.reinit(pod_basis.rows(), pod_basis.cols());

    for (unsigned int m = 0; m < pod_basis.rows(); m++) {
        for (unsigned int n = 0; n < pod_basis.cols(); n++) {
            fullBasis.set(m, n, pod_basis(m, n));
        }
    }

    std::ofstream out_file("POD_basis.txt");
    unsigned int precision = 16;
    fullBasis.print_formatted(out_file, precision);

    const Epetra_CrsMatrix epetra_system_matrix  = this->dg->system_matrix.trilinos_matrix();
    Epetra_Map system_matrix_map = epetra_system_matrix.RowMap();
    Epetra_CrsMatrix epetra_basis(Epetra_DataAccess::Copy, system_matrix_map, pod_basis.cols());

    const int numMyElements = system_matrix_map.NumMyElements(); //Number of elements on the calling processor

    for (int localRow = 0; localRow < numMyElements; ++localRow){
        const int globalRow = system_matrix_map.GID(localRow);
        for(int n = 0 ; n < pod_basis.cols() ; n++){
            epetra_basis.InsertGlobalValues(globalRow, 1, &pod_basis(globalRow, n), &n);
        }
    }

    Epetra_MpiComm epetra_comm(MPI_COMM_WORLD);
    Epetra_Map domain_map((int)pod_basis.cols(), 0, epetra_comm);

    epetra_basis.FillComplete(domain_map, system_matrix_map);

    basis->reinit(epetra_basis);

    return file_found;
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
bool OfflinePOD<dim>::addEntropyVaribles(){
    int const nstate = 3; // Program this into varible later
    Physics::Euler<dim,nstate,double> euler_physics_double
    = Physics::Euler<dim, nstate, double>(
            dg->all_parameters,
            dg->all_parameters->euler_param.ref_length,
            dg->all_parameters->euler_param.gamma_gas,
            dg->all_parameters->euler_param.mach_inf,
            dg->all_parameters->euler_param.angle_of_attack,
            dg->all_parameters->euler_param.side_slip_angle);
    bool file_found = false;
    int num_of_snapshots;
    int global_quad_points;
    int n_quad_pts = dg->volume_quadrature_collection[dg->all_parameters->flow_solver_param.poly_degree].size();
    MatrixXd snapshotMatrix(0,0);
    MatrixXd density(0,0);
    MatrixXd momentum(0,0);
    MatrixXd energy(0,0);
    std::string path = dg->all_parameters->reduced_order_param.path_to_search; //Search specified directory for files containing "solutions_table"

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
            // ROWS = nstate*global_quad_pts
            // COLS = num_of_snapshots
            num_of_snapshots = cols;
            global_quad_points = rows/nstate;
            snapshotMatrix.conservativeResize(rows/nstate, 2*nstate*cols);
            density.conservativeResize(rows/nstate, cols);
            momentum.conservativeResize(rows/nstate, cols);
            energy.conservativeResize(rows/nstate, cols);

            int row = 0;
            int energy_row = 0;
            int momentum_row = 0;
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
                            case 0:
                                energy(energy_row,col) = std::stod(field);
                                break;
                            case 1:
                                momentum(momentum_row,col) = std::stod(field);
                                break;
                            case 2:
                                density(density_row,col) = std::stod(field);
                                break;
                        }
                        col++;
                    }
                }
                switch(istate){
                    case 0:
                        energy_row++;
                        break;
                    case 1:
                        momentum_row++;
                        break;
                    case 2:
                        density_row++;
                        break;
                }
                row++;
            }
            myfile.close();
        }
    }
    std::array<double,nstate> entropy_var;
    for(int row = 0; row < global_quad_points; row++){
        for(int col = 0; col < num_of_snapshots; col++){
            std::array<double,nstate> conservative_soln{{density(row,col),momentum(row,col),energy(row,col)}};
            entropy_var = euler_physics_double.compute_entropy_variables(conservative_soln);
            snapshotMatrix(row,col) = density(row,col);
            snapshotMatrix(row,col+num_of_snapshots) = momentum(row,col);
            snapshotMatrix(row,col+2*num_of_snapshots) = energy(row,col);
            snapshotMatrix(row,col+3*num_of_snapshots) = entropy_var[0];
            snapshotMatrix(row,col+4*num_of_snapshots) = entropy_var[1];
            snapshotMatrix(row,col+5*num_of_snapshots) = entropy_var[2];
        }
    }
    pcout << "Snapshot matrix generated." << std::endl;
    std::ofstream file("Entropy_snapshot.txt");
    const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");
    if (file.is_open()){
        file << snapshotMatrix.format(CSVFormat);
    }
    file.close();
    return !file_found;
}
template class OfflinePOD <PHILIP_DIM>;

}
}
