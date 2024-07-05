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
    pcout << "Assembling_residual📢" << std::endl;
    const bool compute_dRdW = false;
    dg->evaluate_mass_matrices(compute_dRdW);
    pcout << "Searching files..." << std::endl;
    if(dg->all_parameters->reduced_order_param.entropy_varibles_in_snapshots){
        getEntropyPODBasisFromSnapshots();
        //getEntropyProjPODBasisFromSnapshots();
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
void OfflinePOD<dim>::calculatePODBasis(MatrixXd snapshots, std::string reference_type) {
/* Reference for simple POD basis computation: Refer to Algorithm 1 in the following reference:
    "Efficient non-linear model reduction via a least-squares Petrov–Galerkin projection and compressive tensor approximations"
    Kevin Carlberg, Charbel Bou-Mosleh, Charbel Farhat
    International Journal for Numerical Methods in Engineering, 2011
    */
    int num_of_modes = dg->all_parameters->reduced_order_param.number_nodal_modes;
    VectorXd reference_state;
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
    MatrixXd snapshotMatrixCentered = snapshots.colwise() - reference_state;
    Eigen::BDCSVD<MatrixXd, Eigen::DecompositionOptions::ComputeThinU> svd_one(snapshotMatrixCentered);
    MatrixXd pod_basis_one = svd_one.matrixU();
    /// This commented sections adds a col of 1 to the LSV and preforms another SVD.
    
    VectorXd ones = VectorXd::Ones(pod_basis_one.rows());
    pod_basis_one.conservativeResize(pod_basis_one.rows(),pod_basis_one.cols()+1);
    pod_basis_one.col(pod_basis_one.cols()-1) = ones;
    Eigen::BDCSVD<MatrixXd, Eigen::DecompositionOptions::ComputeThinU> svd(pod_basis_one);
    MatrixXd pod_basis = svd.matrixU();
    VectorXd singular_values = svd.singularValues();
    std::ofstream sing_file("singular_values.txt");
    sing_file << singular_values;
    //MatrixXd pod_basis = pod_basis_one; // Comment this line out when wanting to use 1's
    fullBasis.reinit(pod_basis.rows(), pod_basis.cols());

    for (unsigned int m = 0; m < pod_basis.rows(); m++) {
        for (unsigned int n = 0; n < pod_basis.cols(); n++) {
            fullBasis.set(m, n, pod_basis(m, n));
        }
    }

    std::ofstream out_file("POD_basis.txt");
    unsigned int precision = 16;
    fullBasis.print_formatted(out_file, precision);
    if (!num_of_modes == 0){
        Assert(num_of_modes > pod_basis.cols(),
         dealii::ExcMessage("The number of modes selected must be less than the number of snapshots"));
        // 📢 MatrixXd pod_basis_n_modes = Eigen::MatrixXd
    }
    const Epetra_CrsMatrix epetra_system_matrix  = this->dg->global_mass_matrix.trilinos_matrix();
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
    std::string reference_type = "mean";
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
            global_quad_points = rows/nstate;
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
    int solution_idx = 0;
    for(int row = 0; row < global_quad_points; row++){
        for(int col = 0; col < num_of_snapshots; col++){
            std::array<double,nstate> conservative_soln;
            for(int istate = 0; istate < nstate ; istate++){
                const int conservative_density_case = 0;
                const int conservative_energy_case = nstate-1;
                switch(istate){
                    case conservative_density_case:
                        conservative_soln[istate] = density(row,col);
                        break;
                    case conservative_energy_case:
                        conservative_soln[istate] = energy(row,col);
                        break;
                    default:
                        conservative_soln[nstate-istate-1] = momentum[istate-1](row,col);
                        break;
                }
            }
            std::array<double,nstate> entropy_var = euler_physics_double.compute_entropy_variables(conservative_soln);
            for(int istate = 0; istate < nstate; istate++){
                switch(istate){
                    case energy_case:
                        snapshotMatrix(solution_idx+istate*n_quad_pts,col) = energy(row,col);
                        
                        break;
                    case density_case:
                        snapshotMatrix(solution_idx+istate*n_quad_pts,col) = density(row,col);
                        break;
                    default:
                        snapshotMatrix(solution_idx+istate*n_quad_pts,col) = momentum[istate-1](row,col);
                        break;


                }
                snapshotMatrix(solution_idx+istate*n_quad_pts,col+num_of_snapshots) = entropy_var[istate];//entropy_var[nstate-istate-1];
                /* OLD
                snapshotMatrix(solution_idx+2*n_quad_pts,col) = density(row,col);
                snapshotMatrix(solution_idx+1*n_quad_pts,col) = momentum(row,col);
                snapshotMatrix(solution_idx+0*n_quad_pts,col) = energy(row,col);
                snapshotMatrix(solution_idx+2*n_quad_pts,col+1*num_of_snapshots) = entropy_var[0];
                snapshotMatrix(solution_idx+1*n_quad_pts,col+1*num_of_snapshots) = entropy_var[1];
                snapshotMatrix(solution_idx+0*n_quad_pts,col+1*num_of_snapshots) = entropy_var[2];
                */
            }
        }
        solution_idx++;
        if(solution_idx % n_quad_pts == 0){
            solution_idx += (nstate-1)*n_quad_pts;
        }
    }
    pcout << "Snapshot matrix generated." << std::endl;
    calculatePODBasis(snapshotMatrix, reference_type);
    //enrichPOD();
    std::ofstream file("Entropy_snapshot.txt");
    const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");
    if (file.is_open()){
        file << snapshotMatrix.format(CSVFormat);
    }
    file.close();
    return !file_found;
}
template<int dim>
bool OfflinePOD<dim>::getEntropyProjPODBasisFromSnapshots(){
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
            global_quad_points = rows/nstate;
            snapshotMatrix.conservativeResize(rows, old_amount_of_snapshots + cols); // Changing rows from rows/nstate and cols from 2*nstate*cols
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
    int solution_idx = 0;
    for(int row = 0; row < global_quad_points; row++){
        for(int col = 0; col < num_of_snapshots; col++){
            std::array<double,nstate> conservative_soln;
            for(int istate = 0; istate < nstate ; istate++){
                const int conservative_density_case = 0;
                const int conservative_energy_case = nstate-1;
                switch(istate){
                    case conservative_density_case:
                        conservative_soln[istate] = density(row,col);
                        break;
                    case conservative_energy_case:
                        conservative_soln[istate] = energy(row,col);
                        break;
                    default:
                        conservative_soln[nstate-istate-1] = momentum[istate-1](row,col);
                        break;
                }
            }
            std::array<double,nstate> entropy_var = euler_physics_double.compute_entropy_variables(conservative_soln);
            std::array<double,nstate> proj_entropy_conserv_var = euler_physics_double.compute_conservative_variables_from_entropy_variables(entropy_var);
            for(int istate = 0; istate < nstate; istate++){
                switch(istate){
                    case energy_case:
                        snapshotMatrix(solution_idx+istate*n_quad_pts,col) = proj_entropy_conserv_var[nstate-1];
                        break;
                    case density_case:
                        snapshotMatrix(solution_idx+istate*n_quad_pts,col) = proj_entropy_conserv_var[0];
                        break;
                    default:
                        snapshotMatrix(solution_idx+istate*n_quad_pts,col) = proj_entropy_conserv_var[nstate-1-istate];
                        break;
                }
                
                /* OLD
                snapshotMatrix(solution_idx+2*n_quad_pts,col) = density(row,col);
                snapshotMatrix(solution_idx+1*n_quad_pts,col) = momentum(row,col);
                snapshotMatrix(solution_idx+0*n_quad_pts,col) = energy(row,col);
                snapshotMatrix(solution_idx+2*n_quad_pts,col+1*num_of_snapshots) = entropy_var[0];
                snapshotMatrix(solution_idx+1*n_quad_pts,col+1*num_of_snapshots) = entropy_var[1];
                snapshotMatrix(solution_idx+0*n_quad_pts,col+1*num_of_snapshots) = entropy_var[2];
                */
            }
        }
        solution_idx++;
        if(solution_idx % n_quad_pts == 0){
            solution_idx += (nstate-1)*n_quad_pts;
        }
    }
    pcout << "Snapshot matrix generated." << std::endl;
    calculatePODBasis(snapshotMatrix, reference_type);
    //enrichPOD();
    std::ofstream file("Entropy_proj_snapshot.txt");
    const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");
    if (file.is_open()){
        file << snapshotMatrix.format(CSVFormat);
    }
    file.close();
    return !file_found;
}

template <int dim>
bool OfflinePOD<dim>::enrichPOD(){
    dg->evaluate_mass_matrices(false);
    //int nstate = dim+2;
    int const poly_degree = dg->all_parameters->flow_solver_param.poly_degree;
    const unsigned int n_quad_pts  = dg->volume_quadrature_collection[poly_degree].size();
    const unsigned int n_quad_pts_1D  = dg->oneD_quadrature_collection[poly_degree].size();
    const unsigned int n_face_quad_pts  = dg->face_quadrature_collection[poly_degree].size();
    const std::vector<double> &oneD_vol_quad_weights = dg->oneD_quadrature_collection[poly_degree].get_weights();
    const std::vector<double> &oneD_quad_weights_vol= dg->oneD_quadrature_collection[poly_degree].get_weights();
    const unsigned int init_grid_degree = dg->high_order_grid->fe_system.tensor_degree();
    std::vector<dealii::types::global_dof_index> dofs_indices;
    dealii::TrilinosWrappers::SparseMatrix global_Q;
    global_Q.reinit(dg->global_mass_matrix);
    // Basis Def
    OPERATOR::local_basis_stiffness<dim,2*dim,double> flux_basis_stiffness(1, dg->max_degree, init_grid_degree, true); 
    OPERATOR::basis_functions<dim,2*dim,double> flux_basis_int(1, dg->max_degree, init_grid_degree); 
    OPERATOR::basis_functions<dim,2*dim,double> flux_basis_ext(1, dg->max_degree, init_grid_degree); 

    // 📢 Define Flux Basis 📢
    std::vector<std::array<unsigned int,dim>> Hadamard_rows_sparsity_volume(n_quad_pts * n_quad_pts_1D);//size n^{d+1}
    std::vector<std::array<unsigned int,dim>> Hadamard_columns_sparsity_volume(n_quad_pts * n_quad_pts_1D);
    flux_basis_int.sum_factorized_Hadamard_sparsity_pattern(n_quad_pts_1D, n_quad_pts_1D, Hadamard_rows_sparsity_volume, Hadamard_columns_sparsity_volume);
    // TODO : Read in POD Basis
    for (auto cell = dg->dof_handler.begin_active(); cell!=dg->dof_handler.end(); ++cell){
        if (!cell->is_locally_owned()) continue;
        // Reinit Operators for local cell
        flux_basis_int.build_1D_volume_operator(dg->oneD_fe_collection_flux[dg->max_degree], dg->oneD_quadrature_collection[dg->max_degree]);
        flux_basis_int.build_1D_gradient_operator(dg->oneD_fe_collection_flux[dg->max_degree], dg->oneD_quadrature_collection[dg->max_degree]);
        flux_basis_int.build_1D_surface_operator(dg->oneD_fe_collection_flux[dg->max_degree], dg->oneD_face_quadrature);
        flux_basis_int.build_1D_surface_gradient_operator(dg->oneD_fe_collection_flux[dg->max_degree], dg->oneD_face_quadrature);

        flux_basis_ext.build_1D_volume_operator(dg->oneD_fe_collection_flux[dg->max_degree], dg->oneD_quadrature_collection[dg->max_degree]);
        flux_basis_ext.build_1D_gradient_operator(dg->oneD_fe_collection_flux[dg->max_degree], dg->oneD_quadrature_collection[dg->max_degree]);
        flux_basis_ext.build_1D_surface_operator(dg->oneD_fe_collection_flux[dg->max_degree], dg->oneD_face_quadrature);
        flux_basis_ext.build_1D_surface_gradient_operator(dg->oneD_fe_collection_flux[dg->max_degree], dg->oneD_face_quadrature);

        flux_basis_stiffness.build_1D_volume_operator(dg->oneD_fe_collection_flux[dg->max_degree], dg->oneD_quadrature_collection[dg->max_degree]);

        const unsigned int fe_index_curr_cell = cell->active_fe_index();
        // Current reference element related to this physical cell
        const dealii::FESystem<dim,dim> &current_fe_ref = dg->fe_collection[fe_index_curr_cell];
        const unsigned int n_dofs_cell = current_fe_ref.n_dofs_per_cell();
        
        dofs_indices.resize(n_dofs_cell);
        cell->get_dof_indices (dofs_indices);
        //const bool Cartesian_element = (cell->manifold_id() == dealii::numbers::flat_manifold_id);
      
        
        // Compute Q
        dealii::FullMatrix<double> local_Q(n_dofs_cell);
        // Top Left - Strong DG line 1213
        std::array<dealii::FullMatrix<double>,dim> flux_basis_stiffness_skew_symm_oper_sparse;
        for(int idim=0; idim<dim; idim++){
            flux_basis_stiffness_skew_symm_oper_sparse[idim].reinit(n_quad_pts, n_quad_pts_1D);
        }
        // 📢 Define flux_basis_stiffness 📢
        flux_basis_int.sum_factorized_Hadamard_basis_assembly(n_quad_pts_1D, n_quad_pts_1D, 
                                                            Hadamard_rows_sparsity_volume, Hadamard_columns_sparsity_volume,
                                                            flux_basis_stiffness.oneD_skew_symm_vol_oper, 
                                                            oneD_vol_quad_weights,
                                                            flux_basis_stiffness_skew_symm_oper_sparse);
        local_Q.fill(flux_basis_stiffness_skew_symm_oper_sparse[0],0,0,0,0);
        // Both Off diagonal terms - Strong DG line 1641
        
        dealii::FullMatrix<double> surf_oper_sparse(n_face_quad_pts, n_quad_pts_1D);
        for (unsigned int iface=0; iface < dealii::GeometryInfo<dim>::faces_per_cell; ++iface) {
            const int iface_1D = iface % 2;//the reference face number
            const int dim_not_zero = iface / 2;//reference direction of face integer division
            std::vector<unsigned int> Hadamard_rows_sparsity_off(n_face_quad_pts * n_quad_pts_1D);//size n^{d+1}
            std::vector<unsigned int> Hadamard_columns_sparsity_off(n_face_quad_pts * n_quad_pts_1D);
            flux_basis_int.sum_factorized_Hadamard_surface_sparsity_pattern(n_quad_pts_1D, n_quad_pts_1D, Hadamard_rows_sparsity_off, Hadamard_columns_sparsity_off, dim_not_zero);
            flux_basis_int.sum_factorized_Hadamard_surface_basis_assembly(n_face_quad_pts, n_quad_pts_1D, 
                                                                        Hadamard_rows_sparsity_off, Hadamard_columns_sparsity_off,
                                                                        flux_basis_int.oneD_surf_operator[iface_1D], 
                                                                        oneD_quad_weights_vol,
                                                                        surf_oper_sparse,
                                                                        dim_not_zero);
            //debugMatrix(surf_oper_sparse);
            local_Q.fill(surf_oper_sparse,n_quad_pts_1D*(iface+1),0,0,0);
            dealii::FullMatrix<double> surf_oper_sparse_trans;
            surf_oper_sparse_trans.copy_transposed(surf_oper_sparse);
            surf_oper_sparse_trans *= -1;
            local_Q.fill(surf_oper_sparse_trans,0,n_quad_pts_1D*(iface+1),0,0);
        }
        //debugMatrix(flux_basis_stiffness_skew_symm_oper_sparse[0]);
        global_Q.set(dofs_indices, local_Q);
       
    }
    // Compute QV
    
    dealii::TrilinosWrappers::SparseMatrix sparse_basis;
    sparse_basis.copy_from(*basis);
    dealii::TrilinosWrappers::SparseMatrix QV;
    global_Q.mmult(QV,*basis);
    // Inject QV into V
    dealii::FullMatrix<double> enriched_basis(sparse_basis.m(),sparse_basis.n()+global_Q.n()+1);
    for (std::size_t row = 0; row < enriched_basis.m(); ++row){ // Might need to change size_type = std::size_t
        enriched_basis.set(row,0,1.);
        //for (auto entry = full_basis.begin(row); entry != full_basis.end(row);++entry){
        //    enriched_basis.set(row, entry->column()+1) = entry->value();
        //}
    }
    dealii::FullMatrix<double> full_QV;
    dealii::FullMatrix<double> full_basis;
    full_QV.copy_from(QV);
    full_basis.copy_from(sparse_basis);
    enriched_basis.fill(full_basis,0,1);
    enriched_basis.fill(full_QV,0,full_basis.n()+1);
    dealii::LAPACKFullMatrix<double> enriched_basis_LAPACK(n_quad_pts);
    enriched_basis_LAPACK = enriched_basis;
    pcout << "Preforming LAPACK svd" << std::endl;
    enriched_basis_LAPACK.compute_svd();
    dealii::LAPACKFullMatrix<double> LAPACK_test_basis = enriched_basis_LAPACK.get_svd_u();
    const Epetra_CrsMatrix epetra_system_matrix  = this->dg->global_mass_matrix.trilinos_matrix();
    Epetra_Map system_matrix_map = epetra_system_matrix.RowMap();
    Epetra_CrsMatrix epetra_basis(Epetra_DataAccess::Copy, system_matrix_map, LAPACK_test_basis.n_cols());

    const int numMyElements = system_matrix_map.NumMyElements(); //Number of elements on the calling processor

    for (int localRow = 0; localRow < numMyElements; ++localRow){
        const int globalRow = system_matrix_map.GID(localRow);
        for(long unsigned int n = 0 ; n < LAPACK_test_basis.n_cols() ; n++){ // Type of n is long unsigned int
            //epetra_basis.InsertGlobalValues(globalRow, 1, LAPACK_basis(globalRow, n), &n);
            pcout << globalRow << std::endl;
        }
    }
    Epetra_MpiComm epetra_comm(MPI_COMM_WORLD);
    Epetra_Map domain_map((int)LAPACK_test_basis.n_cols(), 0, epetra_comm);

    epetra_basis.FillComplete(domain_map, system_matrix_map);
    basis->reinit(epetra_basis);
    /*
    std::ofstream out_file("enriched_basis.txt");
    unsigned int precision = 16;
    enriched_basis.print_formatted(out_file, precision);
    */
    // Compute Qt hat
    Epetra_CrsMatrix modal_differentiation_matrix_temp(Epetra_DataAccess::View, system_matrix_map, LAPACK_test_basis.n_cols());
    Epetra_CrsMatrix modal_differentiation_matrix(Epetra_DataAccess::View, epetra_basis.DomainMap(), epetra_basis.NumGlobalCols());
    Epetra_CrsMatrix Q(Epetra_DataAccess::Copy,system_matrix_map,global_Q.n());

    EpetraExt::MatrixMatrix::Multiply(Q, false, epetra_basis, false, modal_differentiation_matrix_temp);
    EpetraExt::MatrixMatrix::Multiply(epetra_basis, true, modal_differentiation_matrix_temp, false, modal_differentiation_matrix);
    // Save modal_differentiation_matrix somewhere
    return true;
}

template<int dim>
void OfflinePOD<dim>::debugMatrix(dealii::FullMatrix<double> M){
    for (auto cell = M.begin();cell < M.end();cell++){
        pcout << cell->value() << std::endl;
    }
    return;
};
template class OfflinePOD <PHILIP_DIM>;

}
}
