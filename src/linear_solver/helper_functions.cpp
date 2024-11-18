#include "helper_functions.h"
#include <iostream>
#include <fstream>
#include <eigen/Eigen/QR>
#include <iostream>
#include <vector>
#include <Epetra_MpiComm.h>
#include <Epetra_ConfigDefs.h>
#include <Epetra_Map.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_Vector.h>
#include <Epetra_MultiVector.h>
#include <Epetra_LinearProblem.h>
#include <EpetraExt_MatrixMatrix.h>
#include <deal.II/lac/lapack_full_matrix.h>

using namespace Eigen;
/**  @brief Fills the entries in an empty Eigen::MatrixXd from an Epetra_Vector structure
*    @param col length of the vector
*    @param x Full Epetra vector to copy
*    @param x_eig Empty Eigen::MatrixXd
*/
void epetra_to_eig_vec(int col, Epetra_Vector &x, Eigen::MatrixXd &x_eig){
  // Gather local information
  int local_size = x.MyLength();
  const Epetra_Comm& comm = x.Comm();
  int np = comm.NumProc();

  std::vector<double> x_values(col);
  std::vector<int> local_sizes(np);
  std::vector<int> displacements(np, 0);
  // Gather all local_size into local_sizes. Vector will be the same globally
  MPI_Allgather(&local_size, 1, MPI_INT,
                local_sizes.data(), 1, MPI_INT,
                MPI_COMM_WORLD);
  // Calculate the global coordinates necessary
  for (int i = 1; i < np; ++i){
    displacements[i] = displacements[i-1] + local_sizes[i-1];
  }
  // Store the Epetra_vector values into x_values
  MPI_Allgatherv(x.Values(), local_size, MPI_DOUBLE,
                x_values.data(), local_sizes.data(), displacements.data(), MPI_DOUBLE,
                MPI_COMM_WORLD);
  // Convert epetra vector to eigen vector
  for(int i = 0; i < col; i++){
    x_eig(i,0) = x_values[i];
  }
}

/**  @brief Returns an Epetra_CrsMatrix with the entries from an Eigen::MatrixXd structure
*    @param A_eig Full Eigen Matrix to copy
*    @param col number of columns
*    @param row number of rows
*    @param Comm MpiComm for Epetra Maps
*    @return Full Epetra_CrsMatrix
*/
Epetra_CrsMatrix eig_to_epetra_matrix(Eigen::MatrixXd &A_eig, int col, int row, Epetra_MpiComm &Comm){
  // Create an empty Epetra structure with the right dimensions
  Epetra_Map RowMap(row,0,Comm);
  Epetra_Map ColMap(col,0,Comm);
  Epetra_CrsMatrix A(Epetra_DataAccess::Copy, RowMap, col);
  const int numMyElements = RowMap.NumMyElements();

  // Fill the Epetra_CrsMatrix from the Eigen::MatrixXd
  for (int localRow = 0; localRow < numMyElements; ++localRow){
      const int globalRow = RowMap.GID(localRow);
      for(int n = 0 ; n < A_eig.cols() ; n++){
          A.InsertGlobalValues(globalRow, 1, &A_eig(globalRow, n), &n);
      }
  }
  A.FillComplete(ColMap, RowMap);
  return A;
}

Epetra_CrsMatrix eig_to_epetra_matrix(Eigen::MatrixXd &A_eig, Epetra_Map ColMap, Epetra_Map RowMap){
  // Create an empty Epetra structure with the right dimensions
  Epetra_CrsMatrix A(Epetra_DataAccess::Copy, RowMap, ColMap.NumGlobalElements());
  const int numMyElements = RowMap.NumMyElements();

  // Fill the Epetra_CrsMatrix from the Eigen::MatrixXd
  for (int localRow = 0; localRow < numMyElements; ++localRow){
      const int globalRow = RowMap.GID(localRow);
      for(int n = 0 ; n < A_eig.cols() ; n++){
          A.InsertGlobalValues(globalRow, 1, &A_eig(globalRow, n), &n);
      }
  }
  A.FillComplete(ColMap, RowMap);
  return A;
}

MatrixXd epetra_to_eig_matrix(Epetra_CrsMatrix A_epetra){
  // Create an empty Eigen structure
  MatrixXd A(A_epetra.NumGlobalRows(), A_epetra.NumGlobalCols());
  //int rank = A_epetra.Comm().MyPID();
  //std::ofstream sum_file("Input_"+std::to_string(rank)+".txt");
  //A_epetra.Print(sum_file);
  // Fill the Eigen::MatrixXd from the Epetra_CrsMatrix
  for (int m = 0; m < A_epetra.NumGlobalRows(); m++) {

    /*int count = A_epetra.NumMyCols();
    double *local_row = new double [count];
    double *global_row = new double [A_epetra.NumGlobalCols()];
    if(A_epetra.MyGRID(m)) {
      local_row = A_epetra[m];
    }
    std::cout << local_row << std::endl;
    A_epetra.Comm().GatherAll(local_row,global_row,count);
    for (int n = 0; n < A_epetra.NumGlobalCols(); n++) {
      A(m,n) = global_row[n];
      std::cout << m << "X" << n << std::endl;
    }
    */
    double *global_row = new double [A_epetra.NumGlobalCols()];
    int *indicies = new int [A_epetra.NumGlobalCols()];
    int num_entries = 0;

    const int *GIDList = &m;
    int *PIDList = new int[1];
    int *LIDList = new int[1];
    
    A_epetra.RowMap().RemoteIDList(1, GIDList, PIDList, LIDList);
    A_epetra.ExtractGlobalRowCopy(m,A_epetra.NumGlobalCols(),num_entries,global_row,indicies);
    if(PIDList[0] != 0){
      //std::cout << "Break here" << std::endl;
    }
    A_epetra.Comm().Broadcast(global_row,A_epetra.NumGlobalCols(),PIDList[0]);
    A_epetra.Comm().Broadcast(indicies,A_epetra.NumGlobalCols(),PIDList[0]);
    for (int n = 0; n < A_epetra.NumGlobalCols(); n++) {
      A(m,indicies[n]) = global_row[n];
    }
    delete [] global_row;
    delete [] indicies;
    delete [] PIDList;
    delete [] LIDList;
  }
  return A;
}

/** @brief Returns an Epetra_Vector with entries from an Eigen::Vector structure
*   @param a_eigen Eigen Vector to copy
*   @param size size of vector
*   @param MpiComm for Epetra Maps
*   @return Epetra_Vector
*/
Epetra_Vector eig_to_epetra_vector(Eigen::VectorXd &a_eigen, int size, Epetra_MpiComm &Comm){
  // Create an Epetra Vector distributed along all cores in Comm
  Epetra_Map vecMap(size,0,Comm);
  Epetra_Vector a_epetra(vecMap);
  // Fill the Epetra_Vector with values from the Eigen Vector
  const int numMyElements = vecMap.NumMyElements();
  for (int localElement = 0; localElement < numMyElements; localElement++){
    const int globalElement = vecMap.GID(localElement);
    a_epetra.ReplaceGlobalValues(1, &a_eigen(globalElement), &globalElement);
  }
  return a_epetra;
}

MatrixXd lapack_to_eig_matrix(dealii::LAPACKFullMatrix<double> &lapack_matrix){
  const unsigned int rows = lapack_matrix.m();
  const unsigned int cols = lapack_matrix.n();
  Eigen::MatrixXd eigen_matrix(rows,cols);
  for (unsigned int i = 0; i < rows; ++i){
    for (unsigned int  j = 0; j < cols; ++j){
      eigen_matrix(i,j) = lapack_matrix(i,j);
    }
  }
  return eigen_matrix;
}

dealii::LAPACKFullMatrix<double> eig_to_lapack_matrix(MatrixXd &eigen_matrix){
  /// Assuming that LAPACK is the same along all cores
  const unsigned int rows = eigen_matrix.rows();
  const unsigned int cols = eigen_matrix.cols();
  dealii::LAPACKFullMatrix<double> lapack_matrix(rows,cols);
  for (unsigned int i = 0; i < rows; ++i){
    for (unsigned int  j = 0; j < cols; ++j){
      lapack_matrix(i,j) = eigen_matrix(i,j);
    }
  }
  return lapack_matrix;

}