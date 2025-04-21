#include <vector>
#include <algorithm>
#include <cmath>

#include "Epetra_MpiComm.h"
#include "Epetra_Map.h"
#include "Epetra_CrsMatrix.h"

#include <deal.II/base/parameter_handler.h>

#include "parameters/all_parameters.h"
#include "parameters/parameters.h"
#include "operators/operators.h"

double f() {
  static int i;
  return ++i;
}

int main (int argc, char *argv[]){
  const int dim = PHILIP_DIM;
  const int size = 5;
  const double zero = 0.0;
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  std::srand(0);

  std::vector<std::vector<double> > matrix_vectors(size);

  for(int i = 0; i < size; i++) {
    matrix_vectors[i].resize(size);
    std::generate(matrix_vectors[i].begin(), matrix_vectors[i].end(), f);
  }

  Epetra_MpiComm comm(MPI_COMM_WORLD);
  Epetra_Map map(size,0,comm);
  Epetra_CrsMatrix A(Epetra_DataAccess::Copy,map,map,size);
  Epetra_CrsMatrix B(Epetra_DataAccess::Copy,map,map,size);
  Epetra_CrsMatrix C(Epetra_DataAccess::Copy,map,map,size);
  for(int row = 0; row < size; row++) {
    for(int col = 0; col < size; col++) {
      if(row == 2 && col == 3) {
        A.InsertGlobalValues(row,1,&zero,&col);
        B.InsertGlobalValues(row,1,&matrix_vectors[row][col],&col);
      } else if (row == 3 && col == 1) {
        A.InsertGlobalValues(row,1,&zero,&col);
        B.InsertGlobalValues(row,1,&zero,&col);
      } else if (row == 1 && col == 4) {
        A.InsertGlobalValues(row,1,&matrix_vectors[row][col],&col);
        B.InsertGlobalValues(row,1,&zero,&col);
      } else {
        A.InsertGlobalValues(row,1,&matrix_vectors[row][col],&col);
        B.InsertGlobalValues(row,1,&matrix_vectors[row][col],&col);
      }
    }
  }
  A.FillComplete(map,map);
  B.FillComplete(map,map);
  PHiLiP::OPERATOR::basis_functions<dim,2*dim,double> basis(dim, 3, 1);
  basis.Hadamard_product(A,B,C);
  std::ofstream c_file("C.txt");
  C.Print(c_file);
  std::vector <int> c_indices(size);
  std::vector <double> c_row(size);
  int NumEntries = 0;
  int opt = 0;
  for(int row = 0; row < size; row++) {
    C.ExtractGlobalRowCopy(row,size,NumEntries,c_row.data(),c_indices.data());
    for(int entry = 0; entry < NumEntries; entry++) {
      if(row == 2 && c_indices[entry] == 3) {
        if(c_row[entry] != 0) opt += 1;
      } else if (row == 3 && c_indices[entry] == 1) {
        if(c_row[entry] != 0) opt += 1;
      } else if (row == 1 && c_indices[entry] == 4) {
        if(c_row[entry] != 0) opt += 1;
      } else {
        if(c_row[entry] != pow(matrix_vectors[row][c_indices[entry]],2)) opt += 1;
      }
    }
  }
  return opt;
}