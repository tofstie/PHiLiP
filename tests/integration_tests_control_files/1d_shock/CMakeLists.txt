set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

configure_file(1d_shock.prm 1d_shock.prm COPYONLY)
add_test(
  NAME 1D_SHOCKED_MANUFACTURED_SOLUTION
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_shock.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(1D_SHOCKED_MANUFACTURED_SOLUTION   1D_SHOCK
                                                    1D
                                                    SERIAL
                                                    BURGERS_INVISCID
                                                    RUNGE-KUTTA
                                                    WEAK
                                                    UNCOLLOCATED
                                                    MANUFACTURED_SOLUTION
                                                    MODERATE
                                                    INTEGRATION_TEST)