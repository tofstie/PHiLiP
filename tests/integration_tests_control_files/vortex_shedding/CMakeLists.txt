set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
# =======================================
# Vortex-Shedding -- 2D 
# =======================================
# ----------------------------------------
# - details: [if needed]
# ----------------------------------------
configure_file(vortex_shedding.prm vortex_shedding.prm COPYONLY)
add_test(
    NAME MPI_VORTEX_SHEDDING
    COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/vortex_shedding.prm
    WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
