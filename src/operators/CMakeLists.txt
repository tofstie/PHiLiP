SET(OPERSOURCE
    operators.cpp
#    operators_factory.cpp
    )

foreach(dim RANGE 1 3)
    # Output library
    string(CONCAT OperatorsLib Operator_Lib_${dim}D)
    add_library(${OperatorsLib} STATIC ${OPERSOURCE})

    target_compile_definitions(${OperatorsLib} PRIVATE PHILIP_DIM=${dim})
    # Library dependency
    string(CONCAT ParameterLib ParametersLibrary)
    target_link_libraries(${ParameterLib})

    # Setup target with deal.II
    if(NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${OperatorsLib})
    endif()

    unset(OperatorsLib)

endforeach()
