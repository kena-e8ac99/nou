function(nou_test name)
  add_executable(${name} ${name}.cc)
  target_link_libraries(${name} PRIVATE nou PRIVATE Boost::ut)
  add_test(NAME ${name} COMMAND ${name})
endfunction()

nou_test(ut_make_network)

