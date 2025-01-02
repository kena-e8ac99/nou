unit_tests = [
  'network',
  'is_connectable',
  'make_network',
]

foreach name: unit_tests
  test(
    'ut_' + name,
    executable(
      'ut_' + name,
      name + '.cc',
      include_directories: inc_dirs,
      dependencies: [test_dep],
    )
  )
endforeach

