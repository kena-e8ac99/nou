unit_tests = [
  'network',
  'is_connectable',
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

