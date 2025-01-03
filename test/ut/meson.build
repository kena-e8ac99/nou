unit_tests = [
  'network',
  'is_connectable',
  'make_network',
]

nomalloc = environment({'MALLOC_PERTURB_': '0'})

foreach name: unit_tests
  test(
    'ut_' + name,
    executable(
      'ut_' + name,
      name + '.cc',
      include_directories: inc_dirs,
      dependencies: [test_dep],
    ),
    env: nomalloc,
  )
endforeach

