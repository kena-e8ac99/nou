unit_tests = [
  'network'
]

foreach name: unit_tests
  test(
    'ut_' + name,
    executable(
      'ut_' + name,
      name + '.cc',
      include_directories: inc_dirs
    )
  )
endforeach

