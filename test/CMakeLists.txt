include(FetchContent)

FetchContent_Declare(
  ut
  GIT_REPOSITORY https://github.com/boost-ext/ut.git
  GIT_TAG v2.3.1
)
FetchContent_MakeAvailable(ut)

add_subdirectory(unit_test)

