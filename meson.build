project(
    'micrograd', 
    'cpp',
    default_options : ['cpp_std=c++17'],
    version : '0.0.1'
)

inc = include_directories('include')
lib = library(
    'micrograd',
    sources : [
        'src/value.cpp',
        'src/ops.cpp',
        'src/nn.cpp',
    ],
    include_directories : inc,
)

executable(
    'micrograd', 
    'test.cpp',
    include_directories : inc,
    link_with : lib,
)