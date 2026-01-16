project(
    'micrograd', 
    'cpp',
    default_options : ['cpp_std=c++23'],
    version : '0.0.1'
)

inc = include_directories('include')
lib = library(
    'micrograd',
    sources : [
        'src/value.cpp',
        'src/ops.cpp',
        'src/nn.cpp',
        'src/normalizer.cpp',
        'src/error.cpp'
    ],
    include_directories : inc,
)

executable(
    'micrograd', 
    'multigram.cpp',
    include_directories : inc,
    link_with : lib,
)