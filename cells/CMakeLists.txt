set (SRCS 
  module.cpp
  descriptor_matcher_knn.cpp
  matches_filter.cpp
  )

ectomodule(ecto_example ${SRCS})

set_source_files_properties(${SRCS}
  PROPERTIES
  COMPILE_FLAGS "-Wno-deprecated -Wno-strict-aliasing -Wno-unknown-pragmas -Wno-error"
  )
  

link_ecto(ecto_example ${OpenCV_LIBS})
install_ecto_module(ecto_example)

