if(DEVICE_TYPE MATCHES CUDA)
	add_subdirectory(cuda)
endif()

if(DEVICE_TYPE MATCHES EIGEN)
	add_subdirectory(eigen3)
endif()

