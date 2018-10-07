#ifndef TYPE
#define TYPE

/**
 * Device type.
 */
enum DeviceType {
	GROUP_FILTER = 0xffff0000,

	GROUP_CPU = 0x00000000,

	// NOTE(odashi):
	// DeviceType::CPU is deprecated and will be deleted in the next release.
	CPU = 0x00000000,

	NAIVE = 0x00000000,
	EIGEN = 0x00000001,

	GROUP_CUDA = 0x00010000,
	CUDA = 0x00010000,
	CUDA16 = 0x00010001,

	GROUP_OPENCL = 0x00020000,
	OPENCL = 0x00020000,
};

#endif
