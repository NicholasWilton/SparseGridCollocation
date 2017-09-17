#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <cmath>

#define API _declspec(dllexport)


using namespace std;

namespace Leicester
{

	namespace ThrustLib
	{
		struct API MemoryInfo
		{
			int total;
			int free;
		};

		class API Common
		{
		public:
			static MemoryInfo GetMemory()
			{
				size_t free_bytes;

				size_t total_bytes;

				cudaError_t e = cudaMemGetInfo(&free_bytes, &total_bytes);

				MemoryInfo res;
				res.free = free_bytes;
				res.total = total_bytes;

				return res;
			}
		};
	}
}
