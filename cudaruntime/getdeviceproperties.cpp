#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

int main() {
    // 获取设备数量
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    if (deviceCount == 0) {
        std::cout << "No CUDA-capable devices detected." << std::endl;
        return 0;
    }

    std::cout << "Found " << deviceCount << " CUDA-capable device(s):\n" << std::endl;

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);

        std::cout << "Device " << dev << ": " << prop.name << std::endl;
        std::cout << std::string(50, '-') << std::endl;
        std::cout << std::left << std::setw(30) << "Compute Capability:"         << prop.major << "." << prop.minor << std::endl;
        std::cout << std::left << std::setw(30) << "Global Memory:"              << (size_t)(prop.totalGlobalMem / (1024 * 1024)) << " MB" << std::endl;
        std::cout << std::left << std::setw(30) << "Shared Memory per Block:"    << prop.sharedMemPerBlock << " bytes" << std::endl;
        std::cout << std::left << std::setw(30) << "Registers per Block:"        << prop.regsPerBlock << std::endl;
        std::cout << std::left << std::setw(30) << "Warp Size:"                  << prop.warpSize << " threads" << std::endl;
        std::cout << std::left << std::setw(30) << "Max Threads per Block:"      << prop.maxThreadsPerBlock << std::endl;
        std::cout << std::left << std::setw(30) << "Max Block Dimensions:"       << "[" << prop.maxThreadsDim[0] 
                                                                                   << ", " << prop.maxThreadsDim[1] 
                                                                                   << ", " << prop.maxThreadsDim[2] << "]" << std::endl;
        std::cout << std::left << std::setw(30) << "Max Grid Dimensions:"        << "[" << prop.maxGridSize[0]
                                                                                   << ", " << prop.maxGridSize[1]
                                                                                   << ", " << prop.maxGridSize[2] << "]" << std::endl;
        std::cout << std::left << std::setw(30) << "Multiprocessor Count:"       << prop.multiProcessorCount << std::endl;
        std::cout << std::left << std::setw(30) << "Clock Rate:"                 << prop.clockRate / 1000.0 << " MHz" << std::endl;
        std::cout << std::left << std::setw(30) << "Memory Clock Rate:"          << prop.memoryClockRate / 1000.0 << " MHz" << std::endl;
        std::cout << std::left << std::setw(30) << "Memory Bus Width:"           << prop.memoryBusWidth << " bits" << std::endl;
        std::cout << std::left << std::setw(30) << "L2 Cache Size:"              << prop.l2CacheSize / 1024 << " KB" << std::endl;
        std::cout << std::left << std::setw(30) << "Concurrent Kernels:"         << (prop.concurrentKernels ? "Yes" : "No") << std::endl;
        std::cout << std::left << std::setw(30) << "Unified Addressing:"         << (prop.unifiedAddressing ? "Yes" : "No") << std::endl;
        std::cout << std::left << std::setw(30) << "PCIe Domain ID:"             << prop.pciDomainID << std::endl;
        std::cout << std::left << std::setw(30) << "PCIe Bus ID:"                << prop.pciBusID << std::endl;
        std::cout << std::left << std::setw(30) << "PCIe Device ID:"             << prop.pciDeviceID << std::endl;
        std::cout << std::endl;
    }

    return 0;
}