#include </usr/local/include/onnxruntime/include/onnxruntime_cxx_api.h>
#include <iostream>

int main() {
    std::cout << "ONNX Runtime Version: " << ORT_API_VERSION << std::endl;
    return 0;
}