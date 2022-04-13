#include "image_warp.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT_TENSOR(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor launch_image_warp(const torch::Tensor &image,
                         const torch::Tensor &motion_vector) {
    CHECK_INPUT_TENSOR(image)
    CHECK_INPUT_TENSOR(motion_vector)
    torch::Tensor output = torch::zeros_like(image);
    const int image_size[] = {image.size(0), image.size(1), image.size(2), image.size(3)};
    ImageWarpKernelLauncher(
        (const float*)image.data_ptr(),
        (const float*)motion_vector.data_ptr(),
        image_size,
        (float*)output.data_ptr()
    );
    return output;
}

torch::Tensor launch_image_warp_accum(const torch::Tensor &image,
                         const torch::Tensor &image_dst,
                         const torch::Tensor &motion_vector,
                         const float alpha) {
    CHECK_INPUT_TENSOR(image)
    CHECK_INPUT_TENSOR(image_dst)
    CHECK_INPUT_TENSOR(motion_vector)
    torch::Tensor output(image_dst);
    const int image_size[] = {image.size(0), image.size(1), image.size(2), image.size(3)};
    ImageWarpAccumKernelLauncher(
        (const float*)image.data_ptr(),
        (const float*)motion_vector.data_ptr(),
        image_size,
        alpha,
        (float*)output.data_ptr()
    );
    return output;
}


torch::Tensor launch_image_warp_grad(const torch::Tensor &image,
                         const torch::Tensor &motion_vector,
                         const torch::Tensor &backprop) {
    CHECK_INPUT_TENSOR(image)
    CHECK_INPUT_TENSOR(motion_vector)
    CHECK_INPUT_TENSOR(backprop)
    torch::Tensor output = torch::zeros_like(image);
    const int image_size[] = {image.size(0), image.size(1), image.size(2), image.size(3)};
    ImageWarpGradKernelLauncher(
        (const float*)image.data_ptr(),
        (const float*)motion_vector.data_ptr(),
        (const float*)backprop.data_ptr(),
        image_size,
        (float*)output.data_ptr()
    );
    return output;
}

std::vector<torch::Tensor> launch_image_warp_accum_grad(const torch::Tensor &image,      
                         const torch::Tensor &image_dst,
                         const torch::Tensor &motion_vector,
                         const float alpha,
                         const torch::Tensor &backprop) {
    CHECK_INPUT_TENSOR(image)
    CHECK_INPUT_TENSOR(image_dst)
    CHECK_INPUT_TENSOR(motion_vector)
    CHECK_INPUT_TENSOR(backprop)
    torch::Tensor output1(image);
    torch::Tensor output2(image_dst);
    const int image_size[] = {image.size(0), image.size(1), image.size(2), image.size(3)};
    ImageWarpAccumGradKernelLauncher(
        (const float*)image.data_ptr(),
        (const float*)motion_vector.data_ptr(),
        (const float*)backprop.data_ptr(),
        image_size,
        alpha,
        (float*)output1.data_ptr(),
        (float*)output2.data_ptr()
    );
    return {output1, output2};
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("warp_forward",
          &launch_image_warp,
          "warp an image (forward)");
    m.def("warp_accum_forward",
          &launch_image_warp_accum,
          "warp and accumulate images (forward)");
    m.def("warp_backward",
          &launch_image_warp_grad,
          "warp an image (backward)");
    m.def("warp_accum_backward",
          &launch_image_warp_accum_grad,
          "warp and accumulate images (backward)");
}