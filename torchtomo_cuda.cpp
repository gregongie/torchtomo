#include <torch/extension.h>

torch::Tensor circularFanbeamProjection_cuda(const torch::Tensor image, const int nx, const int ny,
                              const float ximageside, const float yimageside,
                              const float radius, const float source_to_detector,
                              const int nviews, const float slen, const int nbins);

torch::Tensor circularFanbeamBackProjection_cuda(const torch::Tensor sinogram, const int nx, const int ny,
                              const float ximageside, const float yimageside,
                              const float radius, const float source_to_detector,
                              const int nviews, const float slen, const int nbins);

torch::Tensor circularFanbeamWPDProjection_cuda(const torch::Tensor sinogram, const int nx, const int ny,
                              const float ximageside, const float yimageside,
                              const float radius, const float source_to_detector,
                              const int nviews, const float slen, const int nbins);

torch::Tensor circularFanbeamWPDBackProjection_cuda(const torch::Tensor sinogram, const int nx, const int ny,
                              const float ximageside, const float yimageside,
                              const float radius, const float source_to_detector,
                              const int nviews, const float slen, const int nbins);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor circularFanbeamProjection(const torch::Tensor image, const int nx, const int ny, const float ximageside, const float yimageside,
                              const float radius, const float source_to_detector,
                              const int nviews, const float slen, const int nbins) {
  CHECK_INPUT(image);

  return circularFanbeamProjection_cuda(image, nx, ny, ximageside, yimageside,
    radius, source_to_detector, nviews, slen, nbins);
}

torch::Tensor circularFanbeamBackProjection(const torch::Tensor sinogram, const int nx, const int ny, const float ximageside, const float yimageside,
                              const float radius, const float source_to_detector,
                              const int nviews, const float slen, const int nbins) {
  CHECK_INPUT(sinogram);

  return circularFanbeamBackProjection_cuda(sinogram, nx, ny, ximageside, yimageside,
    radius, source_to_detector, nviews, slen, nbins);
}

torch::Tensor circularFanbeamWPDProjection(const torch::Tensor image, const int nx, const int ny, const float ximageside, const float yimageside,
                              const float radius, const float source_to_detector,
                              const int nviews, const float slen, const int nbins) {
  CHECK_INPUT(image);
  return circularFanbeamWPDProjection_cuda(image, nx, ny, ximageside, yimageside,
    radius, source_to_detector, nviews, slen, nbins);
}

torch::Tensor circularFanbeamWPDBackProjection(const torch::Tensor sinogram, const int nx, const int ny, const float ximageside, const float yimageside,
                              const float radius, const float source_to_detector,
                              const int nviews, const float slen, const int nbins) {
  CHECK_INPUT(sinogram);
  return circularFanbeamWPDBackProjection_cuda(sinogram, nx, ny, ximageside, yimageside,
    radius, source_to_detector, nviews, slen, nbins);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("circularFanbeamProjection", &circularFanbeamProjection, "Fanbeam Forward Projection");
  m.def("circularFanbeamBackProjection", &circularFanbeamBackProjection, "Fanbeam Back Projection");
  m.def("circularFanbeamWPDProjection", &circularFanbeamWPDProjection, "Fanbeam Projection, Weighted, Pixel-driven");
  m.def("circularFanbeamWPDBackProjection", &circularFanbeamWPDBackProjection, "Fanbeam Back Projection, Weighted, Pixel-driven");
}
