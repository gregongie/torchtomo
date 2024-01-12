#include <torch/extension.h>
// #include <vector>
// #include <cmath>

torch::Tensor circularFanbeamProjection(const torch::Tensor image, const int nx, const int ny, const float ximageside, const float yimageside,
                              const float radius, const float source_to_detector,
                              const int nviews, const float slen, const int nbins) {
  const float dx = ximageside/nx;
  const float dy = yimageside/ny;
  const float x0 = -ximageside/2.0;
  const float y0 = -yimageside/2.0;

  // compute length of detector so that it views the inscribed FOV of the image array
  const float fanangle2 = std::asin((ximageside/2.0)/radius);  //This only works for ximageside = yimageside
  const float detectorlength = 2.0*std::tan(fanangle2)*source_to_detector;
  const float u0 = -detectorlength/2.0;

  const float du = detectorlength/nbins;
  const float ds = slen/nviews;

  torch::Tensor sinogram = torch::zeros({nviews, nbins});
  auto sinogram_a = sinogram.accessor<float,2>(); //accessor for updating values of sinogram

  const auto image_a = image.accessor<float,2>(); //accessor for reading from image

  //loop over views -- parallelize over this loop!
  for (int sindex = 0; sindex < nviews; sindex++){
    float s = sindex*ds;

    // location of the source
    float xsource = radius*std::cos(s);
    float ysource = radius*std::sin(s);

    // detector center
    float xDetCenter = (radius - source_to_detector)*std::cos(s);
    float yDetCenter = (radius - source_to_detector)*std::sin(s);

    // unit vector in the direction of the detector line
    float eux = -std::sin(s);
    float euy =  std::cos(s);

    //Unit vector in the direction perpendicular to the detector line -- unused
    // auto ewx = cos(s);
    // auto ewy = sin(s);

    //loop over detector views
    for (int uindex = 0; uindex < nbins; uindex++){
      auto u = u0 + (uindex+0.5)*du;
      auto xbin = xDetCenter + eux*u;
      auto ybin = yDetCenter + euy*u;

      auto xl = x0;
      auto yl = y0;

      auto xdiff = xbin-xsource;
      auto ydiff = ybin-ysource;
      auto xad = std::abs(xdiff)*dy;
      auto yad = std::abs(ydiff)*dx;

      float raysum = 0.0; // acculumator variable

      if (xad > yad){  // loop through x-layers of image if xad>yad. This ensures ray hits only one or two pixels per layer
        auto slope = ydiff/xdiff;
        auto travPixlen = dx*std::sqrt(1.0+slope*slope);
        auto yIntOld = ysource+slope*(xl-xsource);
        int iyOld = static_cast<int>(std::floor((yIntOld-y0)/dy));
        // loop over x-layers
        for (int ix = 0; ix < nx; ix++){
           auto x = xl + dx*(ix + 1.0);
           auto yIntercept = ysource + slope*(x-xsource);
           int iy = static_cast<int>(std::floor((yIntercept-y0)/dy));
           if (iy == iyOld){ // if true, ray stays in the same pixel for this x-layer
              if ((iy >= 0) && (iy < ny)) {
                 raysum += travPixlen*image_a[ix][iy];
              }
           } else {    // else case is if ray hits two pixels for this x-layer
              auto yMid=dy*std::max(iy,iyOld)+yl;
              auto ydist1=std::abs(yMid-yIntOld);
              auto ydist2=std::abs(yIntercept-yMid);
              auto frac1=ydist1/(ydist1+ydist2);
              auto frac2=1.0-frac1;
              if ((iyOld >= 0) && (iyOld < ny)){
                 raysum += frac1*travPixlen*image_a[ix][iyOld];
               }
              if ((iy>=0) && (iy<ny)){
                 raysum += frac2*travPixlen*image_a[ix][iy];
               }
           }
           iyOld=iy;
           yIntOld=yIntercept;
         }

      } else {// through y-layers of image if xad<=yad
        auto slopeinv=xdiff/ydiff;
        auto travPixlen=dy*std::sqrt(1.0+slopeinv*slopeinv);
        auto xIntOld=xsource+slopeinv*(yl-ysource);
        int ixOld= static_cast<int>(std::floor((xIntOld-x0)/dx));
        // loop over y-layers
        for (int iy = 0; iy < ny; iy++){
           auto y=yl+dy*(iy + 1.0);
           auto xIntercept=xsource+slopeinv*(y-ysource);
           int ix = static_cast<int>(std::floor((xIntercept-x0)/dx));
           if (ix == ixOld){// if true, ray stays in the same pixel for this y-layer
              if ((ix >= 0) && (ix < nx)){
                 raysum += travPixlen*image_a[ix][iy];
               }
           } else {  // else case is if ray hits two pixels for this y-layer
              auto xMid=dx*std::max(ix,ixOld)+xl;
              auto xdist1=std::abs(xMid-xIntOld);
              auto xdist2=std::abs(xIntercept-xMid);
              auto frac1=xdist1/(xdist1+xdist2);
              auto frac2=1.0-frac1;
              if ((ixOld >= 0) && (ixOld < nx)){
                 raysum += frac1*travPixlen*image_a[ixOld][iy];
              }
              if ((ix>=0) && (ix<nx)){
                 raysum += frac2*travPixlen*image_a[ix][iy];
              }
           }
           ixOld = ix;
           xIntOld = xIntercept;
         }
      }
      sinogram_a[sindex][uindex]=raysum;
   }
 }
 return sinogram;
}

// exact matrix transpose of circularFanbeamProjection
torch::Tensor circularFanbeamBackProjection(const torch::Tensor sinogram, const int nx, const int ny,
                              const float ximageside, const float yimageside,
                              const float radius, const float source_to_detector,
                              const int nviews, const float slen, const int nbins) {
   const float dx = ximageside/nx;
   const float dy = yimageside/ny;
   const float x0 = -ximageside/2.0;
   const float y0 = -yimageside/2.0;

   // compute length of detector so that it views the inscribed FOV of the image array
   const float fanangle2 = std::asin((ximageside/2.0)/radius);  //This only works for ximageside = yimageside
   const float detectorlength = 2.0*std::tan(fanangle2)*source_to_detector;
   const float u0 = -detectorlength/2.0;

   const float du = detectorlength/nbins;
   const float ds = slen/nviews;

   torch::Tensor image = torch::zeros({nx, ny}); //initialize image
   auto image_a = image.accessor<float,2>(); //accessor for updating values of image
   const auto sinogram_a = sinogram.accessor<float,2>(); //accessor for accessing values of sinogram

   //loop over views -- parallelize over this loop!
   for (int sindex = 0; sindex < nviews; sindex++){
     float s = sindex*ds;

     // location of the source
     float xsource = radius*std::cos(s);
     float ysource = radius*std::sin(s);

     // detector center
     float xDetCenter = (radius - source_to_detector)*std::cos(s);
     float yDetCenter = (radius - source_to_detector)*std::sin(s);

     // unit vector in the direction of the detector line
     float eux = -std::sin(s);
     float euy =  std::cos(s);

     for (int uindex = 0; uindex < nbins; uindex++){
       auto sinoval = sinogram_a[sindex][uindex];
       float u = u0+(uindex+0.5)*du;
       float xbin = xDetCenter + eux*u;
       float ybin = yDetCenter + euy*u;

       float xl=x0;
       float yl=y0;

       float xdiff=xbin-xsource;
       float ydiff=ybin-ysource;
       float xad=std::abs(xdiff)*dy;
       float yad=std::abs(ydiff)*dx;

       if (xad>yad){   // loop through x-layers of image if xad>yad. This ensures ray hits only one or two pixels per layer
          float slope=ydiff/xdiff;
          float travPixlen=dx*std::sqrt(1.0+slope*slope);
          float yIntOld=ysource + slope*(xl-xsource);
          int iyOld = static_cast<int>(std::floor((yIntOld-y0)/dy));
          for (int ix = 0; ix < nx; ix++){
             float x=xl+dx*(ix + 1.0);
             float yIntercept=ysource+slope*(x-xsource);
             int iy = static_cast<int>(std::floor((yIntercept-y0)/dy));
             if (iy == iyOld){ // if true, ray stays in the same pixel for this x-layer
                if ((iy >= 0) && (iy < ny)){
                   image_a[ix][iy] += sinoval*travPixlen;
                 }
             } else {    // else case is if ray hits two pixels for this x-layer
                float yMid = dy*std::max(iy,iyOld)+yl;
                float ydist1 = std::abs(yMid-yIntOld);
                float ydist2 = std::abs(yIntercept-yMid);
                float frac1 = ydist1/(ydist1+ydist2);
                float frac2 = 1.0-frac1;
                if ((iyOld >= 0) && (iyOld < ny)){
                   image_a[ix][iyOld] += frac1*sinoval*travPixlen;
                 }
                if ((iy >= 0) && (iy < ny)) {
                   image_a[ix][iy] += frac2*sinoval*travPixlen;
                 }
             }
             iyOld=iy;
             yIntOld=yIntercept;
           }
       } else { //loop through y-layers of image if xad<=yad
          float slopeinv=xdiff/ydiff;
          float travPixlen=dy*std::sqrt(1.0+slopeinv*slopeinv);
          float xIntOld=xsource+slopeinv*(yl-ysource);
          int ixOld = static_cast<int>(std::floor((xIntOld-x0)/dx));
          for (int iy = 0; iy < ny; iy++){
             float y = yl + dy*(iy + 1.0);
             float xIntercept = xsource+slopeinv*(y-ysource);
             int ix = static_cast<int>(std::floor((xIntercept-x0)/dx));
             if (ix == ixOld){ // if true, ray stays in the same pixel for this y-layer
                if ((ix >= 0) && (ix < nx)) {
                   image_a[ix][iy] += sinoval*travPixlen;
                 }
             } else { // else case is if ray hits two pixels for this y-layer
                float xMid = dx*std::max(ix,ixOld)+xl;
                float xdist1 = std::abs(xMid-xIntOld);
                float xdist2 = std::abs(xIntercept-xMid);
                float frac1 = xdist1/(xdist1+xdist2);
                float frac2=1.0-frac1;
                if ((ixOld >= 0) && (ixOld < nx)){
                   image_a[ixOld][iy] += frac1*sinoval*travPixlen;
                 }
                if ((ix >= 0) && (ix < nx)){
                   image_a[ix][iy] += frac2*sinoval*travPixlen;
                 }
             }
             ixOld = ix;
             xIntOld = xIntercept;
          }
        }
      } // end uindex for loop
    } // end sindex for loop
    // also should mask image to fovradius
    return image;
}

// Backprojection as needed for weighted FBP (not exact matrix transpose)
torch::Tensor circularFanbeamBackProjectionPixelDriven(const torch::Tensor sinogram, const int nx, const int ny,
                              const float ximageside, const float yimageside,
                              const float radius, const float source_to_detector,
                              const int nviews, const float slen, const int nbins) {
   const float dx = ximageside/nx;
   const float dy = yimageside/ny;
   const float x0 = -ximageside/2.0;
   const float y0 = -yimageside/2.0;

   // compute length of detector so that it views the inscribed FOV of the image array
   const float fanangle2 = asin((ximageside/2.0)/radius);  //This only works for ximageside = yimageside
   const float detectorlength = 2.0*tan(fanangle2)*source_to_detector;
   const float u0 = -detectorlength/2.0;

   const float du = detectorlength/nbins;
   const float ds = slen/nviews;

   const float fov_radius = ximageside/2.0;

   torch::Tensor image = torch::zeros({nx, ny}); //initialize image
   auto image_a = image.accessor<float,2>(); //accessor for updating values of image
   const auto sinogram_a = sinogram.accessor<float,2>(); //accessor for accessing values of sinogram

   const float pi = 4*atan(1);

   //loop over views -- parallelize over this loop!
   for (int sindex = 0; sindex < nviews; sindex++){
     float s = sindex*ds;

     // location of the source
     float xsource = radius*cos(s);
     float ysource = radius*sin(s);

     // detector center
     float xDetCenter = (radius - source_to_detector)*cos(s);
     float yDetCenter = (radius - source_to_detector)*sin(s);

     // unit vector in the direction of the detector line
     float eux = -sin(s);
     float euy =  cos(s);

     //Unit vector in the direction perpendicular to the detector line
     float ewx = cos(s);
     float ewy = sin(s);

     for (int iy = 0; iy < ny; iy++){
        float pix_y = y0 + dy*(iy+0.5);
        for (int ix = 0; ix < nx; ix++){
           float pix_x = x0 + dx*(ix+0.5);

           float frad = sqrt(pix_x*pix_x + pix_y*pix_y);
           float fphi = atan2(pix_y,pix_x);
           if (frad<=fov_radius){
              float bigu = (radius+frad*sin(s-fphi-pi/2.0))/radius;
              float bpweight = 1.0/(bigu*bigu);

              float ew_dot_source_pix = (pix_x-xsource)*ewx + (pix_y-ysource)*ewy;
              float rayratio = -source_to_detector/ew_dot_source_pix;

              float det_int_x = xsource+rayratio*(pix_x-xsource);
              float det_int_y = ysource+rayratio*(pix_y-ysource);

              float upos = ((det_int_x-xDetCenter)*eux +(det_int_y-yDetCenter)*euy);
              float det_value;

              if ((upos-u0 >= du/2.0) && (upos-u0 < detectorlength-du/2.0)){
                 float bin_loc = (upos-u0)/du + 0.5;
                 int nbin1 = static_cast<int>(bin_loc)-1;
                 int nbin2 = nbin1+1;
                 float frac= bin_loc - static_cast<int>(bin_loc);
                 det_value = frac*sinogram_a[sindex][nbin2]+(1.0-frac)*sinogram_a[sindex][nbin1];
              } else {
                 det_value = 0.0;
              }
              image_a[ix][iy] += bpweight*det_value*ds;
          }
       }
    }
  }
  return image;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("circularFanbeamProjection", &circularFanbeamProjection, "Fanbeam Forward Projection");
  m.def("circularFanbeamBackProjection", &circularFanbeamBackProjection, "Fanbeam Backprojection");
  m.def("circularFanbeamBackProjectionPixelDriven", &circularFanbeamBackProjectionPixelDriven, "Fanbeam Backprojection, Pixel-driven");
}
