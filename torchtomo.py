import torch
import numpy as np
import torchtomo_cuda

torch.backends.cudnn.allow_tf32 = False #essential to add this, otherwise FBP on GPU gives errors (result of conv1d differs)

class CircularFanbeam:
    """
    Defines 2D circular fanbeam projection and backprojection operators,
    plus filtered back projection with a ramp filter, and its adjoint.
    P : Circular Fanbeam Projection operator
    Pt : Associted Backprojection operator (exact adjoint of P)
    WPt : Spatially weighted backprojection (as needed for FBP)
    WP : Exact adjoint of spatially weighted backprojection
    fbp : Filtered back projection
    fbp_adj : Exact adjoint of FBP

    Inputs are
    geom : dictionary containing scan geometry parameters (defined below)
    device : torch GPU device, typically torch.device("cuda:0")
    """
    def __init__(self, geom, device):
        self.nx = geom['nx'] #pixel res in hor. direction
        self.ny = geom['ny'] #pixel res in vert. direction
        self.ximageside = geom['ximageside'] #in cm, size of inscribed rect image
        self.yimageside = geom['yimageside'] #in cm
        self.xlen = geom['xlen'] #in cm, size of global rect
        self.ylen = geom['ylen'] #in cm
        self.radius = geom['radius'] #in cm
        self.source_to_detector = geom['source_to_detector'] #in cm
        self.slen = geom['slen'] #in cm
        self.nviews = geom['nviews'] #num projection views
        self.nbins = geom['nbins']   #num detector bins per view

        if "fbp_filter" in geom:
            self.fbp_filter = geom['fbp_filter']
        else: #default to ram-lak (i.e., pure ramp filter)
            self.fbp_filter = "ram-lak"

        #for backward compatibility -- check for x0 and y0 values in geom dict
        #otherwise replace with defaults 
        if "x0" in geom:
            self.x0 = geom['x0']
        else: 
            self.x0 = -self.ximageside/2.0

        if "y0" in geom:
            self.y0 = geom['y0']
        else: 
            self.y0 = -self.yimageside/2.0

        self.device = device

        #define circular FOV mask
        fanangle2 = np.arcsin((self.xlen/2.)/self.radius) # This only works for ximageside = yimageside
        detectorlength = 2.*np.tan(fanangle2)*self.source_to_detector
        u0 = -detectorlength/2.
        du = detectorlength/self.nbins
        # ds = self.slen/self.nviews #not used
        dup = du*self.radius/self.source_to_detector  #detector bin spacing at iso-center
        dx = self.ximageside/self.nx
        dy = self.yimageside/self.ny
        xar = np.arange(-self.ximageside/2. + dx/2 , self.ximageside/2., dx)[:,np.newaxis]*np.ones([self.nx])
        yar = np.ones([self.nx,self.ny])*np.arange(-self.yimageside/2. + dy/2 , self.yimageside/2., dy)
        rar = np.sqrt(xar**2 + yar**2)
        mask = np.float32(np.zeros([self.nx,self.ny]))
        mask[rar<=self.ximageside/2.]=1.
        self.mask = torch.from_numpy(mask).to(self.device)

        #define ramp filter and data weighting needed for FBP
        if self.fbp_filter == "ram-lak":
            rfilter = ramp_kernel(2*self.nbins-1,dup)*dup
        elif self.fbp_filter == "shepp-logan":
            rfilter = shepplogan_kernel(2*self.nbins-1,dup)*dup
        else: #default to ramp
            rfilter = ramp_kernel(2*self.nbins-1,dup)*dup
        
        u0 = -detectorlength/2.
        uarray = np.arange(u0 + du/2., u0+ du/2. + detectorlength, du)
        uarray *= self.radius/self.source_to_detector
        data_weight = self.radius/np.sqrt(self.radius**2 + uarray**2)
        self.rfilter = torch.from_numpy(np.float32(rfilter)).view(1,1,-1).to(self.device)
        self.data_weight = torch.from_numpy(np.float32(data_weight)).to(self.device)

    def P(self,img):
        return torchtomo_cuda.circularFanbeamProjection(self.mask*img, self.nx, self.ny, self.xlen, self.ylen, self.ximageside, self.yimageside, self.x0, self.y0, self.radius, self.source_to_detector, self.nviews, self.slen, self.nbins)

    def Pt(self,sino):
        return self.mask*torchtomo_cuda.circularFanbeamBackProjection_kbn(sino, self.nx, self.ny, self.ximageside, self.yimageside, self.x0, self.y0, self.radius, self.source_to_detector, self.nviews, self.slen, self.nbins)
    
    # def Pt(self,sino):
    #     return self.mask*torchtomo_cuda.circularFanbeamBackProjection(sino, self.nx, self.ny, self.ximageside, self.yimageside, self.radius, self.source_to_detector, self.nviews, self.slen, self.nbins)

    def WP(self,img):
        return torchtomo_cuda.circularFanbeamWPDProjection(self.mask*img, self.nx, self.ny, self.ximageside, self.yimageside, self.x0, self.y0, self.radius, self.source_to_detector, self.nviews, self.slen, self.nbins)

    def WPt(self,sino):
        return self.mask*torchtomo_cuda.circularFanbeamWPDBackProjection(sino, self.nx, self.ny, self.ximageside, self.yimageside, self.x0, self.y0, self.radius, self.source_to_detector, self.nviews, self.slen, self.nbins)

    # Applies the filter to weighted sinogram needed for fanbeam FBP
    def fbp(self,sino):
        [nbatch,nviews,nbins] = sino.shape
        sinoflat = sino.view(nbatch*nviews,1,-1) #flatten sinogram to perform parallel 1-D convolution
        filteredweightedsino = torch.nn.functional.conv1d(sinoflat*self.data_weight,self.rfilter,padding='same').view(nbatch,nviews,nbins)
        return self.WPt(filteredweightedsino)

    def fbp_adj(self,img):
        sino = self.WP(img)
        [nbatch,nviews,nbins] = sino.shape
        sinoflat = sino.view(nbatch*nviews,1,-1) #flatten sinogram to perform parallel 1-D convolution
        filteredweightedsino = torch.nn.functional.conv1d(sinoflat,self.rfilter,padding='same').view(nbatch,nviews,nbins)
        return filteredweightedsino*self.data_weight

# Ramp filter from Kak and Slaney "Principles of Computerized Tomographic Imaging" needed for FBP
def ramp_kernel(n,du):
  it_is_even = 0
  if np.mod(n,2)==0:
      it_is_even = 1

  if it_is_even:
      nr = np.arange(-n/2., n/2., 1.)
      rfilter = nr*0.
      rfilter = ((-1.)**nr)/(2.*du*du*np.pi*nr + du*du*np.pi)- 1./((8.*du*du)*(np.pi*nr/2. +np.pi/4.)**2)

  else:
      nr = np.arange(-(n-1)/2., (n-1)/2. + 1., 1.)
      rfilter = nr*0.
      rfilter[int((n-1)/2+1)::2] = -1./(np.pi*du*nr[int((n-1)/2+1)::2])**2
      rfilter[int((n-1)/2-1)::-2] = -1./(np.pi*du*nr[int((n-1)/2-1)::-2])**2
      rfilter[int((n-1)/2)::2] = 0.
      rfilter[int((n-1)/2)::-2] = 0.
      rfilter[int((n-1)/2)] = 1./(4.*du*du)

  return rfilter/2.

#Shepp-logan filter
def shepplogan_kernel(n,du):
    it_is_even = 0
    if np.mod(n,2)==0:
        it_is_even = 1

    if it_is_even:
        nr = np.arange(-n/2., n/2., 1.)

    else:
        nr = np.arange(-(n-1)/2., (n-1)/2. + 1., 1.)

    filter = -2./(((np.pi*du)**2)*(4.*nr**2-1))

    return filter/2.

def diffP(TObj):
    """
    Wrapper to define autograd compatible projector operator in torch
    """
    class Projection(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return TObj.P(x)

        @staticmethod
        def backward(ctx, grad_x):
            if not grad_x.is_contiguous():
                grad_x = grad_x.contiguous()
            return TObj.Pt(grad_x)

    return Projection.apply

def diffFBP(ttObj):
    """
    Wrapper to define autograd compatible FBP operator in torch
    """
    class FBP(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return ttObj.fbp(x)

        @staticmethod
        def backward(ctx, grad_x):
            if not grad_x.is_contiguous():
                grad_x = grad_x.contiguous()
            return ttObj.fbp_adj(grad_x)

    return FBP.apply
