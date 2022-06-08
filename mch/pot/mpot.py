import numpy as np

def mpot(rvecs, am, phi, vm):
  bm = 4*np.pi/3**0.5/am
  raxes = bm*np.array([
    [3**0.5/2, 0.5],
    [0, 1],
  ])
  #gfracs = np.array([
  #  [1, 0],
  #  [-1, 1],
  #  [0, -1],
  #])  # B site at (1./3, 2./3)
  gfracs = np.array([
    [0, 1],
    [-1, 0],
    [1, -1],
  ])  # B site at (2./3, 1./3)
  gvecs = np.dot(gfracs, raxes)
  rg = np.einsum('id,jd->ij', rvecs, gvecs)
  val = vm*2*np.cos(rg+phi).sum(axis=-1)
  return val
