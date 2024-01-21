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
  #])  # B site at (2./3, 1./3)
  gfracs = np.array([
    [0, 1],
    [-1, 0],
    [1, -1],
  ])  # B site at (1./3, 2./3)
  gvecs = np.dot(gfracs, raxes)
  rg = np.einsum('id,jd->ij', rvecs, gvecs)
  val = vm*2*np.cos(rg+phi).sum(axis=-1)
  return val

def pw_init(raxes, ecut):
  from qharv.inspect import axes_pos
  from qharv.seed.hamwf_h5 import get_ksphere
  kcut = (2*ecut)**0.5
  kvecs = get_ksphere(raxes, kcut)
  gvecs = axes_pos.get_nvecs(raxes, kvecs)
  return gvecs

def ham_one_body(kvecs, vkcut, vm, phi, lam=1./2):
  #pshift = np.pi/6  # B site at (1./3, 2./3)
  pshift = np.pi/6+np.pi/3  # B site at (2./3, 1./3)
  # kinetic
  kmags = np.linalg.norm(kvecs, axis=-1)
  k2 = kmags**2
  kmat = np.diag(lam*k2.astype(np.complex128))
  # potential
  kdisp = kvecs[:, np.newaxis] - kvecs[np.newaxis, :]
  kdist = np.linalg.norm(kdisp, axis=-1)
  sel = kdist < vkcut
  for ik, (k1, sel1) in enumerate(zip(kvecs, sel)):
    idx = [i for i in np.where(sel1)[0] if i != ik]
    # connect kp, k1 separated by g
    dkl = [kp-k1 for kp in kvecs[sel1] if not np.allclose(k1, kp)]
    kx, ky = np.array(dkl).T
    phis = np.arctan2(ky, kx)+pshift
    phase = np.cos(3*phis)
    # set interaction matrix elements
    kmat[ik, idx] = vm*np.exp(1j*phase*phi)
  return kmat
