import numpy as np

def volume_per_particle(rs, ndim=2):
  return 2*(ndim-1)/ndim*np.pi*rs**ndim

def square_cell(rs, nelec, ndim=2):
  vol = volume_per_particle(rs, ndim=ndim)
  alat = (vol*nelec)**(1./ndim)
  axes = alat*np.eye(ndim)
  return axes

def triangular_primive_cell(rs):
  vol = volume_per_particle(rs, ndim=2)
  alat = ((2./3**0.5)*vol)**0.5
  axes0 = alat*np.array([
    [1, 0],
    [-1./2, 3**0.5/2],
  ])
  return axes0

def matrix_2d_to_3d(mat2d):
  mat3d = np.eye(3)
  mat3d[:2, :2] = mat2d
  return mat3d

def make_ase_atoms(axes, pos, elem=None):
  from ase import Atoms
  from ase.units import Bohr  # convert Bohr to Angstrom
  if elem is None:
    elem = ['H']*len(pos)
  axes1 = axes*Bohr
  pos1 = pos*Bohr
  atoms = Atoms(''.join(elem), cell=axes1, positions=pos1, pbc=True)
  return atoms

def sort_pos2d(pos, axes, nx, ny):
  fracs = np.dot(pos, np.linalg.inv(axes))
  # first sort x
  idx = np.argsort(fracs[:, 0])
  pos1 = pos[idx]
  fracs = fracs[idx]
  # second sort y
  for ix in range(nx):
    istart = ny*ix
    i1 = np.argsort(fracs[istart:istart+ny, 1])
    pos1[range(istart, istart+ny)] = pos1[istart+i1]
  return pos1

def nxny_from_nelec(nelec):
  nx = int(round(np.sqrt(nelec)))
  ny = nx//2
  if nelec == 56:
    nx = 7
    ny = 4
  nexpect = nx*ny*2
  if nexpect != nelec:
    msg = 'expected %dx%dx2=%d not %d' % (nx, ny, nexpect, nelec)
    raise RuntimeError(msg)
  return nx, ny

def simulationcell2d(axes, handler='ewald_strict2d', rckc=30):
  from qharv.inspect import axes_pos
  from qharv.seed import xml, qmcpack_in
  axes1 = np.eye(3)
  axes1[:2, :2] = axes[:2, :2]
  rcut = axes_pos.rins(axes1[:2, :2])
  axes1[2, 2] = 2*rcut  # fake Lz
  sc = qmcpack_in.simulationcell_from_axes(axes1)
  xml.set_param(sc, "LR_handler", handler, new=True)
  xml.set_param(sc, "ndim", '2', new=True)
  xml.set_param(sc, "bconds", "p p n")
  xml.set_param(sc, "LR_dim_cutoff", str(rckc))
  return sc
