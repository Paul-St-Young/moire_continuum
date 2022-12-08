import numpy as np

def density_to_rs(n_in_cmm2, eps=1, mstar=1):
  bohr = 0.529177210903
  a_in_meter = 1./(np.pi*n_in_cmm2)**0.5/1e2
  a_in_bohr = a_in_meter*1e10/bohr
  return a_in_bohr*mstar/eps

def rs_to_density(rs, eps=1, mstar=1):
  bohr = 0.529177210903
  a_in_bohr = rs*eps/mstar
  a_in_cm = a_in_bohr*bohr*1e-8
  n_in_cmm2 = 1./(np.pi*a_in_cm**2)
  return n_in_cmm2

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
  if nelec == 30:
    nx = 5  # 2ny-1
    ny = 3
  elif nelec == 56:
    nx = 7  # 2ny-1
    ny = 4
  elif nelec == 80:
    nx = 8  # 2ny-2
    ny = 5
  elif nelec == 90:
    nx = 9  # 2ny-1
    ny = 5
  elif nelec == 120:  # 11^2
    nx = 10
    ny = 6
  elif nelec == 168:  # 13^2
    nx = 12
    ny = 7
  elif nelec == 224:  # 15^2
    nx = 14
    ny = 8
  elif nelec == 288:  # 17^2
    nx = 16
    ny = 9
  elif nelec == 360:  # 19^2
    nx = 18
    ny = 10
  elif nelec == 418:
    nx = 19
    ny = 11
  nexpect = nx*ny*2
  if nexpect != nelec:
    msg = 'expected %dx%dx2=%d not %d' % (nx, ny, nexpect, nelec)
    raise RuntimeError(msg)
  return nx, ny

def tile_cell(axes0, tmat, edge_tol=1e-8):
  from qharv.inspect import axes_pos
  ndim = len(axes0)
  nsh = 2*np.diag(tmat).max()
  fracs = axes_pos.cubic_pos(nsh, ndim=ndim)
  f1 = np.dot(fracs, np.linalg.inv(tmat))
  sel = np.ones(len(f1), dtype=bool)
  for idim in range(ndim):
    sel = sel & (0-edge_tol <= f1[:, idim]) & (f1[:, idim] < 1-edge_tol)
  pos = np.dot(fracs[sel], axes0)
  return pos

def simulationcell2d(axes, handler='ewald_strict2d', rckc=30, nondiag=False):
  from qharv.inspect import axes_pos
  from qharv.seed import xml, qmcpack_in
  axes1 = np.eye(3)
  if nondiag:
    axes1[:2, :2] = axes
  else:
    axes1[:2, :2] = np.diag(np.diag(axes))
    if not np.allclose(axes1[:2, :2], axes):
      msg = 'cell is not diagonal %s' % str(axes)
      raise RuntimeError(msg)
  rcut = axes_pos.rins(axes1[:2, :2])
  axes1[2, 2] = 2*rcut  # fake Lz
  sc = qmcpack_in.simulationcell_from_axes(axes1)
  xml.set_param(sc, "LR_handler", handler, new=True)
  xml.set_param(sc, "ndim", '2', new=True)
  xml.set_param(sc, "bconds", "p p n")
  xml.set_param(sc, "LR_dim_cutoff", str(rckc))
  return sc

def random_pos(axes, nelec):
  fracs2d = np.random.rand(nelec, 2)
  pos2d = np.dot(fracs2d, axes)
  pos = np.zeros([nelec, 3])
  pos[:, :2] = pos2d
  return pos

def make_sposet(nelecs, twist, spo_type='pw'):
  from qharv.seed import xml
  spo_name = 'spo-ud'  # !!!! hard-code one SPO set for all species
  spo_map = dict()
  nl = []
  for name, n in nelecs.items():
    spo_map[name] = spo_name
    nl.append(n)
  mxelec = max(nl)
  tt = ' '.join(twist.astype(str))
  bb = xml.make_node('sposet_builder', {'type': spo_type})
  spo = xml.make_node('sposet', {'name': spo_name, 'size': str(mxelec), 'twist': tt})
  bb.append(spo)
  return bb, spo_map

def make_detset(spo_map):
  from qharv.seed import xml
  detset = xml.make_node('determinantset')
  sdet = xml.make_node('slaterdeterminant')
  detset.append(sdet)
  for species, spo_name in spo_map.items():
    det = xml.make_node('determinant', {'id': 'det%s' % species, 'sposet': spo_name})
    sdet.append(det)
  return detset
