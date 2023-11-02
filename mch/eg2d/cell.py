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

def magnetic_unit_cell(mag, rs, n3=False, rect=False, n2=False):
  # ref: tbeg/018-vdmc/d_lda1/workflow/wuinp.py & wginp.py
  # primitive cell
  axes0 = triangular_primive_cell(rs)
  # magnetic unit cell
  if n2 and (mag != 'stripe0'):
    msg = 'n2 applies to stripe0 only'
    raise RuntimeError(msg)
  if n2 and (not rect):
    msg = 'n2 requires rectangular cell'
    raise RuntimeError(msg)
  if mag == 'para':
    tmat = np.eye(2, dtype=int)
    order = np.zeros(1, dtype=int)
  elif mag == 'ferro':
    tmat = np.eye(2, dtype=int)
    order = np.zeros(1, dtype=int)
  elif mag == 'stripe':
    tmat = 2*np.eye(2, dtype=int)
    order = np.array([0, 0, 1, 1], dtype=int)
  elif mag == 'stripe0':
    tmat = 2*np.eye(2, dtype=int)
    order = np.array([0, 1, 0, 1], dtype=int)
    if n2:
      tmat = np.array([[1, 0], [1, 2]], dtype=int)
      order = np.array([0, 1], dtype=int)
  elif mag == 'stripe60':
    tmat = 2*np.eye(2, dtype=int)
    order = np.array([0, 1, 1, 0], dtype=int)
  elif mag == '120':
    if n3:
      tmat = np.ones(2, dtype=int)+np.eye(2, dtype=int)
      order = np.array([0, 2, 1])
    else:
      tmat = 3*np.eye(2)
      order = np.array([0, 1, 2, 1, 2, 0, 2, 0, 1], dtype=int)
  pos = tile_cell(axes0, tmat)
  axes = np.dot(tmat, axes0)
  # assign magnetic texture
  assert len(pos) == len(order)
  elem = np.array(['H%d' % i for i in order])
  elem[elem == 'H0'] = 'H'
  return axes, elem, pos

def get_nprim(mag, n3=False, n2=False):
  if mag in ['para', 'ferro']:
    nprim = 1
  elif mag.startswith('stripe'):
    nprim = 4
    if n2:
      nprim = 2
  elif mag == '120':
    nprim = 9
    if n3:
      nprim = 3
  else:
    raise RuntimeError(mag)
  return nprim

def get_magnetic_tilematrix(mag, nelec, ndim=2, n3=False, rect=False, n2=False):
  nprim = get_nprim(mag, n3=n3, n2=n2)
  ntile = nelec//nprim
  assert nprim*ntile == nelec
  if rect:
    if n2:
      tmat = np.eye(2, dtype=int)
    else:
      nx, ny = nxny_from_nelec(ntile)
      tmat = np.array([
        [nx, 0],
        [ny, 2*ny],
      ], dtype=int)
    if mag == '120':
      assert n3
      tmat = np.array([
        [2*ny, -ny],
        [0, nx],
      ], dtype=int)
  else:
    nx = int(round(ntile**0.5))
    tmat = nx*np.eye(ndim, dtype=int)
  # check tmat
  nvol = np.linalg.det(tmat)
  if not np.isclose(ntile, nvol):
    msg = 'wrong tile matrix volume increase %.2f != %.2f' % (nvol, ntile)
    raise RuntimeError(msg)
  return tmat

def nsite_per_magnetic_unit_cell(mag):
  nsite_map = {'para': 1, 'ferro': 1, 'stripe': 4, '120': 9}
  if mag not in nsite_map:
    msg = '%s not in %s' % (mag, str(nsite_map.keys()))
    raise RuntimeError(msg)
  nprim = nsite_map[mag]
  return nprim

def show_structure(ax, aep, ndim=2):
  from qharv.inspect import crystal
  axes, elem, pos = aep
  crystal.draw_cell(ax, axes[:ndim, :ndim])
  for e1 in np.unique(elem):
    sel = elem == e1
    ax.plot(*pos[sel, :ndim].T, ls='', marker='.')

def tile_magnetic(aep, tmat):
  from qharv.inspect.axes_elem_pos import ase_tile
  axes, elem, pos = aep
  ndim = len(axes)
  if ndim == 2:
    axes, pos = extend_axes_pos(axes, pos)
    tmat = matrix_2d_to_3d(tmat)
    tmat[ndim, ndim] = 1
  lalias = 'H1' in elem  # magnetic ions
  if lalias:  # map magnetic ions to difference elements
    # !!!! HACK
    elems = ['H', 'H1', 'H2']
    alias = ['H', 'He', 'Li']
    elem_alias = dict()
    alias_elem = dict()
    for e, name in zip(elems, alias):
      elem_alias[e] = name
      alias_elem[name] = e
    elem = [elem_alias[e] for e in elem]
  axes, elem, pos = ase_tile(axes, elem, pos, tmat)
  if ndim == 2:
    axes = axes[:ndim, :ndim]
    pos = pos[:, :ndim]
  if lalias:  # map back to magnetic ions
    elem = [alias_elem[e] for e in elem]
  elem = np.array(elem)
  return axes, elem, pos

# ============================ xml input ============================
def matrix_2d_to_3d(mat2d, mrow=1, mcol=1):
  nrow, ncol = mat2d.shape
  mat3d = np.zeros([nrow+mrow, ncol+mcol], dtype=mat2d.dtype)
  mat3d[:nrow, :ncol] = mat2d
  return mat3d

def extend_axes_pos(axes, pos):
  from qharv.inspect.axes_pos import rwsc
  ndim = len(axes)
  if ndim == 3:
    return axes, pos
  elif ndim != 2:
    msg = 'ndim = %d' % ndim
    raise RuntimeError(msg)
  alatz = rwsc(axes)
  axes = matrix_2d_to_3d(axes)
  axes[ndim, ndim] = 2*alatz
  pos = matrix_2d_to_3d(pos, mrow=0)
  return axes, pos

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
  if nelec == 2:
    nx = ny = 1
  elif nelec == 4:
    nx = 2
    ny = 1
  elif nelec == 12:
    nx = 3
    ny = 2
  elif nelec == 16:
    nx = 4
    ny = 2
  elif nelec == 30:
    nx = 5  # 2ny-1
    ny = 3
  elif nelec == 48:
    nx = 6
    ny = 4
  elif nelec == 56:
    nx = 7  # 2ny-1
    ny = 4
  elif nelec == 80:
    nx = 8  # 2ny-2
    ny = 5
  elif nelec == 90:
    nx = 9  # 2ny-1
    ny = 5
  elif nelec == 144:
    nx = 12
    ny = 6
  elif nelec == 120:  # 11^2
    nx = 10
    ny = 6
  elif nelec == 168:  # 13^2
    nx = 12
    ny = 7
  elif nelec == 192:
    nx = 12
    ny = 8
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
  elif nelec == 576:
    nx = 24
    ny = 12
  elif nelec == 720:
    nx = 24
    ny = 15
  elif nelec == 750:
    nx = 25
    ny = 15
  elif nelec == 960:
    nx = 30
    ny = 16
  elif nelec == 2880:
    nx = 48
    ny = 30
  elif nelec == 3000:
    nx = 50
    ny = 30
  else:
    raise NotImplementedError(nelec)
  nexpect = nx*ny*2
  if nexpect != nelec:
    msg = 'expected %dx%dx2=%d not %d' % (nx, ny, nexpect, nelec)
    raise RuntimeError(msg)
  return nx, ny

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
  xml.set_param(sc, "bconds", "p p n")
  xml.set_param(sc, "LR_dim_cutoff", str(rckc))
  return sc

def random_pos(axes, nelec):
  fracs2d = np.random.rand(nelec, 2)
  pos2d = np.dot(fracs2d, axes)
  pos = np.zeros([nelec, 3])
  pos[:, :2] = pos2d
  return pos

def randomize_pos(pos, sigma):
  npart, ndim = pos.shape
  return pos + sigma*np.random.randn(*pos.shape)

def make_sposet(nelecs, twist, spo_type='free'):
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
