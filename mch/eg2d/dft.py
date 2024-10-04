import numpy as np

def default_pwdict(ecut_pre, rs, vm_by_w, pmoire, func, vfill=1):
  # CODATA 2018
  bohr = 0.529177210903
  ha = 27.211386245988
  # vmoire in effective ha
  vm = -vm_by_w/rs**2
  # amoire in effective bohr
  am = rs*(2*np.pi/3**0.5)**0.5 * vfill**0.5

  ecut = ecut_pre/rs**2  # !!!! fix FFT grid size

  pwdict = dict(
    control = dict(
    ),
    system = dict(
      lmoire = True,
      vmoire_in_mev = vm*ha*1e3,
      amoire_in_ang = am*bohr,
      pmoire_in_deg = pmoire,
      noinv = True,
      nosym = True,
      ecutwfc = ecut, 
      occupations = 'smearing',
      degauss = 1e-4/rs,
    ),
    electrons = dict( 
      diagonalization = 'cg',
      conv_thr = 3e-7/rs,
    ),
  )
  
  if func == 'hf':
    pwdict['system']['input_dft'] = func 
    pwdict['system']['exxdiv_treatment'] = 'madelung'
    pwdict['system']['x_gamma_extrapolation'] = False
    pwdict['electrons'] = dict()
    pwdict['electrons']['adaptive_thr'] = True
  elif func == 'lda':
    #input_dft = 'LDA_X_2D LDA_C_2D_AMGB'
    input_dft = 'XC-019L-015L-000I-000I-000I-000I'
    pwdict['system']['input_dft'] = input_dft
  elif func.startswith('exx'):
    input_dft = 'XC-006I-015L-000I-000I-000I-000I'
    pwdict['system']['input_dft'] = input_dft
    frac = float(func[3:]) 
    pwdict['system']['exx_fraction'] = frac
    pwdict['system']['exxdiv_treatment'] = 'madelung'
    pwdict['system']['x_gamma_extrapolation'] = False
  elif func == 'ni':  # non-interacting
    input_dft = 'XC-019L-015L-000I-000I-000I-000I'
    pwdict['system']['input_dft'] = input_dft
    pwdict['control']['lob'] = True
  else: 
    msg = 'unknown functional %s' % func
    raise RuntimeError(msg)
  return pwdict

def qe_seed_input():
  text = '''&control
  verbosity = 'high'
  outdir = 'qeout'
  disk_io = 'low'
  pseudo_dir = '.'
/
&system
  nosym = .true.
  noinv = .true.
  ibrav = 0
/
&electrons
  electron_maxstep = 1000
/
'''
  return text

def qe_input(aep, pwdict):
  from qharv.cross import pwscf
  from mch.eg2d.cell import extend_axes_pos
  axes, elem, pos = aep
  axes, pos = extend_axes_pos(axes, pos)
  species = np.unique(elem)
  text = qe_seed_input()
  # set keywords
  for group, params in pwdict.items():
    for key, val in params.items():
      text = pwscf.change_keyword(text, group, key, val)
  text = pwscf.change_keyword(text, 'system', 'ntyp', len(species))
  text = pwscf.change_keyword(text, 'system', 'nat', len(pos))
  # add atomic species
  text += '\n\nATOMIC_SPECIES\n'
  for e1 in species:
    text += '%3s 1.0 H.upf\n' % e1
  # add atoms
  fracs = np.dot(pos, np.linalg.inv(axes))
  elem_pos = dict(elements=elem, positions=fracs)
  text += pwscf.atomic_positions(elem_pos)
  # add cell
  text += pwscf.cell_parameters(axes)
  return text

def meta_from_input(finp, ndim=2):
  from qharv.cross import pwscf
  from qharv.inspect import axes_pos
  # CODATA 2018
  bohr = 0.529177210903
  ha = 27.211386245988
  with open(finp, 'r') as f:
    text = f.read()
  inps = pwscf.parse_keywords(text)
  # system
  unit, axes = pwscf.parse_cell_parameters(text, ndim=ndim)
  if unit == 'bohr':
    pass
  elif unit == 'angstrom':
    axes /= bohr
  else:
    raise RuntimeError(unit)
  nat = int(inps['nat'])
  rs = axes_pos.rs(axes, nat)
  vmoire = float(inps['vmoire_in_mev'])*1e-3/ha
  mu = -vmoire*rs**2
  # functional
  func = inps['input_dft']
  if '006I' in func:  # hybrid
    exx = float(inps['exx_fraction'])
    func = 'exx%.2f' % exx
  elif '019L' in func:  # lda
    func = 'lda'
  meta = dict(
    rs = np.around(rs, 3),
    mu = np.around(mu, 3),
    func = func,
  )
  meta.update(inps)
  return meta

def change_rs(text, rs1):
  from qharv.inspect import axes_pos
  from qharv.cross import pwscf
  inps1 = {}
  inps0 = pwscf.parse_keywords(text)

  cell_unit, cell0 = pwscf.parse_cell_parameters(text, ndim=3)

  bohr = 0.529177210544
  nelec = int(inps0['nat'])
  cell_in_bohr = cell0/bohr if cell_unit.lower() == 'angstrom' else cell0
  rs0 = axes_pos.rs(cell_in_bohr[:2, :2], nelec)

  ratio = rs1/rs0

  # rescale lengths
  am0 = float(inps0['amoire_in_ang'])
  am1 = am0 * ratio
  inps1['amoire_in_ang'] = am1

  cell1 = cell0 * ratio
  cell_text = pwscf.cell_parameters(cell1, unit=cell_unit)
  text1 = pwscf.change_block(text, 'CELL_PARAMETERS', cell_text)

  # reset convergence parameters
  ecut0 = float(inps0['ecutwfc'])
  epre = ecut0*rs0**2
  ecut1 = epre/rs1**2
  inps1['ecutwfc'] = ecut1

  inps1['degauss'] = 1e-4/rs1
  inps1['conv_thr'] = 3e-7/rs1

  text1 = pwscf.set_keywords(text1, inps1)
  return text1

def units(mstar, eps):
  length = mstar/eps
  energy = eps*eps/mstar
  return length, energy
