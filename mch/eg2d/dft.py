import numpy as np

def default_pwdict(ecut_pre, rs, vm_by_w, pmoire, func):
  # CODATA 2018
  bohr = 0.529177210903
  ha = 27.211386245988
  # vmoire in effective ha
  vm = -vm_by_w/rs**2
  # amoire in effective bohr
  am = rs*(2*np.pi/3**0.5)**0.5

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
