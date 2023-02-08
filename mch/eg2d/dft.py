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
  degauss = 1e-4  # !!!! fix smearing width

  pwdict = dict(
    system = dict(
      lmoire = True,
      vmoire_in_mev = vm*ha*1e3,
      amoire_in_ang = am*bohr,
      pmoire_in_deg = pmoire,
      noinv = True,
      nosym = True,
      ecutwfc = ecut, 
      occupations = 'smearing',
      degauss = degauss,
    ),
    electrons = dict( 
      diagonalization = 'cg',
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
  else: 
    msg = 'unknown functional %s' % func
    raise RuntimeError(msg)
  return pwdict

