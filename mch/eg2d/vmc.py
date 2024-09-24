import numpy as np
from qharv.inspect import axes_pos
from qharv.cross import pwscf, pwscf_xml, pwscf_hdf

def read_wf(finp: str, ikpt: int=0, ndim: int=2) -> dict:
    bohr = 0.529177210903
    ha = 27.211386245988
    # read metadata
    fxml = pwscf.find_xml(finp)
    doc = pwscf_xml.read(fxml)
    mesh = pwscf_xml.read_fft_mesh(doc)[:ndim]
    cell = pwscf_xml.read_cell(doc)[:ndim, :ndim]
    elem, pos = pwscf_xml.read_elem_pos(doc)
    wcpos = pos[:, :ndim]

    nspin = pwscf_xml.read_nspin(doc)
    # read orbital coefficients
    dsave = fxml[:-4] + '.save'
    # !!!! HACK: assume polarized or noncolin
    pre = 'wfcup' if nspin == 2 else 'wfc'
    fwf = '%s/%s%d.hdf5' % (dsave, pre, ikpt+1)
    gvecs, coeff = pwscf_hdf.read_save_hdf(fwf)
    gvecs = gvecs[:, :ndim]
    data = {
        'cell': cell,
        'pos': wcpos,
        'nspin': nspin,
        'mesh': mesh,
        'gvecs': gvecs,
        'coeff': coeff,
    }
    # read moire parameters
    inps = pwscf.input_keywords(finp)
    moire_params = extract_moire_params(inps)
    am = moire_params['amoire_in_ang']/bohr
    vm = moire_params['vmoire_in_mev']/1e3/ha
    phi = moire_params['pmoire_in_deg']*np.pi/180
    data['moire'] = {
      'am_length': am,
      'vm_depth': vm,
      'phi_shape': phi,
    }
    if 'wmoire_in_mev' in moire_params:
      wm = moire_params['wmoire_in_mev']/1e3/ha
      data['moire']['wm_depth'] = wm
    if 'moire_dfield_in_mev' in moire_params:
      uD = moire_params['moire_dfield_in_mev']/1e3//ha
      data['moire']['uD_field'] = uD
    if 'dgate' in moire_params:
      data['moire']['dgate_dist'] = moire_params['dgate']
    return data

def extract_moire_params(inps):
  mydict = {}
  keys = ['amoire_in_ang', 'vmoire_in_mev', 'wmoire_in_mev', 'moire_dfield_in_mev', 'dgate', 'epsmoire',
    'pmoire_in_deg', 'mstar']
  for key in keys:
    if key in inps:
      mydict[key] = float(inps[key])
  return mydict

def get_system_info(data):
    cell = data['cell']
    wcpos = data['pos']
    gvecs = data['gvecs']
    coeff = data['coeff']
    nspin = data['nspin']
    nelec = len(wcpos)
    rs = axes_pos.rs(cell, nelec)
    recvec = axes_pos.raxes(cell)
    kvecs = gvecs @ recvec
    # occupy lowest orbitals
    npw = len(gvecs)
    if nspin == 4:
      spins = (nelec,)
      psi_up = coeff[:nelec, :npw]
      psi_dn = coeff[:nelec, npw:]
      mycoeff = np.vstack([psi_up, psi_dn]).reshape(2, nelec, npw)
    elif nspin == 2:  # !!!! HACK: assume polarized
      spins = (nelec,)
      mycoeff = coeff[:nelec]
    elif nspin == 1:
      nup = nelec//2
      ndn = nelec-nup
      assert nup == ndn
      spins = (nup, ndn)
      mycoeff = coeff[:nup]
    nd = len(mycoeff.shape)
    idx = list(range(nd-1, -1, -1))
    return spins, cell, wcpos, kvecs, mycoeff.transpose(idx)
