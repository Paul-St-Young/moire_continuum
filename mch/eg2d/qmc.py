def add_dmc_columns(dfd, ymult=1):
  # from: tb30/qmcco/analysis/plot.py
  dfd['etot_mean'] = dfd['LocalEnergy_mean']/dfd['nelec']*ymult
  dfd['etot_error'] = dfd['LocalEnergy_error']/dfd['nelec']*ymult
  dfd['ke_mean'] = dfd['Kinetic_mean']/dfd['nelec']*ymult
  dfd['ke_error'] = dfd['Kinetic_error']/dfd['nelec']*ymult
  dfd['vee_mean'] = dfd['ElecElec_mean']/dfd['nelec']*ymult
  dfd['vee_error'] = dfd['ElecElec_error']/dfd['nelec']*ymult
  dfd['var_mean'] = dfd['Variance_mean']/dfd['nelec']*ymult**2
  dfd['var_error'] = dfd['Variance_error']/dfd['nelec']*ymult**2
  if 'moire_mean' in dfd.columns:
    dfd['um_mean'] = dfd['moire_mean']/dfd['nelec']*ymult
    dfd['um_error'] = dfd['moire_error']/dfd['nelec']*ymult

def cta(df, extra_columns=None, extra_ynames=None):
  from qharv.sieve import mean_df
  #print(mean_df.categorize_columns(df, nosuf=True))
  columns = ['mag', 'rs', 'mu', 'nelec', 'dn', 'func']
  if extra_columns is not None:
    columns += extra_columns
  ynames = ['etot', 'var', 'ke', 'vee']
  if 'um_mean' in df.columns:
    ynames.append('um')
  if extra_ynames is not None:
    ynames += extra_ynames
  adf = df.groupby(columns).apply(mean_df.dfme, ynames)
  for yname in ynames:
    columns += ['%s_mean' % yname, '%s_error' % yname]
  return adf.reset_index()[columns]
