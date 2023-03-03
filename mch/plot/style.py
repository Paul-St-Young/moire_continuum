def colormap():
  mag_v2c = {
    'ferro': '#1f78b4',
    '120': '#e31a1c',
    'stripe': '#ff7f00',
    'para': '#6a3d9a',
    'heg': '#4daf4a',
  }
  return mag_v2c

def add_legend(ax, mags, marker=None, **kwargs):
  from qharv.field import kyrt
  mag_v2c = colormap()
  styles = []
  for mag in mags:
    myc = mag_v2c[mag]
    style = {'color': myc}
    if marker is not None:
      style['marker'] = marker
      style['ls'] = ''
    styles.append(style)
  label_map = {
    'para': 'Paramagnet',
    'ferro': 'Ferromagnet',
    '120': r'120$^\circ$ Neel',
    'stripe': 'Stripe',
    'heg': '2DEG',
  }
  labels = [label_map[mag] for mag in mags]
  #labels = mags
  return kyrt.create_legend(ax, styles, labels, **kwargs)
