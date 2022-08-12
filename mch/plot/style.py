def colormap():
  mag_v2c = {
    'ferro': '#1f78b4',
    '120': '#e31a1c',
    'stripe': '#ff7f00',
    'para': '#6a3d9a',
  }
  return mag_v2c

def add_legend(ax, mags, **kwargs):
  from qharv.field import kyrt
  mag_v2c = colormap()
  styles = [{'color': mag_v2c[mag]} for mag in mags]
  labels = mags
  return kyrt.create_legend(ax, styles, labels, **kwargs)
