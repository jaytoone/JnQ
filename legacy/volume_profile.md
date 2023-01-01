```
import pandas as pd
import numpy as np
from scipy import stats, signal
import plotly.express as px
import plotly.graph_objects as go
```



### Fetch OHLCV data
```python
data = some_data_load_function('EURUSD')
volume = data['volume']
close = data['close']

px.histogram(data, x='volume', y='close', nbins=150, orientation='h').show()
```



### Kernel Density Estimator

```
kde_factor = 0.05
num_samples = 500
kde = stats.gaussian_kde(close,weights=volume,bw_method=kde_factor)
xr = np.linspace(close.min(),close.max(),num_samples)
kdy = kde(xr)
ticks_per_sample = (xr.max() - xr.min()) / num_samples

def get_dist_plot(c, v, kx, ky):
    fig = go.Figure()
    fig.add_trace(go.Histogram(name='Vol Profile', x=c, y=v, nbinsx=150, 
                               histfunc='sum', histnorm='probability density',
                               marker_color='#B0C4DE'))
    fig.add_trace(go.Scatter(name='KDE', x=kx, y=ky, mode='lines', marker_color='#D2691E'))
    return fig

get_dist_plot(close, volume, xr, kdy).show()
```

<img src="https://miro.medium.com/max/1254/1*I9jke4zawLL12Ay7z9bdtg.png" alt="img" style="zoom: 80%;" />



### Finding Volume Nodes
```python
peaks,_ = signal.find_peaks(kdy)
pkx = xr[peaks]
pky = kdy[peaks]

pk_marker_args=dict(size=10)
fig = get_dist_plot(close, volume, xr, kdy)
fig.add_trace(go.Scatter(name="Peaks", x=pkx, y=pky, mode='markers', marker=pk_marker_args))
```

<img src="https://miro.medium.com/max/1400/1*fQ_OFT4wfUALrXWEU4I9Ag.png" alt="img" style="zoom:80%;" />

### Prominence

```

min_prom = 1
peaks, peak_props = signal.find_peaks(kdy, prominence=min_prom)
pkx = xr[peaks]
pky = kdy[peaks]

fig = get_dist_plot(close, volume, xr, kdy)
fig.add_trace(go.Scatter(name='Peaks', x=pkx, y=pky, mode='markers', marker=pk_marker_args))

# Draw prominence lines
left_base = peak_props['left_bases']
right_base = peak_props['right_bases']
line_x = pkx
line_y0 = pky
line_y1 = pky - peak_props['prominences']

for x, y0, y1 in zip(line_x, line_y0, line_y1):
    fig.add_shape(type='line',
        xref='x', yref='y',
        x0=x, y0=y0, x1=x, y1=y1,
        line=dict(
            color='red',
            width=2,
        )
    )
fig.show()
```

<img src="https://miro.medium.com/max/1400/1*FVBDecof9xEUaRsXRdPDJQ.png" alt="img" style="zoom:80%;" />![img](https://miro.medium.com/max/1400/1*tlQUlPQfa9jsCXTd0e_syQ.png)

```
min_prom = kdy.max() * 0.3
```

![img](https://miro.medium.com/max/1400/1*tlQUlPQfa9jsCXTd0e_syQ.png)

### Peak Width

```

width_range=1
peaks, peak_props = signal.find_peaks(kdy, prominence=min_prom, width=width_range)

left_ips = peak_props['left_ips']
right_ips = peak_props['right_ips']
width_x0 = xr.min() + (left_ips * ticks_per_sample)
width_x1 = xr.min() + (right_ips * ticks_per_sample)
width_y = peak_props['width_heights']

fig = get_dist_plot(close, volume, xr, kdy)
fig.add_trace(go.Scatter(name='Peaks', x=pkx, y=pky, mode='markers', marker=pk_marker_args))

for x0, x1, y in zip(width_x0, width_x1, width_y):
    fig.add_shape(type='line',
        xref='x', yref='y',
        x0=x0, y0=y, x1=x1, y1=y,
        line=dict(
            color='red',
            width=2,
        )
    )
fig.show()
```

<img src="https://miro.medium.com/max/1400/1*mX0fmXrJExNDIWeylNGUtA.png" alt="img" style="zoom: 80%;" />

```
pipsize = 0.0001
max_width_pips = 20
min_prom = kdy.max() * 0.3
width_range=(1, max_width_pips * pipsize / ticks_per_sample)
peaks, peak_props = signal.find_peaks(kdy, width=width_range, prominence=min_prom)
pkx = xr[peaks]
pky = kdy[peaks]
```

<img src="https://miro.medium.com/max/1400/1*AxWOMnnOkF0cIYQ67s4UlA.png" alt="img" style="zoom:80%;" />

### Density

```
min_prom = kdy.max() * 0.3
peaks, peak_props = signal.find_peaks(kdy, prominence=min_prom)

left_base = peak_props['left_bases']
right_base = peak_props['right_bases']
int_from = xr.min() + (left_base * ticks_per_sample)
int_to = xr.min() + (right_base * ticks_per_sample)

[kde.integrate_box_1d(x0, x1) for x0, x1 in zip(int_from, int_to)]
```

**\> [0.631087355798145, 0.366996783471251]**

