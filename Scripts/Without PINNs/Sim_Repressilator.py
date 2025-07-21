# Simulation of the Repressilator with Odeint and visualization using Bokeh

# %% Install packages
import subprocess
import sys

packages = [
    "biocircuits",
    "colorcet",
    "watermark",
    "bokeh",
]

for package in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package])

# %% Import libraries and configure plotting
import numpy as np
import scipy.integrate
import scipy.optimize
import biocircuits
import bokeh.plotting
import bokeh.io
import colorcet
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Configure Bokeh and Matplotlib for plotting in Python scripts
from bokeh.plotting import output_file, show

# Millora la resolució dels gràfics de Matplotlib
plt.rcParams['figure.dpi'] = 200  # Equivalent a "retina"

# Decideix com vols visualitzar els gràfics interactius de Bokeh
interactive_python_plots = True  # o False si no vols obrir navegador

if interactive_python_plots:
    # Mostra els gràfics Bokeh en un fitxer HTML temporal
    output_file("bokeh_plot.html")
    # Quan tinguis un plot `p`, el mostraràs amb: show(p)
else:
    # Si no vols interactius, pots ignorar output_file i usar show amb reserva
    pass

# %% Protein Repressilator Model
def protein_repressilator_rhs(x, t, beta, n):
    """
    Returns 3-array of (dx_1/dt, dx_2/dt, dx_3/dt)
    """
    x_1, x_2, x_3 = x

    return np.array(
        [
            beta / (1 + x_3 ** n) - x_1,
            beta / (1 + x_1 ** n) - x_2,
            beta / (1 + x_2 ** n) - x_3,
        ]
    )

# Initial condiations
x0 = np.array([1, 1, 1.2])

# Number of points to use in plots
n_points = 1000

# Widgets for controlling parameters
beta_slider_protein = bokeh.models.Slider(title="β", start=0, end=100, step=0.1, value=10)
n_slider_protein = bokeh.models.Slider(title="n", start=1, end=5, step=0.1, value=3)

# Solve for species concentrations
def _solve_protein_repressilator(beta, n, t_max):
    t = np.linspace(0, t_max, n_points)
    x = scipy.integrate.odeint(protein_repressilator_rhs, x0, t, args=(beta, n))

    return t, x.transpose()

# Obtain solutions
t, x = _solve_protein_repressilator(beta_slider_protein.value, n_slider_protein.value, 40.0)

# %% Plotting the Protein Repressilator
# Build the plot
colors = colorcet.b_glasbey_category10[:3]

p_rep = bokeh.plotting.figure(
    frame_width=550, frame_height=200, x_axis_label="t", x_range=[0, 40.0]
)

cds = bokeh.models.ColumnDataSource(data=dict(t=t, x1=x[0], x2=x[1], x3=x[2]))
labels = dict(x1="x₁", x2="x₂", x3="x₃")
for color, x_val in zip(colors, labels):
    p_rep.line(
        source=cds,
        x="t",
        y=x_val,
        color=color,
        legend_label=labels[x_val],
        line_width=2,
    )

p_rep.legend.location = "top_left"


# Set up plot
p_phase = bokeh.plotting.figure(
    frame_width=200, frame_height=200, x_axis_label="x₁", y_axis_label="x₂",
)

p_phase.line(source=cds, x="x1", y="x2", line_width=2)

# Set up callbacks
def _callback(attr, old, new):
    t, x = _solve_protein_repressilator(beta_slider_protein.value, n_slider_protein.value, p_rep.x_range.end)
    cds.data = dict(t=t, x1=x[0], x2=x[1], x3=x[2])

beta_slider_protein.on_change("value", _callback)
n_slider_protein.on_change("value", _callback)
p_rep.x_range.on_change("end", _callback)

# Build layout
protein_repressilator_layout = bokeh.layouts.column(
    p_rep,
    bokeh.layouts.Spacer(height=10),
    bokeh.layouts.row(
        p_phase,
        bokeh.layouts.Spacer(width=70),
        bokeh.layouts.column(beta_slider_protein, n_slider_protein,width=150),
    ),
)

# Build the app
def protein_repressilator_app(doc):
    doc.add_root(protein_repressilator_layout)

# Add the layout to the current document (for Bokeh server)
from bokeh.io import curdoc
curdoc().add_root(protein_repressilator_layout)
