from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt import NavigationToolbar2QT
from matplotlib.figure import Figure

class GraphCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=3, height=3, dpi=100, position=111):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(position)
        super(GraphCanvas, self).__init__(self.fig)