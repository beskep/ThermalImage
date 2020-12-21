import matplotlib.pyplot as plt

from interface.widget.backend_kivyagg import FigureCanvas


def image_widget(image):
  fig, ax = plt.subplots(1, 1)
  ax.imshow(image)
  ax.set_xticks([])
  ax.set_yticks([])
  fig.tight_layout()

  widget = FigureCanvas(fig)

  return widget


def figure_widget(fig):
  widget = FigureCanvas(fig)

  return widget
