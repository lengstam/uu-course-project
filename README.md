# uu-course-project
Project within Advanced Scientific Programming with Python 2024

For the project I aim to prepare the code for my current study to be ready for official publishing on GitHub when the manuscript is submitted.

The study is based on a model of a so-called integrated power-to-gas system which is the content of the integrated_p2g package. Hydrogen is produced from electricity and converted to methane using carbon dioxide from biogas from co-digestion. The by-products, heat and oxygen, are utilized in a wastewater treatment plant to reduce the energy use of wastewater treatment. Currently, the project consists of several modules within the package. The main model can be found in 'simulation.py', within which calls to functions in the other modules are made.

The following changes are to be made to the project from its previous condition:
- Improve documentation to a consistent standard similar to numpydoc
- Further modularization of the simulation script and implement classes for variable saving to improve readability
- Performance improvements through increased NumPy use and optimization of for loops
- Removal of unused segments

Before optimization the total time for one annual hourly simulation was around 56 seconds. Most of the time consumption origins in a linear optimization program, the improvement of which is beyond the scope of this project, but smaller gains can likely be made in the surrounding code.
