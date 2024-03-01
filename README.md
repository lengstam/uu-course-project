# uu-course-project
Project within Advanced Scientific Programming with Python 2024

For the project I aim to prepare the code for my current study to be ready for official publishing on GitHub when the manuscript is submitted.

The study is based on a model of a so-called integrated power-to-gas system which is the content of the integrated_p2g directory. Hydrogen is produced from electricity and converted to methane using carbon dioxide from biogas from co-digestion. The by-products, heat and oxygen, are utilized in a wastewater treatment plant to reduce the energy use of wastewater treatment. Currently, the project consists of several modules within the directory. The main model can be found in 'simulation.py', within which calls to functions in the other modules are made.

The following changes are to be made to the project from its previous condition:
- Improve documentation to a consistent standard similar to numpydoc.
- Further modularization of the simulation script and implement classes for variable saving to improve readability.
- Performance improvements through increased NumPy use and optimization of for loops. Most of the time consumption origins in a linear optimization program, the improvement of which is beyond the scope of this project, but smaller gains can likely be made in the surrounding code.
- Removal of unused segments.

Project outcome:
- Documentation has been provided within the parameters.py and components.py, as well as a general description of the model in simulation.py.
- All calculations are now done with NumPy instead of with Pandas arrays/series or lists. Furthermore, any calculations that could be done outside for loops are now done that way. This has led to some improvement in runtime (appears to be a few seconds), but since the main time consumption comes from other places, the overall gains were small (total runtime is still ~1 minute).
- The size of the main script (simulation.py) has been greatly reduced through increased modularity, implementation of classes and removal of unused code, from almost 1900 rows to around 550, greatly improving readability. 
