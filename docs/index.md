# LAMTA

[![SPASSOv2.0 paper](https://img.shields.io/badge/SPASSOv2.0-paper-blue)](https://doi.org/10.1175/JTECH-D-24-0071.1)
[![SPEC 0 — Minimum Supported Dependencies](https://img.shields.io/badge/SPEC-0-green?labelColor=%23004811&color=%235CA038)](https://scientific-python.org/specs/spec-0000/)

**LAMTA** (**LA**grangian **M**anifolds **T**racking **A**lgorithm) is a Python code designed to compute numerical particle trajectories within ocean current 2-D fields and to derive a range of Lagrangian diagnostics for detecting and tracking (sub)mesoscale ocean features.

---

## Manuscript and code

A dedicated manuscript is currently under development to describe the theoretical background, numerical methods, and applications of LAMTA in detail.

The source code is openly developed and maintained, with a focus on reproducible Lagrangian analyses in oceanography.

LAMTA and its Lagrangian diagnostics are also described in the context of SPASSOv2.0 in:

Rousselet, L., d’Ovidio, F., Izard, L., Della Penna, A., Petrenko, A., Barrillon, S., Nencioli, F., and Doglioli, A. (2025).
*A Software Package for an Adaptive Satellite-Based Sampling for Oceanographic Cruises (SPASSOv2.0): Tracking Fine-Scale Features for Physical and Biogeochemical Studies*. *Journal of Atmospheric and Oceanic Technology*, **42**(8), 979–990. https://doi.org/10.1175/JTECH-D-24-0071.1

---

## Using LAMTA with the examples

Tutorials and example workflows are provided as Jupyter notebooks in the separate [**LAMTA Examples repository**](https://github.com/OceanCruises/LAMTA_examples) and are documented in the [**LAMTA Examples documentation**](https://lamta-examples.readthedocs.io/).

If your goal is to run the tutorial notebooks, follow the installation instructions provided in the LAMTA Examples documentation.

The dedicated `lamta_examples` environment contains the notebook and plotting dependencies, and installing `LAMTA_examples` also installs LAMTA automatically.

The examples include:

- Initialising and advecting particles in analytical flows
- Working with `ParticleSet`
- Configuring advection schemes and parameters
- Handling periodic boundary conditions
- Applying LAMTA to realistic ocean current fields

---

## Installation

LAMTA is currently **not distributed as a packaged release on PyPI**. As a result, the command:

```bash
pip install lamta
```

will **not work** at this stage.

LAMTA must therefore be installed from source.

### Step 1 — Clone LAMTA

```{warning}
If you plan to contribute to the code or documentation, we recommend forking the repository first and cloning your fork instead.
```

```bash
# Users
git clone https://github.com/OceanCruises/LAMTA

# Contributors
# git clone https://github.com/<your-username>/LAMTA

cd LAMTA
```

### Step 2 — Create and activate a Python environment

Using **Conda**:

```bash
conda create -n lamta python=3.12
conda activate lamta
```

### Step 3 — Install LAMTA

From the root of the cloned repository:

```bash
python -m pip install -e .
```

This installs LAMTA and its required Python dependencies in editable mode. Any modification to the local source code will therefore be immediately reflected when importing LAMTA.

You can verify the installation with:

```bash
python -c "import lamta; print('LAMTA import OK')"
```

---

## Developing LAMTA with the examples

If you are developing LAMTA itself and want to test your changes directly in the example notebooks, we recommend keeping both repositories side-by-side:

```text
lamta_dev/
├─ LAMTA/           # core package
└─ LAMTA_examples/  # notebooks
```

We recommend using Visual Studio Code and opening the parent folder (for example `lamta_dev/`) so both repositories are available in the same workspace.

- [VS Code](https://code.visualstudio.com/)
- [VS Code workspaces](https://code.visualstudio.com/docs/editor/multi-root-workspaces)

Create the examples environment following the LAMTA Examples documentation, then install your local LAMTA checkout into that environment:

```bash
conda activate lamta_examples
python -m pip install -e ../LAMTA
```

This replaces the GitHub-installed version of LAMTA with your local editable checkout. Changes made to the local LAMTA source code will then be immediately available in the notebooks.

---

## Future packaging

A standard packaged release (e.g. via PyPI) is planned for the future. Until then, installing LAMTA from source as described above is the supported and recommended approach.

## Funding and support

The development of LAMTA has been supported by the following organisations:

- **LOCEAN** — *Laboratoire d’Océanographie et du Climat: Expérimentations et Approches Numériques*, Sorbonne University
- **CNES** — *Centre National d’Études Spatiales*

## Acknowledgements

The development of LAMTA has benefited from the remarkable coding efforts and scientific contributions of:

- Louise Rousselet
- Francesco d’Ovidio
- Gina Fifani
