# An example of extension of the jupyter stack 'scipy-notebook'
# with pip modules ('pip install ...') and their system dependancies ('apt-get install -y ...')
ARG JUPYTER_IMG=${JUPYTER_IMG:-jupyter/datascience-notebook}
FROM ${JUPYTER_IMG}

# Add conda channels and install Mamba
RUN conda config --add channels R --system \
  && conda config --add channels bioconda --system \
  && conda config --add channels conda-forge --system \
  && conda install -y -n base -c conda-forge mamba

# WORKDIR ${SHINY_SERVERDIR}
# Build's context should be set in the App directory
COPY binder binder
WORKDIR binder

# Install system dependencies
USER root
RUN if [ -f "apt.txt" ]; then \
  apt-get update -qq; \
  apt-get -y --no-install-recommends install `grep -v "^#" apt.txt | tr '\n' ' '`; \
  fi

# Create a conda environment, use it as a Python kernel and link it to jupyter
RUN if [ -f "environment.yml" ]; then \
  mamba env create -f environment.yml \
  && conda_env=$( grep name environment.yml | awk '{ print $2}' ) \
  && "${CONDA_DIR}/envs/${conda_env}/bin/python" -m ipykernel install --user --name="${conda_env}" \
  && fix-permissions "${CONDA_DIR}" \
  && fix-permissions "/home/${NB_USER}"; \
  fi

USER ${NB_USER}
WORKDIR /home/${NB_USER}

COPY notebooks ./OMERO-FAIRly-notebooks