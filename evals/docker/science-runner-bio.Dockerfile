FROM zhili-science-runner:latest

# Python 3.11-compatible scientific biology/chemistry profile.  Versions are
# pinned from official PyPI metadata so re-evaluations remain reproducible.
USER root
RUN pip install --no-cache-dir \
    rdkit==2026.3.5 \
    scanpy==1.11.5 \
    pgmpy==1.1.2 \
    biopython==1.88 \
    msprime==1.3.4 \
    scikit-allel==1.3.13 \
    cobra==0.32.1 \
    statsmodels==0.15.0

USER 65534:65534
WORKDIR /tmp
