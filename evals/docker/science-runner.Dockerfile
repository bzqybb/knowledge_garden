# Docker Hub authentication is not reliably reachable on every deployment
# network.  This is the official Dev Containers Python image published to
# Microsoft's MCR and remains Python 3.11 / Debian Bookworm.
FROM mcr.microsoft.com/devcontainers/python:3.11-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp/.cache \
    MPLCONFIGDIR=/tmp/matplotlib

RUN pip install --no-cache-dir \
    numpy==2.2.6 \
    scipy==1.15.3 \
    sympy==1.14.0 \
    matplotlib==3.10.3 \
    networkx==3.4.2 \
    pandas==2.3.0 \
    scikit-learn==1.7.0 \
    z3-solver==4.15.1.0

USER 65534:65534
WORKDIR /tmp

CMD ["python", "-I", "-"]
