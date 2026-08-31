FROM zhili-science-runner:latest

USER root
RUN pip install --no-cache-dir \
    qiskit==2.5.2 \
    pymatching==2.4.0

RUN pip install --no-cache-dir qiskit-aer==0.17.2

RUN pip install --no-cache-dir \
    qiskit-nature==0.8.0 \
    pyscf==2.14.0

USER 65534:65534
WORKDIR /tmp
