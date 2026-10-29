# Environment for Open_Duck_reference_motion_generator (placo 0.6.3 has only a manylinux x86_64 wheel).
# Build: docker build --platform linux/amd64 -f configs/refgen.Dockerfile -t duck-refgen configs
FROM --platform=linux/amd64 python:3.10.12-slim
COPY requirements-refgen.txt /tmp/requirements-refgen.txt
RUN pip install --no-cache-dir -r /tmp/requirements-refgen.txt
