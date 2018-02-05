FROM continuumio/miniconda3:latest
MAINTAINER Francesco Ceccon

COPY requirements.txt /src/requirements.txt
RUN pip install --no-cache-dir -r /src/requirements.txt

COPY suspect /src/suspect
COPY setup.py /src/setup.py
COPY scripts /src/scripts
RUN pip install /src

ENTRYPOINT ["model_summary.py"]
