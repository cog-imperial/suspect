FROM continuumio/miniconda3:latest
MAINTAINER Francesco Ceccon

COPY requirements.txt /src/requirements.txt
RUN pip install --no-cache-dir -r /src/requirements.txt

COPY . /src
RUN pip install /src

ENTRYPOINT ["model_summary.py"]