FROM python:3.8

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ARG DockerHOME

RUN mkdir -p ${DockerHOME}
RUN pip install --upgrade pip

WORKDIR ${DockerHOME}
COPY requirements.txt requirements.txt
RUN sed '12,13d' requirements.txt > requirementsapp.txt
RUN pip install -r requirementsapp.txt

EXPOSE 8000
COPY . .
