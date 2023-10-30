FROM bitnami/pytorch:2.0.1-debian-11-r11
#FROM gcr.io/crowdworks-aiml-ops/image-retrieval:v1

COPY ./ ./

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

USER root
RUN useradd kserve -m -u 1000 -d /home/kserve
USER 1000
