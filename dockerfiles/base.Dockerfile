# Start from a linux with min spark requirement (java8)
ARG debian_buster_image_tag=8-jre-slim
FROM openjdk:${debian_buster_image_tag}

# Set shared workspace location (spark master & worker)
ARG shared_workspace=/opt/workspace

# Install requirement to use Pyspark (python 3 latest)
RUN mkdir -p ${shared_workspace} && \
    apt-get update -y && \
    apt-get install -y python3 && \
    ln -s /usr/bin/python3.7.0 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

ENV SHARED_WORKSPACE=${shared_workspace}

# Install python3 requirements
RUN apt-get update -q
RUN apt-get install -y --no-install-recommends -qq \
  gcc \
  python3-dev \
  python3-pip \
  libpq-dev \
  g++

COPY dockerfiles/requirements.txt .
RUN pip3 install --upgrade setuptools && \
  pip3 install --upgrade pip && \
  pip3 install --default-timeout=100 --user -r requirements.txt

# CUSTOM -> Download Spacy model for nlp
RUN python3 -m spacy download en_core_web_sm

# Runtime
VOLUME ${shared_workspace}
CMD ["bash"]
