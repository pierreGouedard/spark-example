# -- Software Stack Version
SPARK_VERSION="3.1.2"
HADOOP_VERSION="2.7"

# -- Building the images
docker build \
  -f dockerfiles/base.Dockerfile \
  -t base .

docker build \
  --build-arg spark_version="${SPARK_VERSION}" \
  --build-arg hadoop_version="${HADOOP_VERSION}" \
  -f dockerfiles/spark-base.Dockerfile \
  -t spark-base .

docker build \
  -f dockerfiles/spark-master.Dockerfile \
  -t spark-master .

docker build \
  -f dockerfiles/spark-worker.Dockerfile \
  -t spark-worker .
