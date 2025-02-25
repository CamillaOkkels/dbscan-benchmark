FROM dbscan-benchmark

RUN apt update -y && apt install -y libtbb-dev
RUN git clone https://maau:fjg31PUf@github.itu.dk/maau/multileveldbscan && cd multileveldbscan && python3 setup.py install #1 #

RUN python3 -c 'import dbscan_srr as dbscan'
