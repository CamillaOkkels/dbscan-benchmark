FROM dbscan-benchmark

RUN apt update -y && apt install -y libtbb-dev
RUN git clone https://github.com/CamillaOkkels/srrdbscan && cd srrdbscan && python3 setup.py install

RUN python3 -c 'import dbscan_srr as dbscan'
