FROM dbscan-benchmark

RUN git clone https://github.com/Markus-Goetz/hpdbscan
RUN apt update && apt install -y cmake libhdf5-dev g++ build-essential
RUN cd hpdbscan && mkdir build && cd build && cmake .. && make