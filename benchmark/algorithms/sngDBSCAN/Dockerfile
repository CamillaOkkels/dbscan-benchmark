FROM dbscan-benchmark

RUN pip install Cython
RUN git clone https://github.com/jenniferjang/subsampled_neighborhood_graph_dbscan && cd subsampled_neighborhood_graph_dbscan && python3 setup.py install

RUN python3 -c 'import SubsampledNeighborhoodGraphDBSCAN as dbscan'