FROM tensorflow/tensorflow
ENV GRPC_VERBOSITY=debug
ENV SERVERNAME=server-0-dockemu
ENV SERVERPORT=8080

# Copy wrapper and python script
COPY server.py /server.py
COPY wrapper.sh /wrapper.sh
RUN chmod +x /wrapper.sh

# Update pip
RUN pip3 install --upgrade pip
# Install flwr
RUN pip3 install flwr

CMD /wrapper.sh >> /var/log/dockemu/server.log
EXPOSE 8080
