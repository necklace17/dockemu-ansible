FROM tensorflow/tensorflow
ENV GRPC_VERBOSITY=debug
ENV SERVERNAME=server-0-dockemu
ENV SERVERPORT=8080

# Copy wrapper and python script
COPY client.py /client.py
COPY wrapper.sh /wrapper.sh

# update pip
RUN pip3 install --upgrade pip
RUN pip3 install flwr

RUN chmod +x /wrapper.sh

CMD sleep 60 \
    && /wrapper.sh >> /var/log/dockemu/client.log
EXPOSE 8080
