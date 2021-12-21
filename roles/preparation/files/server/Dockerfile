FROM tensorflow/tensorflow
ENV GRPC_VERBOSITY=debug

# wget
RUN apt-get update && apt-get install iproute2 iputils-ping iptables nmap vim -y

COPY server.py /server.py
COPY wrapper.sh /wrapper.sh

# update pip
RUN pip3 install --upgrade pip
# flwr tensorflow flwr torch torchvision
RUN pip3 install flwr


RUN chmod +x /server.py
RUN chmod +x /wrapper.sh

CMD /wrapper.sh > /var/log/dockemu/coap-server.log
EXPOSE 8080
