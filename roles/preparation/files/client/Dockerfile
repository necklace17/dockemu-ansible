FROM tensorflow/tensorflow
ENV GRPC_VERBOSITY=debug
# wget
RUN apt-get update && apt-get install net-tools iproute2 iputils-ping iptables nmap vim -y

COPY client.py /client.py
COPY pickle_cifar /pickle_cifar
COPY wrapper.sh /wrapper.sh

# update pip
RUN pip3 install --upgrade pip
# flwr tensorflow flwr torch torchvision
RUN pip3 install flwr

RUN chmod +x /client.py
RUN chmod +x /wrapper.sh

CMD sleep 60 \
    && /wrapper.sh >> /var/log/dockemu/coap-client.log
EXPOSE 8080
