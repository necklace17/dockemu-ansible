FROM python:3.8
ENV GRPC_VERBOSITY=debug

# wget
RUN apt-get update && apt-get install iproute2 iputils-ping iptables nmap vim -y

COPY server_greeter.py /server_greeter.py
COPY helloworld_pb2.py /helloworld_pb2.py
COPY helloworld_pb2_grpc.py /helloworld_pb2_grpc.py
COPY wrapper.sh /wrapper.sh

# update pip
RUN pip3 install --upgrade pip
# flwr tensorflow flwr torch torchvision
RUN pip3 install websockets ifaddr grpcio grpcio-tools


RUN chmod +x /server_greeter.py
RUN chmod +x /helloworld_pb2.py
RUN chmod +x /helloworld_pb2_grpc.py
RUN chmod +x /wrapper.sh

#CMD ["python", "./server_greeter.py"]
CMD /wrapper.sh > /var/log/dockemu/coap-server.log
EXPOSE 50051

#FROM alpine:latest as builder
## ENV libcoap_dir=/libcoap
## COPY . $libcoap_dir
## WORKDIR $libcoap_dir
#RUN apk add --update \
#    bash    \
#    autoconf \
#    automake  \
#    pkgconf \
#    libtool \
#    build-base \
#    git\
#  && rm -rf /var/cache/apk/* \
#  && ls
#RUN git clone https://github.com/obgm/libcoap.git
#WORKDIR /libcoap
#RUN ./autogen.sh --clean && ./autogen.sh \
#  && ./configure --disable-tests\
#       --disable-documentation --enable-examples\
#       --disable-dtls --disable-shared \
#  && make \
#  && cp /libcoap/examples/coap-server /tmp/. \
#  && cp /libcoap/examples/coap-client /tmp/. \
#  && ls /tmp
#
#
#FROM alpine:latest
#COPY --from=builder /tmp/coap-* /libcoap/
#CMD /libcoap/coap-server -v 7 > /var/log/dockemu/coap-server.log
#EXPOSE 5684
#
#STOPSIGNAL SIGTERM
#
#
## make it multi-stage to build the library and the application fisrt, and then copy it in Apline so that we can execute a huge number of instance and save space and
## RAM and CPU resources
