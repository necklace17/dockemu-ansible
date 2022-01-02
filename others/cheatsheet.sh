# Get server logs from container
docker logs $(docker ps -aqf "name=fliot-server-0") -f

# Get client logs from container
docker logs $(docker ps -aqf "name=fliot-client-0") -f

# Get client logs from file
tail -f ~/src/dockemu/coap-experiment/logs/fliot-0/client.log

# Get server logs from file
tail -f ~/src/dockemu/coap-experiment/logs/fliot-server-0/server.log

# Connect to container
docker exec -it $(docker ps -aqf "name=fliot-client-0") /bin/bash

# Remove container
docker rm $(docker ps -aqf "name=fliot-client-6")

# Remove all container
docker rm -f $(docker ps -a -q)