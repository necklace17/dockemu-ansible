# Get server logs from container
docker logs $(docker ps -aqf "name=coapcont-server-0") -f

# Get client logs from container
docker logs $(docker ps -aqf "name=coapcont-client-0") -f

# Get client logs from file
tail -f ~/src/dockemu/coap-experiment/logs/coapcont-0/coap-client.log

# Get server logs from file
tail -f ~/src/dockemu/coap-experiment/logs/coapcont-server-0/coap-server.log

# Connect to container
docker exec -it $(docker ps -aqf "name=coapcont-client-0") /bin/bash

# Remove container
docker rm $(docker ps -aqf "name=coapcont-client-6")