#!/bin/bash
sudo docker rm -f $(docker ps -a -q) && sudo ansible-playbook dockemu.yml -t cleanup