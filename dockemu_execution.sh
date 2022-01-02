#!/bin/bash
sudo ansible-playbook dockemu.yml -t prepare && \
sudo ansible-playbook dockemu.yml -t execute