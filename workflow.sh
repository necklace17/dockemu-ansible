#!/bin/bash
sudo ansible-playbook dockemu.yml -t cleanup && \
sudo ansible-playbook dockemu.yml -t prepare && \
sudo ansible-playbook dockemu.yml -t execute -v