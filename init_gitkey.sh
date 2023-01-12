#!/bin/bash
eval "$(ssh-agent -s)"
ssh-add $HOME/.ssh/id_ed25519_euler_remote