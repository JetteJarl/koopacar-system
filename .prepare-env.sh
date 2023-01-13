#!/bin/bash
source ~/.bashrc

# Install missing dependencies
rosdep install -i --from-path src --rosdistro foxy -y

# Build packages unless an option is specified
if [ "$1" != "--no-build" ]; then
    colcon build --packages-select perception localization
fi

# source package and nodes
. install/setup.bash

colcon build

source install/local_setup.bash

$@