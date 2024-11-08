FROM ros:humble

# Set the working directory
WORKDIR /root

# Copy the repository into the container
COPY . /root/mppi_controller

# Update the package list and install dependencies
RUN apt-get update && apt-get install -y \
    ros-humble-rviz2 \
    libogre-1.9-dev

# Allow rviz to find the Ogre libraries
RUN export LD_LIBRARY_PATH=/opt/ros/humble/opt/rviz_ogre_vendor/lib:$LD_LIBRARY_PATH

# Source the ROS environment
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# Set the entrypoint to bash
ENTRYPOINT ["/bin/bash"]


############################################
# To run this properly with RViz use:
# xhost +local:root
# docker run -it --rm \
# --net=host \
# -e DISPLAY=$DISPLAY \
# -v /tmp/.X11-unix:/tmp/.X11-unix \
# dynablox
############################################
