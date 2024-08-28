#!/bin/bash

# Create and move into the interproscan directory
mkdir -p my_interproscan
cd my_interproscan

# Download InterProScan and its checksum file
wget https://ftp.ebi.ac.uk/pub/software/unix/iprscan/5/5.69-101.0/interproscan-5.69-101.0-64-bit.tar.gz
wget https://ftp.ebi.ac.uk/pub/software/unix/iprscan/5/5.69-101.0/interproscan-5.69-101.0-64-bit.tar.gz.md5

# Verify the checksum to ensure the download was successful
md5sum -c interproscan-5.69-101.0-64-bit.tar.gz.md5

# Check if the checksum is correct
if [ $? -eq 0 ]; then
    echo "Checksum is OK, proceeding with extraction."
else
    echo "Checksum failed. Please re-download the file."
    exit 1
fi

# Extract the tar.gz file
tar -pxvzf interproscan-5.69-101.0-64-bit.tar.gz

# Navigate into the extracted InterProScan directory
cd interproscan-5.69-101.0

# Run the setup script
python3 setup.py -f interproscan.properties

# Update the package list and install Java
sudo apt update
sudo apt install -y default-jdk default-jre libc6-i386 libc6-x32 libxi6 libxtst6

# Download and install Java 17
wget https://download.oracle.com/java/17/latest/jdk-17_linux-x64_bin.deb
sudo dpkg -i jdk-17_linux-x64_bin.deb

# Verify Java installation
java -version

echo "InterProScan installation completed successfully."
