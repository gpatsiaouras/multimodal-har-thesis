#!/bin/bash

#==============================================================================
#title           :download_datasets
#description     :This script automatically downloads the datasets needed for the project
#author          :Giorgos Patsiaouras
#organization    :Maastricht University
#date            :20201114
#version         :1.0.0
#usage           :download_datasets
#notes           :This script requires wget,unzip
#==============================================================================

# Colors
RED='\033[0;31m'
YELLOW='\033[0;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default settings
DATASET_DIR=$(dirname $0)/datasets
UTD_MHAD_DIR=utd_mhad
DOWNLOAD_UTD_MHAD=true
DOWNLOAD_MMACT=true

function download_utd_mhad () {
  echo
  echo -e "${CYAN}Downloading UTD-MHAD:${NC}"
  echo -e "Inertial Data 1/4..."
  wget http://www.utdallas.edu/~kehtar/UTD-MAD/Inertial.zip -P "$DATASET_DIR"
  echo -e "Depth Data 2/4..."
  wget http://www.utdallas.edu/~kehtar/UTD-MAD/Depth.zip -P "$DATASET_DIR"
  echo -e "Skeleton Data 3/4..."
  wget http://www.utdallas.edu/~kehtar/UTD-MAD/Skeleton.zip -P "$DATASET_DIR"
  echo -e "RGB Videos 4/4..."
  wget http://www.utdallas.edu/~kehtar/UTD-MAD/RGB.zip -P "$DATASET_DIR"
}

function extract_utd_mhad () {
  echo
  echo -e "${CYAN}Extracting UTD-MHAD:${NC}"
  mkdir -p "$DATASET_DIR/$UTD_MHAD_DIR"
  (cd "$DATASET_DIR" && unzip Depth.zip -d $UTD_MHAD_DIR/)
  (cd "$DATASET_DIR" && unzip Inertial.zip -d $UTD_MHAD_DIR/)
  (cd "$DATASET_DIR" && unzip RGB.zip -d $UTD_MHAD_DIR/)
  (cd "$DATASET_DIR" && unzip Skeleton.zip -d $UTD_MHAD_DIR/)

}

function clean_utd_mhad () {
  (cd "$DATASET_DIR" && rm -f *.zip)
}

function download_mmact () {
  echo
  echo -e "${YELLOW} Mmact has to be downloaded manually! ${NC}"
}

# Execute main program
if [ "$DOWNLOAD_UTD_MHAD" == "true" ]; then
  if [ -d "$DATASET_DIR/$UTD_MHAD_DIR" ]; then
    echo -e "${YELLOW}[Warning]${NC}"
    echo -en "\tIt seems like utd-mhad is already downloaded under the directory: "
    echo -e "${CYAN}$DATASET_DIR/$UTD_MHAD_DIR${NC}"
    echo -e "\tRemove it if you want to download the dataset either way"
    exit 0
  fi
  download_utd_mhad
  extract_utd_mhad
  clean_utd_mhad
fi

if [ "$DOWNLOAD_MMACT" == "true" ]; then
  download_mmact
fi


