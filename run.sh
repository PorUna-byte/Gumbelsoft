#!/bin/bash
export model_dir=/path/to/your/model/directory
export experiments_path=/path/to/experiment_results/directory
export PYTHONPATH=$PYTHONPATH:/path/to/project/directory
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

bash table1.sh
bash table2.sh
bash table3.sh
bash table4.sh

bash case_study1.sh
bash case_study2.sh


