#!/bin/bash

source /scratch/cluster/ankgarg/gqa/gqa_venv/bin/activate
cd ../src
python -m graphQA.main --mode=train $@ @../condor/sample_config.txt
python -m graphQA.main --mode=eval $@ @../condor/sample_config.txt
cd ../condor