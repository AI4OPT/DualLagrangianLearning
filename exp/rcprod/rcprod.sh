#!/bin/bash
cd <path_to_project_root>
if [[ $SLURM_NODEID == "0" ]]; then
    ml_method="DLL"
elif [[ $SLURM_NODEID == "1" ]]; then
    ml_method="DC3"
else
    ml_method="XXX"
fi

julia --project=. -t12 exp/rcprod/run.jl ${SLURM_ARRAY_TASK_ID} 42 data/rcprod/ > exp/rcprod/logs/rcprod_n${SLURM_ARRAY_TASK_ID}_s42_vb0_${ml_method}.log 2>&1
