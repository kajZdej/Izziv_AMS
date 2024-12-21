#!/usr/bin/env bash
docker pull jchen245/transmorph_brain_mri_registration:transmorph_brain_mri_t1_v2

docker run --rm  \
        --ipc=host \
        --memory 256g \
        --mount type=bind,source=/Izziv_AMS/Docker/test_dataset.json,target=/input_dataset.json \
        --mount type=bind,source=/Izziv_AMS/Docker/configs_registration.json,target=/configs_registration.json \
        --mount type=bind,source=Izziv_AMS/Docker/input,target=/input \
        --mount type=bind,source=Izziv_AMS/Docker/onput,target=/output \
        jchen245/transmorph_brain_mri_registration:transmorph_brain_mri_t1_v2

