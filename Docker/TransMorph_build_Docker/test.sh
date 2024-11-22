#!/usr/bin/env bash
bash ./build.sh
#docker load --input reg_model.tar.gz
#jchen245/transmorph_brain_mri_registration:transmorph_brain_mri_t1_v0
docker run --rm  \
        --ipc=host \
        --memory 256g \
        --mount type=bind,source=/Izziv_AMS/Docker/TransMorph_build_Docker/test_dataset.json,target=/input_dataset.json \
        --mount type=bind,source=/Izziv_AMS/Docker/TransMorph_build_Docker/configs_registration.json,target=/configs_registration.json \
        --mount type=bind,source=/Izziv_AMS/Docker/TransMorph_build_Docker/input,target=/input \
        --mount type=bind,source=/Izziv_AMS/Docker/TransMorph_build_Docker/output,target=/output \
        transmorph_brain_mri_t1

