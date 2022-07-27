sudo docker run -it --rm \
--mount type=bind,source=${PWD}/tmp_dir,target=/dwcc-input \
--mount type=bind,source=${PWD}/tmp_dir,target=/dwcc-output \
datadotworld/dwcc:latest catalog-dbt \
--account=aebs-dev \
--profile-file=/dwcc-input/profiles.yml \
--profile=aebs_trevor \
--target=dev \
--artifact-directory=/dwcc-input \
--name=dbt_catalog \
--output=/dwcc-output
