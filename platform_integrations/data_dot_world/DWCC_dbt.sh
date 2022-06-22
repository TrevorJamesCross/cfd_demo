sudo docker run -it --rm \
--mount type=bind,source=${PWD}/dbt_tmp,target=/dwcc-input \
--mount type=bind,source=${PWD}/dbt_tmp,target=/dwcc-output \
datadotworld/dwcc:latest catalog-dbt \
--account=aebs-dev \
--dbt-profile-file=/dwcc-input/profiles.yml \
--profile=aebs_trevor \
--target=dev \
--data-dir=/dwcc-input \
--name=dbt_catalog \
--output=/dwcc-output