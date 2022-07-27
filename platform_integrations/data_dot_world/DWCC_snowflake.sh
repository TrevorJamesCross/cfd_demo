sudo docker run -it --rm \
--mount type=bind,source=${PWD}/tmp_dir,target=/dwcc-output \
--mount type=bind,source=${PWD}/tmp_dir,target=/app/log \
datadotworld/dwcc:latest catalog-snowflake \
--account=aebs-dev \
--server=<server> \
--user=<user> \
--password=<password> \
--database=<database> \
#--all-schemas \
--schema=<schema> \
--role=SYSADMIN \
--name=sf_catalog \
--output=/dwcc-output
