sudo docker run -it --rm \
--mount type=bind,source=${PWD}/sf_tmp,target=/dwcc-output \
--mount type=bind,source=${PWD}/sf_tmp,target=/app/log \
datadotworld/dwcc:latest catalog-snowflake \
--account=aebs-dev \
--server=aebs.us-east-1.snowflakecomputing.com \
--user=trevor.cross \
--password=Trevor!=720 \
--database=CFB_DEMO \
--all-schemas \
--role=SYSADMIN \
--name=sf_catalog \
--output=/dwcc-output