sudo docker run -it --rm \
--mount type=bind,source=${PWD}/tmp_dir,target=/dwcc-output \
--mount type=bind,source=${PWD}/tmp_dir,target=/app/log \
datadotworld/dwcc-cert:latest catalog-tableau \
--account=aebs-dev \
--tableau-api-base-url=<url> \
--tableau-site=<site> \
--tableau-username=<user> \
--tableau-password=<password> \
--tableau-project=<project> \
--name=tableau_catalog \
--output=/dwcc-output