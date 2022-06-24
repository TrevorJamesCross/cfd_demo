sudo docker run -it --rm \
--mount type=bind,source=${PWD}/tab_tmp,target=/dwcc-output \
--mount type=bind,source=${PWD}/tab_tmp,target=/app/log \
datadotworld/dwcc:latest catalog-tableau \
--account=aebs-dev \
--tableau-api-base-url=<url> \
--tableau-site=<site> \
--tableau-username=<user> \
--tableau-password=<password> \
--tableau-project="CFB Dashboards" \
--name=tableau_catalog \
--output=/dwcc-output