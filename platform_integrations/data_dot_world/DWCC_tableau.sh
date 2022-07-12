sudo docker run -it --rm \
--mount type=bind,source=${PWD}/tmp_dir,target=/dwcc-output \
--mount type=bind,source=${PWD}/tmp_dir,target=/app/log \
datadotworld/dwcc-cert:latest catalog-tableau \
--account= \
--tableau-api-base-url= \
--tableau-site=AEBS \
--tableau-username= \
--tableau-password= \
--tableau-project="CFB Dashboards" \
--name=tableau_catalog \
--output=/dwcc-output