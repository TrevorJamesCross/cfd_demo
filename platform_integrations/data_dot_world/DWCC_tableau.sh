sudo docker run -it --rm \
--mount type=bind,source=${PWD}/tmp_dir,target=/dwcc-output \
--mount type=bind,source=${PWD}/tmp_dir,target=/app/log \
datadotworld/dwcc-cert:latest catalog-tableau \
--account=aebs-dev \
--tableau-api-base-url=https://tableau.aebsinternal.com/api/3.15 \
--tableau-site=AEBS \
--tableau-username=trevor.cross \
--tableau-password=iplayedwithmydogtoday \
--tableau-project="CFB Dashboards" \
--name=tableau_catalog \
--output=/dwcc-output