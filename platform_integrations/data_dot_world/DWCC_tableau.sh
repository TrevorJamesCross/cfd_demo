sudo docker run -it --rm \
--mount type=bind,source=${PWD}/tab_tmp,target=/dwcc-output \
--mount type=bind,source=${PWD}/tab_tmp,target=/app/log \
datadotworld/dwcc:latest catalog-tableau \
--account=aebs-dev \
--tableau-api-base-url=http://tableau.aebsinternal.com/api/3.10/ \
--tableau-username=<username> \
--tableau-password=<password> \
--tableau-project=<project> \
--tableau-site="https://region.online.tableau.com/#/site/AEBS/home" \
--name=tableau_catalog \
--output=/dwcc-output