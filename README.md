General introduction: 
2 scripts

Gather data script: 

Must download Sentinel 2 tile grid kml file from here and keep in the same directory: https://sentiwiki.copernicus.eu/web/s2-products
The file must have its original filename - "S2A_OPER_GIP_TILPAR_MPC__20151209T095117_V20150622T000000_21000101T000000_B00.kml"

Runtime variables. 

jpegs. slurm.

Docker/requirements

Output/directory creation.


Build model script: 
Not included in requirements file - but need standard data science libraries including tensorflow and keras.
Need the output dataset. 
Assumes a directory called 'jpeg_files' with all the jpegs saved. 

Output/directory creation.