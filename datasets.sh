# UNet Related
# cd /tmp/mingyang/datasets/house
# gdown 1fY_Fh7ZVSDHEEEVjFY8_DzzX86FcDHvr
# tar -zxvf msd_unet_processed_dataset.tar.gz
# rm -rf msd_unet_processed_dataset.tar.gz
# cd /home/mingyang/workspace/layout/floorplan/experiments/unet
# ln -s /tmp/mingyang/datasets/house/msd_unet_processed_dataset dataset_processed

cd /tmp/mingyang/datasets/house                             # reminder: change path
gdown 10DLrJz4ubM0Ujs6ZQI1_AhOuPDUpbhSQ                     # modified_swiss_dwellings.tar.gz
tar -zxvf modified_swiss_dwellings.tar.gz
rm -rf modified_swiss_dwellings.tar.gz

gdown 1N-HfrwoiIPAvwd9duUaQSVTyIO2O817Z                     # Modified Swiss Dwellings.zip
unzip 'Modified Swiss Dwellings.zip'
mv mds_V2_5.372k.csv modified-swiss-dwellings-v2
rm -rf 'Modified Swiss Dwellings.zip'

cd /tmp/mingyang/datasets/house/modified-swiss-dwellings-v2/test    # reminder: change path
gdown 1VrRpNlSk8TGMnRFPaVkcV9Ewq-CkMdC6
gdown 1ePuu10ee7kl41o82FGoGdPPIkFpuJEho
tar -zxvf graph_pred_n_corners.tar.gz
tar -zxvf graph_pred.tar.gz
rm -rf graph_pred_n_corners.tar.gz
rm -rf graph_pred.tar.gz


cd /home/mingyang/workspace/layout/floorplan/experiments/house-diffusion/model/datasets     # reminder: change path
ln -s /tmp/mingyang/datasets/house/modified_swiss_dwellings ./
ln -s /tmp/mingyang/datasets/house/modified-swiss-dwellings-v2 ./

cd /tmp/mingyang/datasets/house
mkdir ckpts
cd /home/mingyang/workspace/layout/floorplan/experiments/house-diffusion/model/scripts
ln -s /tmp/mingyang/datasets/house/ckpts ./
