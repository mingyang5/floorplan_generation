# floorplan_generation

# MSD: A Benchmark Dataset for Floor Plan Generation of Building Complexes

- Add tldr;
- Add link to paper, project page, data (both Kaggle and 4TU.ResearchData);

## Modified swiss dwellings datasets

- Add introduction to dataset;
- Add text about the data formats in MSD (explain structure of Kaggle etc.);
- Add links to where the data can be downloaded;
- Add links to Jupyter notebook explaining how to load and use the data.

## Floor plan generation

- Quick recap on two approaches (including results figure);
- Setup environment;
- How to train on MHD;
- How to train on U-Net.

## References

- Add references (if needed);
- Add acknowledgements (if need).


# Tips:
```shell
cd ./experiments/house-diffusion/model/datasets
gdown h10DLrJz4ubM0Ujs6ZQI1_AhOuPDUpbhSQ
tar -xzvf modified_swiss_dwellings.tar.gz

# Training
cd experiments/house-diffusion/model/scripts
python image_train.py --dataset modified_swiss_dwellings --batch_size 32 --set_name train --timeout 36:00:00 --save_interval 2000 --test_interval 1000 --use_wall_self_attention true
```