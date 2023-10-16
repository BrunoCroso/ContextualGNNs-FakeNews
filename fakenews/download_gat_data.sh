#!/usr/bin/env bash

cd produced_data


echo Connecting to the server
scp  -T astroturfer@10.100.54.29:"/home/astroturfer/astroturfing/fakenews/produced_data/gat_dataset0.tar.gz \
                                  /home/astroturfer/astroturfing/fakenews/produced_data/gat_dataset1.tar.gz \
				  /home/astroturfer/astroturfing/fakenews/produced_data/gat_dataset2.tar.gz \
				  /home/astroturfer/astroturfing/fakenews/produced_data/gat_dataset3.tar.gz \
				  /home/astroturfer/astroturfing/fakenews/produced_data/gat_dataset4.tar.gz \
				  /home/astroturfer/astroturfing/fakenews/produced_data/gat_dataset5.tar.gz \
				  /home/astroturfer/astroturfing/fakenews/produced_data/gat_dataset6.tar.gz \
				  /home/astroturfer/astroturfing/fakenews/produced_data/gat_dataset7.tar.gz \
				  /home/astroturfer/astroturfing/fakenews/produced_data/gat_dataset8.tar.gz \
				  /home/astroturfer/astroturfing/fakenews/produced_data/gat_dataset9.tar.gz" .
								  
echo Uncompressing the downloaded files
for i in {0..9}
do
  rm -r gat_dataset$i
  tar -xzf gat_dataset$i.tar.gz -C .
  mv gat_dataset$i/* dataset$i/
done
