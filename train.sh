#!/bin/sh

# usual
#python manage.py train --tub ../sergem_tub/,../sergem_tub_flip/,../sergem_tub_sunlight/,../sergem_tub_sunlight_flip/,../tubs_jun30_cleaned/tub_10_18-05-31/,../tubs_jun30_cleaned/tub_13_18-05-31/,../tubs_jun30_cleaned/tub_15_18-05-31/,../tubs_jun30_cleaned/tub_5_18-05-31/,../tubs_jun30_cleaned/tub_9_18-05-31/ --model ./models/sergem_20190604_w_flip_w_sunlight_w_jun30cleaned >> ./logs/sergem_20190604_w_flip_w_sunlight_w_jun30.log 2>&1

# with bird view
python manage.py train --tub ../sergem_tub/,../sergem_tub_flip/,../sergem_tub_sunlight/,../sergem_tub_sunlight_flip/,../tubs_jun30_cleaned/tub_10_18-05-31/,../tubs_jun30_cleaned/tub_13_18-05-31/,../tubs_jun30_cleaned/tub_15_18-05-31/,../tubs_jun30_cleaned/tub_5_18-05-31/,../tubs_jun30_cleaned/tub_9_18-05-31/ --model ./models/sergem_20190626_bird_transform --model_class donkeycar.parts.keras.KerasLinear  >> ./logs/sergem_20190626_bird_transform_.log 2>&1
