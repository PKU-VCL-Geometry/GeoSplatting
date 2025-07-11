bash eval.sh tsir-arm tsir_arm && \
python tests/model/test_geosplat_defer.py reliteval --load outputs/geosplat_defer/tsir_arm/task.py --skip-nvs --render-rlit --render-albedo
bash eval.sh tsir-ficus tsir_ficus && \
python tests/model/test_geosplat_defer.py reliteval --load outputs/geosplat_defer/tsir_ficus/task.py --skip-nvs --render-rlit --render-albedo
bash eval.sh tsir-hotdog tsir_hotdog && \
python tests/model/test_geosplat_defer.py reliteval --load outputs/geosplat_defer/tsir_hotdog/task.py --skip-nvs --render-rlit --render-albedo
bash eval.sh tsir-lego tsir_lego && \
python tests/model/test_geosplat_defer.py reliteval --load outputs/geosplat_defer/tsir_lego/task.py --skip-nvs --render-rlit --render-albedo