bash eval.sh s4r-air s4r_air && \
python tests/model/test_geosplat_defer.py reliteval --load outputs/geosplat_defer/s4r_air/task.py --skip-nvs --render-rlit --render-albedo
bash eval.sh s4r-chair s4r_chair && \
python tests/model/test_geosplat_defer.py reliteval --load outputs/geosplat_defer/s4r_chair/task.py --skip-nvs --render-rlit --render-albedo
bash eval.sh s4r-hotdog s4r_hotdog && \
python tests/model/test_geosplat_defer.py reliteval --load outputs/geosplat_defer/s4r_hotdog/task.py --skip-nvs --render-rlit --render-albedo
bash eval.sh s4r-jugs s4r_jugs && \
python tests/model/test_geosplat_defer.py reliteval --load outputs/geosplat_defer/s4r_jugs/task.py --skip-nvs --render-rlit --render-albedo