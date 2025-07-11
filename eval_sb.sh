bash eval.sh ball ball && \
python tests/model/test_geosplat_defer.py nvseval --load outputs/geosplat_defer/ball/task.py
bash eval.sh car car && \
python tests/model/test_geosplat_defer.py nvseval --load outputs/geosplat_defer/car/task.py
bash eval.sh coffee coffee && \
python tests/model/test_geosplat_defer.py nvseval --load outputs/geosplat_defer/coffee/task.py
bash eval.sh helmet helmet && \
python tests/model/test_geosplat_defer.py nvseval --load outputs/geosplat_defer/helmet/task.py
bash eval.sh teapot teapot && \
python tests/model/test_geosplat_defer.py nvseval --load outputs/geosplat_defer/teapot/task.py
bash eval.sh toaster toaster && \
python tests/model/test_geosplat_defer.py nvseval --load outputs/geosplat_defer/toaster/task.py