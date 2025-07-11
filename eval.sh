python tests/model/test_geosplat.py $1 && \
python tests/model/test_geosplat.py export --load outputs/geosplat/$2/task.py --output exports/$2.pkl && \
python tests/model/test_geosplat_mc.py $1 && \
python tests/model/test_geosplat_mc.py export --load outputs/geosplat_mc/$2/task.py --output exports/$2.mc.pkl && \
python tests/model/test_geosplat_defer.py $1 --trainer.no-hold-after-train
