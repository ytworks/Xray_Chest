#! /usr/bin/bash

echo '================== Start AE ===================='
python AE.py

echo '================== Start CAE ==================='
python CAE.py

echo '================== Start DECAE ================='
python DECAE.py

echo '================== Start TEMPLATE =============='
python dnn_template.py

echo '================== Start DAE =================='
python dae.py

echo '================== Start Simple DAE ==========='
python simple_dae_template.py
