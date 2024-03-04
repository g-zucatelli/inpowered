CONFIG=$1

python initialize.py -c ${CONFIG}
python data_processor.py -c ${CONFIG}
python trainer.py -c ${CONFIG}
python tester.py -c ${CONFIG}
python cpe_minimizer.py -c ${CONFIG}


