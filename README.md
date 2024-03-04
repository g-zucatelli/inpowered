# InPowered Data Challenge 

The Report of this Data Challenge is available at `Report_Guilherme_Zucatelli.pdf`. Please review the proposed solution and insights in available in the pdf file.

To run the code of the repository move to `Code/` and execute
> bash main.sh configs/baseline.json

or any other JSON in the `Code/configs/` directory.

The `main.sh` will execute a series of python functions of the complete pipeline and CPE minimization.

```
CONFIG=$1

python initialize.py -c ${CONFIG}
python data_processor.py -c ${CONFIG}
python trainer.py -c ${CONFIG}
python tester.py -c ${CONFIG}
python cpe_minimizer.py -c ${CONFIG}
```

At the end of execution you shold see an estimation for Bid and Budget that leads to the minimum CPE as:

```
-----------------------------------
Sample Number:           235
-----------------------------------
Target Bid:              0.3632
Target Budget:           538.08
-----------------------------------
Min CPE:                 0.4449
Engagement:              188
Media Spend:             83.94 
-----------------------------------
```

If a Media Spend constraint is included, this value the Bid Budget values care calculated next, for an example of 200 constraint:

```
# Max Media Spend Constraint: 200 #
-----------------------------------
Sample Number:           235
-----------------------------------
Target Bid:              0.3105
Target Budget:           502.61
-----------------------------------
Min CPE:                 0.4391
Engagement:              216
Media Spend:             200.00 (*)
-----------------------------------
(*) Constraint
```

In order to train the BERT-MLP solution a Google Colab version of the code is shared at <a href="https://colab.research.google.com/drive/1hPcLCdkYhoQa-tXSKBgZoW5q6rIHjapO?usp=sharing">BERTonColab.ipynb</a>, some path modifications are necessary.

