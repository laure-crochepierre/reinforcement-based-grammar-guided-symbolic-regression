# Reinforcement Based Grammar Guided Symbolic Regression
Reinforcement based approach to symbolic regression using a Backus Naur Form grammar as action space. 


### Install 
This project uses Python3.8 and uses the libraries detailed in `requirements.txt` file. The suggested installation is the following : 
```bash 
virtualenv rbg2sr
source rbg2sr/bin/activate
pip install -r requirements.txt
```

### Command-line launch
An example of code to run the algorithm is given in the `src/examples` folder. It exposes the following arguments : 
```
-dataset DATASET, --d DATASET
                        Dataset name
  -batch_size BATCH_SIZE, --b BATCH_SIZE
                        Batch Size
  -max_horizon MAX_HORIZON, --m MAX_HORIZON
                        Max Horizon
  -min_horizon MIN_HORIZON, --n MIN_HORIZON
                        Min Horizon
  -hidden_dim HIDDEN_DIM, --h HIDDEN_DIM
                        Hidden Dim
  -embedding_dim EMBEDDING_DIM, --f EMBEDDING_DIM
                        Embedding dim
  -risk_eps RISK_EPS, --r RISK_EPS
                        Risk Epsilon
  -entropy_coeff ENTROPY_COEFF, --e ENTROPY_COEFF
                        Entropy Coefficient
  -learning_rate LEARNING_RATE, --l LEARNING_RATE
                        Learning rate
  -observe_parent OBSERVE_PARENT, --p OBSERVE_PARENT
                        Observe parent (True or False)
  -observe_siblings OBSERVE_SIBLINGS, --s OBSERVE_SIBLINGS
                        Observe siblings (True or False)
  -autoencoder AUTOENCODER, --a AUTOENCODER
                        Use autoencoder (True or False)
  -init_type INIT_TYPE, --i INIT_TYPE
                        Initialisation type (randint or zeros)
```

You can launch it as follows :
```bash
cd src/examples
python main_reinforce.py --d nguyen1
```

The `src/examples` folder also contains code for the symbolic regression benchmark and ablation study experiments. Baselines codes are given in `src/baselines`. 


### Contact 
Laure Crochepierre 
`laure.crochepierre@rte-france.com`
