# sent_sem_match_tf
Sentence semantics matching in tensorflow with simple software architecture.

## Usage


### build data
Run command in ./main directory, but you may can skip this process since there is preprocessed data in ./data/xx
```
python sent_sem_cdssm.py -c ../configs/atec/cdssm.json -s build_data
```

### train model
Run command in ./main directory
```
python sent_sem_cdssm.py -c ../configs/atec/cdssm.json
```



