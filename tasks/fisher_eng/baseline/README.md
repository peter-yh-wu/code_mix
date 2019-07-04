# Fisher English Pre-training Instructions

- ```cd baseline```

- Run ```python3 main.py``` to train LAS model on Fisher English data (make sure to use same args as Tagalog baseline run)

- Move the resulting ```model.ckpt``` into the ```Tagalog/baseline``` folder (i.e. wherever the model gets loaded in ```Tagalog/baseline/main.py```)

- ```cd Tagalog/baseline```

- Run ```python3 main.py``` with the same args as those used for the Fisher English run