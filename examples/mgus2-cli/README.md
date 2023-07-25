## Train Models from the Commandline

Using the `cli.py` script, it's relatively straightforward
to configure and train models directly from the commandline. 
Using the `mgus2` data and the configuration in `conf.yaml`,
a model can be trained with just two commands.

```{bash}
py cli.py fit -c conf.yaml
py cli.py test -c conf.yaml --ckpt_path lightning_logs/<…>
```
