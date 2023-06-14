==================================
Getting started with ``discotime``
==================================

.. container:: cell

   .. code:: python

      import warnings

      import numpy as np
      import pandas as pd
      import torch

      from lightning import Trainer
      from lightning.pytorch.callbacks import EarlyStopping

      from discotime.datasets import Mgus2, DataConfig
      from discotime.models import LitSurvModule, ModelConfig

For this example, we will be using the ``Mgus2`` dataset from the
``survival`` pacakage in ``R``. The dataset contains natural history of
1341 patients with monoclonal gammopathy of undetermined significance
(MGUS) [1]_. In the ``discotime`` package, this dataset is supplied as a
``LightningDataModule`` that automatically handles downloading,
preprocessing, splitting into training and testing data, etc.

For initializiation of the data module, arguments are supplied in the
form of a ``DataConfig`` object as follows.

.. container:: cell

   .. code:: python

      mgus2dm = Mgus2(
          DataConfig(
              batch_size=128,
              n_time_bins=20,
              discretization_scheme="number",
          )
      )
      mgus2dm

   .. container:: cell-output cell-output-display

      ::

         <discotime.datasets.from_rstats.Mgus2 object at 0x7effb9c66770>
             n_features: 7, n_risks: 2, n_time_bins: 20

If we want to inspect the data, we can load the fit (train + val) and
test data by calling setup on the object. During normal use, this is
handled automatically by ``Lightning.Trainer``.

.. container:: cell

   .. code:: python

      mgus2dm.prepare_data()
      mgus2dm.setup()
      mgus2dm.dset_fit

   .. container:: cell-output cell-output-display

      ::

         <discotime.datasets.utils.SurvDataset at 0x7effa2047e50>

.. container:: cell

   .. code:: python

      mgus2dm.dset_fit[0]

   .. container:: cell-output cell-output-display

      ::

         SurvData(features=array([ 0.        ,  1.        ,  1.0580127 ,  0.87056947, -0.7417322 ,
                -0.4214596 , -1.1587092 ], dtype=float32), event_time_disc=4, event_status_disc=1, event_time_cont=29, event_status_cont=1)

The data consist of features (a dataframe) and labels. The features
should be fairly self explanatory, and the labels is a tuple of survival
times and event indicators. In the ``Discotime`` package and across the
other examples, an event/status of 0 indicates censoring.

To create the survival model, we use a different configuration object.

.. container:: cell

   .. code:: python

      model = LitSurvModule(
          ModelConfig(
              learning_rate=1e-3,
              n_hidden_units=15,
              n_sequential_blocks=5,
              use_skip_connections=True,
          )
      )

The last thing we need now is to instantiate the Lightning trainer. For
this example, we will only be using a single CPU, but with the lightning
Trainer it’s easy to scale training to multiple devices and/or GPUs.

.. container:: cell

   .. code:: python

      early_stopping = EarlyStopping(
          monitor="val_loss", min_delta=0.001, patience=30, mode="min"
      )

      warnings.filterwarnings("ignore", "GPU available")
      trainer = Trainer(
          accelerator="cpu",
          devices=1,
          max_epochs=3000,
          enable_checkpointing=False,
          enable_progress_bar=False,
          logger=False,
          reload_dataloaders_every_n_epochs=1,
          callbacks=[early_stopping],
      )

   .. container:: cell-output cell-output-stderr

      ::

         GPU available: True (cuda), used: False

   .. container:: cell-output cell-output-stderr

      ::

         TPU available: False, using: 0 TPU cores

   .. container:: cell-output cell-output-stderr

      ::

         IPU available: False, using: 0 IPUs

   .. container:: cell-output cell-output-stderr

      ::

         HPU available: False, using: 0 HPUs

.. container:: cell

   .. code:: python

      warnings.filterwarnings("ignore", ".*bottleneck")
      trainer.fit(model, mgus2dm)

   .. container:: cell-output cell-output-stderr

      ::

         /home/pechris/work/discotime/.venv/lib/python3.10/site-packages/lightning/pytorch/utilities/model_summary/model_summary.py:411: UserWarning: A layer with UninitializedParameter was found. Thus, the total number of parameters detected may be inaccurate.
           warning_cache.warn(

           | Name   | Type | Params
         --------------------------------
         0 | _model | Net  | 0     
         --------------------------------
         0         Trainable params
         0         Non-trainable params
         0         Total params
         0.000     Total estimated model params size (MB)

.. container:: cell

   .. code:: python

      __ = trainer.test(model, mgus2dm)

   .. container:: cell-output cell-output-display

      ::

         ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
         ┃        Test metric        ┃       DataLoader 0        ┃
         ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
         │      test_IPA_cause1      │    0.19919501087221964    │
         │      test_IPA_cause2      │   -0.08615107008082612    │
         │         test_loss         │     2.665034770965576     │
         └───────────────────────────┴───────────────────────────┘

.. [1]
   R. Kyle, T. Therneau, V. Rajkumar, J. Offord, D. Larson, M. Plevak,
   and L. J. Melton III, A long-terms study of prognosis in monoclonal
   gammopathy of undertermined significance. New Engl J Med, 346:564-569
   (2002).
