.. discotime documentation master file, created by
   sphinx-quickstart on Fri May 19 12:26:12 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

discotime
=========

Discotime is a python package for discrete-time 
`survival analysis <https://en.wikipedia.org/wiki/Survival_analysis>`_
with competing risks using neural networks.
It builds on 
`PyTorch Lightning <https://lightning.ai/docs/pytorch/latest/>`_
to provide an easy-to-use interface,
but can still be customized to your heart's content. 
The packages contains an implementation of discrete time-to-event models
for neural networks (using PyTorch), different evaluation metrics,
and a couple of different competing risk datasets. 

.. container:: cell

   .. code:: python

      from lightning import Trainer
      from discotime.datasets import Mgus2, DataConfig
      from discotime.models import LitSurvModule, ModelConfig

      datamodule = Mgus2(DataConfig())
      survmodule = LitSurvModule(ModelConfig())
      trainer = Trainer(survmodule, datamodule)
      trainer.fit()

.. raw:: html

    <div class="row">
      <div class="col-lg-6 col-md-6 col-sm-6 col-xs-12 d-flex p-1">
        <div class="flex-fill tile">
          <a class="tile-link" href="installation.html">
            <h3 class="tile-title">Installation
            <i class="fas fa-download tile-icon"></i>
            </h3>
          </a>
          <div class="tile-desc">
			<p>Install with <code>pip</code> or <code>conda</code>,
            or get the development version from github.
            </p>
          </div>
        </div>
      </div>

      <div class="col-lg-6 col-md-6 col-sm-6 col-xs-12 d-flex p-1">
        <div class="flex-fill tile">
          <a class="tile-link" href="install.html">
            <h3 class="tile-title">Examples
            <i class="fas fa-book-open tile-icon"></i>
            </h3>
            <div class="tile-desc">
              <p>A collection of notebooks/vignettes showcasing
              how the package can be used.
              </p>
            </div>
          </a>
        </div>
      </div>

      <div class="col-lg-6 col-md-6 col-sm-6 col-xs-12 d-flex p-1">
        <div class="flex-fill tile">
          <a class="tile-link" href="reference/index.html">
            <h3 class="tile-title">API reference
            <i class="fas fa-cogs tile-icon"></i>
            </h3>
            <div class="tile-desc">
              <p>Detailed overview of package content.</p>
            </div>
          </a>
        </div>
      </div>

      <div class="col-lg-6 col-md-6 col-sm-6 col-xs-12 d-flex p-1">
        <div class="flex-fill tile">
          <a class="tile-link" href="install.html">
            <h3 class="tile-title">Contributing
            <i class="fas fa-code tile-icon"></i>
            </h3>
            <div class="tile-desc">
              <p>How do I contribute?</p>
            </div>
          </a>
        </div>
      </div>
    </div>

.. toctree::
   :hidden:
   :maxdepth: 1
   :titlesonly:

   
   installation
   examples/00_mgus2-analysis
   reference/index
   reference/modules
