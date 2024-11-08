# Instalation guide:
sudo apt-get update && apt-get upgrade \
sudo apt-get install build-essential \

<create venv> \
pip install pyarrow==16.0.0 \
pip install pybind11 \
sudo apt-get install libfftw3-dev \
\
git clone https://repo.hca.bsc.es/gitlab/aingura-public/pybnesian \
cd PyBNesian \
python setup.py install \



<!-- # En pybnesian se han realizado las siguientes modificaciones:

- models/FourierNetwork.cpp y hpp --> las nuevas FSBNs
- factors/continuous/FBKernel.cpp y hpp --> es el equivalente a CKDE en las nuevas FourierNetwork (FSBNs)
- factors/factors.hpp --> Linea 120 incluye un nuevo template para "generic_new_factor". Ya que cv_likelihood.cpp realiza llamadas a la función "new_factor" 
durante el aprendizaje de estructuras con HC en SPBNs (factor CKDE), este sería su equivalente en las nuevas FourierNetwork (Factor FBKernel en cv_likelihood_fast)

- kde/fast_handler/* --> las funciones que realizan el data binning, calculo de la matriz de kernel y el fft-kde 
(fft-kde realiza F-1(F(c)*F(k)), donde c son pesos del binning y k los valores del kernel)

- kde/FastKDE.cpp y hpp --> Hace lo mismo que KDE pero con ciertas modificaciones para incluir el fastkde_handler, sparse matrices, etc. Es decir, la optimiciación  
de tiempos de computo del KDE. Principalmente cambia las funciones fit y logl_buffer()
** Todo lo que tenga que ver con sparse matrices trabaja con hashmaps **


- learning/scores/cv_likelihood_fast.cpp y hpp --> hace lo mismo que cv_likelihood pero para las FSBN incluyendo los parametros 
grid y linear (si el data binning es lineal o simple )

- utils_fft/* --> algunas funciones de sporte para adaptar ciertas funciones de numpy(python) a c++. Las uso desde en los scripts de "kde/fast_handler/" principalmente.

- pybindings/pybindings_learning/pybindings_scores.cpp --> Incluye el CVLikelihoodFT
- pybindings/pybindings_factors.cpp --> Incluye el nuevo factor FBKernel
- pybindings/pybindings_models.cpp --> Incluye la nueva FSBN bajo el nombre de FourierNetwork
- pybindings/pybindings_kde.cpp --> Incluye el nuevo FastKDE  -->






