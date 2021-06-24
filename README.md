# ADSP-TF
Trabalho final da disciplina Aplicações de Processamento Digital de Sinais

# Introdução
O aplicação tem por objetivo auxiliar a dirigibilidade de um veículo, fornecendo informações como o traçado da pista, correçãode posicionamento e identificação de pessoas e objetos por meio de uma rede neural

# Dependências
* Python >= 3
* NumPy
* OpenCV 

# Yolo repositório
Clone o repositório no diretório base deste:

```
cd ADSP-TF/
git clone https://github.com/rafixcs/Yolo_ADSPTF.git -b master
```

# Instalação
Criei o ambiente conda:

``` 
conda create -n yolo python=3.7 
conda activate yolo
```

Se possuir GPU NVidia Instale as dependencias do cuda e do pytorch:

``` 
conda install cudatoolkit=10.1 cudnn=7.6.0

conda install -c pytorch pytorch==1.7.0 torchvision cudatoolkit=10.1 
```

E por fim instale as dependencias da Yolo:

```
cd Yolo_ADSPTF
pip install  -r requirements.txt
```

# Baixe os pessos da rede

`https://brpucrs-my.sharepoint.com/:f:/g/personal/rafael_s_edu_pucrs_br/EsQXAaQDGpZDsptTaURmbk4BjVyT0V9-Oo637OyG6cviHw?e=kKKdLY`

# Execução do app

Na pasta base execute o seguinte comando para executar a detecção de linhas junto com a Yolo:

``` python -m src.app --path <path do arquivo de video .mp4> ```

Também é possével executar primeiro a detecção de objetos e posteriormente a detecção das linhas, tem performance melhor porém demora a inciar

``` python -m src.app --path <path do arquivo de video .mp4> --pre-detect ```



