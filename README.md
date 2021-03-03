# PreProcessing-MillionDatasetsPlaylist

This repository releases a source code to pre-process the data files published together with the paper ***[Melon Playlist Dataset: a public dataset for audio-based playlist generation and music tagging.](https://arxiv.org/abs/2102.00201)*** by 	Andres Ferraro, Yuntae Kim, Soohyeon Lee, Biho Kim, Namjun Jo, Semi Lim, Suyon Lim, Jungtaek Jang, Sehwan Kim, Xavier Serra, Dmitry Bogdanov.

## Step 0. Install the Dependencies
After having clone this repository with 
```
git clone repo-name
```
we suggest creating e virtual environment install the required Python dependencies with the following commands

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Step 1. Download the Data
Download the dataset provided form the public [Melon link](https://arena.kakao.com/melon_dataset) placing them in the

```
.original_dataset/
```

## Step 2. Run the Script
To measure all the results it is necessary to run the following command in the terminal
```
python main.py
```

## Step 3. Output Files
The result files will be stored in
```
.melon/*
```
with the following formats:

* ```dataset.tsv``` has ```playlist_id [TAB] song_id [TAB] 1 [TAB] sequence-order```
* ```playlist_title.tsv``` has ```playlist_id [TAB] title```
* ... same pattern in the other files
