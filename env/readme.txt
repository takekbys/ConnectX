このフォルダで
docker-compose up --build
を実行してdocker環境を構築

docker環境内で
apt-get install python-opengl ffmpeg freeglut3-dev xvfb
を実行して必要なものをインストール

xvfb-run -a jupyter notebook --ip=0.0.0.0 --allow-root --no-browser
でjupyter notebookを実行。普通の方法ではnotebook上でのenv.render()でエラーが出る

setup.ipynbを一通り実行して、stable-baselines3をインストール、gym、kaggle-environmentの動作確認
