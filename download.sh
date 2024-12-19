sudo apt update
sudo apt install python3
sudo apt update
sudo apt install pip
sudo apt update
sudo apt install git
sudo apt update
sudo apt install unzip
sudo apt update
sudo apt install python3-venv
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision numpy matplotlib tqdm psutil pillow gdown
# download training data:
gdown https://drive.google.com/uc?id=186yGq03kv1omNMyB9srZk6XJ8e35dLI8 -O img_align_celeba.zip
unzip img_align_celeba.zip
rm img_align_celeba.zip
git clone https://github.com/edwindn/alphazero
cd alphazero
