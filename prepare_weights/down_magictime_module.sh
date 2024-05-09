git lfs install
git clone https://huggingface.co/BestWishYsh/MagicTime
rm -rf MagicTime/.git
mv MagicTime/Magic_Weights/* ckpts/Magic_Weights
rm -r MagicTime
