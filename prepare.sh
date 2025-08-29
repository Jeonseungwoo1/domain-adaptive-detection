mkdir -p data
cd .data

gdown 1LvxwCOfUa-OklIvBJhB8zJlochjJiPFS
gdown 1fa2L6oaPSjZ1_WqlTmIp6i2RbdR2y1Pw
gdown 1bZtVWcxxFrijE_ALvNPjH1MXIKio6BIr

names=(clipart watercolor comic) 
for name in "${names[@]}"
do
    unzip ${name}.zip
    rm ${name}.zip
done

cd ..
python utils/convertFormat.py